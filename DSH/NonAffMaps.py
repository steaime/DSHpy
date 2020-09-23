import os
import numpy as np
import scipy as sp
from scipy import signal
import logging

from DSH import MIfile as MI
from DSH import MIstack as MIs
from DSH import Config as cf
from DSH import SharedFunctions as sf

def _get_kw_from_config(conf=None, section='naffmap_parameters'):
    logging.info('Loading NonAffMaps initialization parameter from section of a Config object : ' + str(conf))
    # Read options for velocity calculation from DSH.Config object
    def_kw = {'qz_fw':0.0,\
              'qz_bk':1.0,\
              'trans_bk_matrix':[1,0,0,1],\
              'trans_bk_offset':[0,0],\
               't_range':None,\
               'cropROI':None,\
               'lag_range':None,\
               'smooth_kernel_specs':None,\
               'norm_range':None}
    if (conf is None):
        logging.debug('Default configuration returned')
        return def_kw
    else:
        ret_kw = {'qz_fw':conf.Get(section, 'qz_fw', def_kw['qz_fw'], float),\
                'qz_bk':conf.Get(section, 'qz_bk', def_kw['qz_bk'], float),\
                'trans_bk_matrix':conf.Get(section, 'trans_bk_matrix', def_kw['trans_bk_matrix'], float, silent=False),\
                'trans_bk_offset':conf.Get(section, 'trans_bk_offset', def_kw['trans_bk_offset'], float, silent=False),\
                't_range':conf.Get(section, 't_range', def_kw['t_range'], int),\
                'lag_range':conf.Get(section, 'lag_range', def_kw['lag_range'], int),\
                'smooth_kernel_specs':conf.Get(section, 'smooth_kernel_specs', def_kw['smooth_kernel_specs'], None),\
                'norm_range':conf.Get(section, 'norm_range', def_kw['norm_range'], int)}
        logging.debug('Return updated configuration: ' + str(ret_kw))
        return ret_kw

class NonAffMaps():
    """ Class to compute maps of out-of-plane mean square displacements comparing correlation maps from forward-scattered and back-scattered speckles
    (note: here 'forward-scattered' and 'back-scattered' are improperly used as a proxy for light scattered at small and large angles, respectively)
    """
    
    def __init__(self, cmaps_fw, cmaps_bk, outFolder, qz_fw=0.0, qz_bk=1.0, trans_bk_matrix=None, trans_bk_offset=None, t_range=None, cropROI=None, lag_range=None, smooth_kernel_specs=None, norm_range=None):
        """Initialize NonAffMaps
        
        Parameters
        ----------
        cmaps_fw :      CorrMaps object with information on available forward-scattered correlation maps, metadata and lag times
        cmaps_bk :      CorrMaps object with information on available backscattered correlation maps, metadata and lag times
        outFolder :     Folder where maps will be saved
        qz_fw :         z component of the scattering vector for forward-scattered light
        qz_bk :         z component of the scattering vector for backscattered light
        trans_bk_matrix : parameter for transforming the back-scattered correlation maps to overlap them with the forward-scattered ones
                        has to be either None (no transformation), a 2x2 matrix or a 4D vector that will be reshaped into a 2x2 matrix
        trans_bk_offset : either None or a 2D vector
        t_range :       restrict analysis to given time range [min, max, step].
                        if None, analyze full correlation maps
        cropROI :       analyze portion of the correlation maps
        lag_range :     restrict analysis to given range of lagtimes [min, max]
                        if None, analyze all available correlation maps
        smooth_kernel_specs : kernel for smoothing correlations prior to processing, to reduce noise impact
                        It has to be either None (no smoothing) or a dictionnary with kernel parameters
        norm_range :    range used to normalize correlation functions to filter out the spontaneous dynamics.
                        has to be either None (in which case no normalization is performed) or
                        [frame_start, frame_end, topleftx, toplefty, width, height]
                        note 1: even if t_range or cropROI are specified, coordinates are given
                                with respect to the first frame of the original MIfile
                        note 2: for the backscattering, the range is eventually compute after transformation
        """
        
        self.cmaps_fw = cmaps_fw
        self.cmaps_bk = cmaps_bk
        self.outFolder = outFolder
        self.qz_fw = qz_fw
        self.qz_bk = qz_bk
        self.trans_bk_matrix = trans_bk_matrix
        if self.trans_bk_matrix is None:
            self.trans_bk_matrix = [1,0,0,1]
        self.trans_bk_offset = trans_bk_offset
        if self.trans_bk_offset is None:
            self.trans_bk_offset = [0,0]
        self.t_range = t_range
        self.lag_range = lag_range
        self.cropROI = cropROI
        self.norm_range = norm_range
        self.smooth_kernel_specs = smooth_kernel_specs
        if self.smooth_kernel_specs is not None:
            if (self.smooth_kernel_specs['type']=='Gauss'):
                self.smooth_kernel_specs['sigma_xy'] = float(self.smooth_kernel_specs['sigma_xy'])
                self.smooth_kernel_specs['sigma_z'] = float(self.smooth_kernel_specs['sigma_z'])
                self.smooth_kernel_specs['cutoff'] = float(self.smooth_kernel_specs['cutoff'])
                self.smooth_kernel_specs['size_xy'] = int(self.smooth_kernel_specs['sigma_xy']*self.smooth_kernel_specs['cutoff'])
                self.smooth_kernel_specs['size_z'] = int(self.smooth_kernel_specs['sigma_z']*self.smooth_kernel_specs['cutoff'])
            elif (self.smooth_kernel_specs['type']=='Flat'):
                self.smooth_kernel_specs['size_xy'] = int(self.smooth_kernel_specs['size_xy'])
                self.smooth_kernel_specs['size_z'] = int(self.smooth_kernel_specs['size_z'])
            else:
                raise ValueError('Kernel type "' + str(self.Kernel['type']) + '" not supported')
        self.lagList = None
        self.naffMapStack = None

    def __repr__(self):
        return '<NonAffMaps class>'
    
    def __str__(self):
        str_res  = '\n|-------------------|'
        str_res += '\n| NonAffMaps class: |'
        str_res += '\n|-------------------+---------------'
        str_res += '\n| Forward CorrMaps  : ' + str(self.cmaps_fw.outFolder)
        str_res += '\n| Backward CorrMaps : ' + str(self.cmaps_bk.outFolder)
        if (self.lagList is None):
            str_res += '\n| lag times       : [not loaded]' 
        else:
            str_res += '\n| lag times (' + str(len(self.lagList)).zfill(2) + ')    : ' 
            lag_per_row = 20
            if (len(self.lagList) <= lag_per_row):
                str_res += str(self.lagList)
            else:
                str_res += '['
                for i in range(1, len(self.lagList)):
                    if (i % lag_per_row == 0):
                        str_res += '\n|                    '
                    str_res += str(self.lagList[i]) + ', '
                str_res = str_res[:-2] + ']'
        str_res += '\n| image range       : ' + str(self.t_range)
        str_res += '\n| crop ROI          : ' + str(self.cropROI)
        str_res += '\n| Transform matrix  : ' + str(self.trans_bk_matrix)
        str_res += '\n| Transform offset  : ' + str(self.trans_bk_offset)
        if (self.smooth_kernel_specs is None):
            str_res += '\n| Smoothing kernel  : NONE'
        else:
            str_res += '\n| Smoothing kernel  : ' + str(self.smooth_kernel_specs['type'])
            if (self.smooth_kernel_specs['type']=='Flat'):
                str_res += '\n| Kernel size (x,y,z): ' + str([self.smooth_kernel_specs['size_xy'],\
                                                             self.smooth_kernel_specs['size_xy'],\
                                                             self.smooth_kernel_specs['size_z']])
            else:
                str_res += ' - '
                for key in self.smooth_kernel_specs:
                    if key not in ['type', 'padw', 'padding', 'size']:
                        str_res += str(key) + '=' + str(self.smooth_kernel_specs[key]) + ', '
                str_res += '\n| Kernel size       : ' + str(self.smooth_kernel_specs['size']) + ' '
                if (self.smooth_kernel_specs['padding']):
                    str_res += 'PADDING (width=' + str(self.smooth_kernel_specs['size']) + ')'
                else:
                    str_res += 'NO PADDING (trimming margin=' + str(self.smooth_kernel_specs['size']) + ')'
        str_res += '\n|-------------------+---------------'
        return str_res
        
    def __del__(self):
        self.cmaps_fw.CloseMaps()
        self.cmaps_bk.CloseMaps()
        
    def ExportConfiguration(self):
        cf.ExportDict({'fw_corrmap_metadata' : self.cmaps_fw.outMetaData,
                       'bk_corrmap_metadata' : self.cmaps_bk.outMetaData,
                       'naffmap_parameters' : {'out_folder' : self.outFolder,
                                               'lag_range' : self.lag_range,
                                               'lags' : self.lagList,
                                               'img_range' : self.t_range,
                                               'crop_roi' : self.cropROI,
                                               'norm_range' : self.norm_range,
                                               'qz_fw' : self.qz_fw,
                                               'qz_bk' : self.qz_bk,
                                               'trans_bk_matrix' : self.trans_bk_matrix,
                                               'trans_bk_offset' : self.trans_bk_offset
                                               },
                        'smooth_kernel' : self.smooth_kernel_specs
                       }, os.path.join(self.outFolder, 'NaffMapsConfig.ini'))

    def LoadKernel(self, KernelSpecs):
        """Computes the convolution kernel for smoothing correlation functions
        """
        if (KernelSpecs['type']=='Flat'):
            ker3D = np.ones([2*KernelSpecs['size_z']+1, 2*KernelSpecs['size_xy']+1, 2*KernelSpecs['size_xy']+1])
        else:
            x = np.asarray(range(-KernelSpecs['size_xy'], KernelSpecs['size_xy']+1))
            y = np.asarray(range(-KernelSpecs['size_xy'], KernelSpecs['size_xy']+1))
            z = np.asarray(range(-KernelSpecs['size_z'], KernelSpecs['size_z']+1))
            grid = np.meshgrid(x,y,z)
            if (KernelSpecs['type']=='Gauss'):
                ker3D = np.multiply(np.exp(np.divide(np.square(grid[0])+np.square(grid[1]),-np.square(KernelSpecs['sigma_xy']))),\
                                    np.exp(np.divide(np.square(grid[2]),-np.square(KernelSpecs['sigma_z']))))
            else:
                raise ValueError('Kernel type "' + str(KernelSpecs['type']) + '" not supported')
        # Whatever kernel we are using, let's normalize so that weights has unitary integral
        ker3D = np.true_divide(ker3D, np.sum(ker3D))
        return ker3D
    
    def Compute(self):
        
        sf.CheckCreateFolder(self.outFolder)
        
        logging.info('NonAffMaps.Compute() started! Result will be saved in folder ' + str(self.outFolder))
        
        # Search for correlation map MIfiles, skip autocorrelation maps
        
        fw_mistack = self.cmaps_fw.GetCorrMaps(openMIfiles=True)
        bk_mistack = self.cmaps_bk.GetCorrMaps(openMIfiles=True)
        common_lags = list(set(fw_mistack.IdxList).intersection(bk_mistack.IdxList))
        if self.lag_range is None:
            if 0 in common_lags: common_lags.remove(0)
        else:
            if self.lag_range[1] < 0:
                self.lag_range[1] = np.max(common_lags)+1
            common_lags = [lag for lag in common_lags if (lag != 0 and lag >= self.lag_range[0] and lag <= self.lag_range[1])]
        
        self.lagList = common_lags
        
        # Export configuration
        self.ExportConfiguration()
        
        if self.trans_bk_matrix is not None:
            tr_matrix = np.reshape(np.asarray(self.trans_bk_matrix), (2,2))
            logging.debug('Backscattered correlation maps will be transformed using matrix ' + str(tr_matrix) + ' and offset ' + str(self.trans_bk_offset))
        
        # For each couple of correlation maps (with equal lagtime)
        for lidx in range(len(self.lagList)):
            
            logging.info('Now working on lagtime ' + str(lidx) + '/' + str(len(self.lagList)) + ' (d' + str(self.lagList[lidx]) + ')')
            
            fw_lidx = fw_mistack.IdxList.index(self.lagList[lidx])
            bk_lidx = bk_mistack.IdxList.index(self.lagList[lidx])
            
            # eventually compute normalization factors
            if self.norm_range is not None:
                fw_norm_factor = np.mean(fw_mistack.MIfiles[fw_lidx].Read(zRange=self.norm_range[:2], cropROI=self.norm_range[2:], closeAfter=False))
                if self.trans_bk_matrix is None and self.trans_bk_offset is None:
                    bk_norm_factor = np.mean(bk_mistack.MIfiles[bk_lidx].Read(zRange=self.norm_range[:2], cropROI=self.norm_range[2:], closeAfter=False))
                else:
                    bk_norm_data = bk_mistack.MIfiles[bk_lidx].Read(zRange=self.norm_range[:2], cropROI=None, closeAfter=False)
                    if len(bk_norm_data.shape)>2:
                        bk_norm_data = np.mean(bk_norm_data, axis=0)
                    logging.debug('shape before transformation: ' + str(bk_norm_data.shape))
                    bk_norm_data = sp.ndimage.affine_transform(bk_norm_data, tr_matrix, offset=self.trans_bk_offset,\
                                                      output_shape=bk_norm_data.shape, order=1, mode='constant', cval=1.0)
                    norm_cropROI = MI.ValidateROI(self.norm_range[2:], bk_norm_data.shape, replaceNone=True)
                    logging.debug('shape after transformation: ' + str(bk_norm_data.shape) + ' will be cropped with ROI ' + str(norm_cropROI))
                    bk_norm_factor = np.mean(bk_norm_data[norm_cropROI[1]:norm_cropROI[1]+norm_cropROI[3],\
                                                          norm_cropROI[0]:norm_cropROI[0]+norm_cropROI[2]])
                    bk_norm_data = None
            else:
                fw_norm_factor, bk_norm_factor = 1, 1
            
            logging.info('Normalization factors: ' + str(fw_norm_factor) + ' (front) and ' + str(bk_norm_factor) + ' (back)')
            
            # load, normalize and eventually smooth correlation maps.
            fw_data = np.true_divide(fw_mistack.MIfiles[fw_lidx].Read(zRange=self.t_range, cropROI=self.cropROI, closeAfter=True), fw_norm_factor)
            bk_data = np.true_divide(bk_mistack.MIfiles[bk_lidx].Read(zRange=self.t_range, cropROI=self.cropROI, closeAfter=True), bk_norm_factor)
            
            if self.smooth_kernel_specs is not None:
                Kernel3D = self.LoadKernel(self.smooth_kernel_specs)
                logging.debug('Smoothing with kernel with shape ' + str(Kernel3D.shape))
                fw_data = signal.convolve(fw_data, Kernel3D, mode='same')
                bk_data = signal.convolve(bk_data, Kernel3D, mode='same')
                    
            # transform backscattered images
            if self.trans_bk_matrix is not None:
                tr_matrix3D = np.asarray([[1,0,0],[0,tr_matrix[0,0],tr_matrix[0,1]],[0,tr_matrix[1,0],tr_matrix[1,1]]])
                tr_offset3D = np.asarray([0,self.trans_bk_offset[0],self.trans_bk_offset[1]])
                bk_data = sp.ndimage.affine_transform(bk_data, tr_matrix3D, offset=tr_offset3D,\
                                                      output_shape=fw_data.shape, order=1, mode='constant', cval=1.0)
        
            # sigma2 = ln(forward-scattering corr / backscattering corr) * 6 / (qz_bk^2 - qz_fw^2)
            sigma2 = np.log(np.true_divide(fw_data, bk_data)) * 6.0 / (self.qz_bk**2 - self.qz_fw**2)
            
            # For the first lagtime, generate and export metadata
            if (lidx==0):        
                out_meta = fw_mistack.MIfiles[fw_lidx].GetMetadata().copy()
                out_meta['hdr_len'] = 0
                out_meta['gap_bytes'] = 0
                out_meta['shape'] = list(sigma2.shape)
                if ('fps' in out_meta):
                    val_tRange = fw_mistack.MIfiles[fw_lidx].Validate_zRange(self.t_range)
                    out_meta['fps'] = float(out_meta['fps']) * 1.0 / val_tRange[2]
                exp_config = cf.Config()
                exp_config.Import(out_meta, section_name='MIfile')
                metadata_fname = os.path.join(self.outFolder, 'NAffMap_metadata.ini')
                exp_config.Export(metadata_fname)
                logging.info('Metadata exported to file ' + str(metadata_fname))
            
            # export data
            cur_fname = 'NaffMap_d' + str(self.lagList[lidx]).zfill(4) + '.dat'
            MI.MIfile(os.path.join(self.outFolder, cur_fname), metadata_fname).WriteData(sigma2)
            
            logging.info('Result saved to file ' + str(cur_fname))
            
            fw_mistack.MIfiles[fw_lidx].Close()
            bk_mistack.MIfiles[bk_lidx].Close()
    
    def GetNaffMaps(self, openMIfiles=True):
        """Searches for MIfile correlation maps
        
        Parameters
        ----------
        openMIfiles: if true, it opens all MIfiles for reading.
        getAutocorr: if True, returns d0 in the list of correlation maps
                    otherwise, returns None instead of the autocorrelation map
        
        Returns
        -------
        corr_config: configuration file for correlation maps
        corr_mifiles: list of correlation maps, one per time delay
        lag_list: list of lagtimes
        """
        
        if (self.naffMapStack is None):
            self.naffMapStack = MIs.LoadFolder(self.outFolder, config_fname=os.path.join(self.outFolder, 'NAffMap_metadata.ini'),\
                                                config_section='MIfile', mi_prefix='NaffMap_d', mi_ext='.dat', mi_sort='ASC', open_mifiles=openMIfiles)
    
        return self.naffMapStack
    
    def CloseMaps(self):
        self.naffMapStack.CloseAll()
        self.naffMapStack = None
    
    def GetNaffMapsNumber(self):
        assert (self.naffMapStack is not None), 'Correlation maps not loaded yet'
        return self.naffMapStack.Count()
