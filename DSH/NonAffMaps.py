import os
import numpy as np
import scipy as sp
from scipy import signal

from DSH import MIfile as MI
from DSH import Config as cf
from DSH import SharedFunctions as sf

def _get_kw_from_config(conf=None, section='naffmap_parameters'):
    # Read options for velocity calculation from DSH.Config object
    def_kw = {'qz_fw':0.0,\
              'qz_bk':1.0,\
              'trans_bk':[[1,0,0,1],[0,0]],\
               't_range':None,\
               'cropROI':None,\
               'lag_range':None,\
               'smooth_kernel_specs':None,\
               'norm_range':None}
    if (conf is None):
        return def_kw
    else:
        return {'qz_fw':conf.Get(section, 'qz_fw', def_kw['qz_fw'], float),\
                'qz_bk':conf.Get(section, 'qz_bk', def_kw['qz_bk'], float),\
                'trans_bk':[conf.Get(section, 'transf_matrix', def_kw['trans_bk'][0], float),\
                            conf.Get(section, 'transf_offset', def_kw['trans_bk'][1], float)],\
                't_range':conf.Get(section, 't_range', def_kw['t_range'], int),\
                'lag_range':conf.Get(section, 'lag_range', def_kw['lag_range'], int),\
                'smooth_kernel_specs':conf.Get(section, 'smooth_kernel_specs', def_kw['smooth_kernel_specs'], None),\
                'norm_range':conf.Get(section, 'norm_range', def_kw['norm_range'], int)}

class NonAffMaps():
    """ Class to compute maps of out-of-plane mean square displacements comparing correlation maps from forward-scattered and back-scattered speckles
    (note: here 'forward-scattered' and 'back-scattered' are improperly used as a proxy for light scattered at small and large angles, respectively)
    """
    
    def __init__(self, cmaps_fw, cmaps_bk, outFolder, qz_fw=0.0, qz_bk=1.0, trans_bk=None, t_range=None, cropROI=None, lag_range=None, smooth_kernel_specs=None, norm_range=None):
        """Initialize NonAffMaps
        
        Parameters
        ----------
        cmaps_fw :      CorrMaps object with information on available forward-scattered correlation maps, metadata and lag times
        cmaps_bk :      CorrMaps object with information on available backscattered correlation maps, metadata and lag times
        outFolder :     Folder where maps will be saved
        qz_fw :         z component of the scattering vector for forward-scattered light
        qz_bk :         z component of the scattering vector for backscattered light
        trans_bk :      parameter for transforming the back-scattered correlation maps to overlap them with the forward-scattered ones
                        has to be either None (no transformation) or [Matrix, Offset], where Matrix is a 2x2 matrix and Offset is a 2D vector
                        note: Matrix can also be flattened in a 4D array. In that case it will be reshaped
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
        self.trans_bk = trans_bk
        if self.trans_bk is None:
            self.trans_bk = [np.asarray([[1,0],[0,1]], dtype=float), np.asarray([0,0], dtype=float)]
        else:
            self.trans_bk = [np.reshape(np.asarray(self.trans_bk[0], dtype=float), (2,2)),\
                             np.asarray(self.trans_bk[1], dtype=float)]
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
                                               'trans_bk' : self.trans_bk
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
        
        # Search for correlation map MIfiles, skip autocorrelation maps
        fw_cmap_config, fw_cmap_mifiles, fw_cmap_lagtimes = self.cmaps_fw.GetCorrMaps(openMIfiles=True, getAutocorr=False)
        bk_cmap_config, bk_cmap_mifiles, bk_cmap_lagtimes = self.cmaps_fw.GetCorrMaps(openMIfiles=True, getAutocorr=False)
        common_lags = list(set(fw_cmap_lagtimes).intersection(bk_cmap_lagtimes))
        if self.lag_range is None:
            if 0 in common_lags: common_lags.remove(0)
        else:
            if self.lag_range[1] < 0:
                self.lag_range[1] = np.max(common_lags)+1
            common_lags = [lag for lag in common_lags if (lag != 0 and lag >= self.lag_range[0] and lag <= self.lag_range[1])]
        
        self.lagList = common_lags
        
        # Export configuration
        self.ExportConfiguration()
        
        # For each couple of correlation maps (with equal lagtime)
        for lidx in range(len(self.lagList)):
            fw_lidx = fw_cmap_lagtimes.index(self.lagList[lidx])
            bk_lidx = bk_cmap_lagtimes.index(self.lagList[lidx])
            
            # eventually compute normalization factors
            if self.norm_range is not None:
                fw_norm_factor = np.mean(fw_cmap_mifiles[fw_lidx].Read(zRange=self.norm_range[:2], cropROI=self.norm_range[2:], closeAfter=False))
                if self.trans_bk is None:
                    bk_norm_factor = np.mean(bk_cmap_mifiles[bk_lidx].Read(zRange=self.norm_range[:2], cropROI=self.norm_range[2:], closeAfter=False))
                else:
                    bk_norm_data = bk_cmap_mifiles[bk_lidx].Read(zRange=self.norm_range[:2], cropROI=None, closeAfter=False)
                    bk_norm_data = sp.ndimage.affine_transform(bk_norm_data, self.trans_bk[0], offset=self.trans_bk[1],\
                                                      output_shape=bk_norm_data.shape, order=1, mode='constant', cval=1.0)
                    norm_cropROI = MI.ValidateROI(self.norm_range[2:], bk_norm_data.shape, replaceNone=True)
                    bk_norm_factor = np.mean(bk_norm_data[norm_cropROI[1]:norm_cropROI[1]+norm_cropROI[3],\
                                                          norm_cropROI[0]:norm_cropROI[0]+norm_cropROI[2]])
                    bk_norm_data = None
            else:
                fw_norm_factor, bk_norm_factor = 1, 1
            
            # load, normalize and eventually smooth correlation maps.
            fw_data = np.true_divide(self.fw_cmap_mifiles[fw_lidx].Read(zRange=self.t_range, cropROI=self.cropROI, closeAfter=True), fw_norm_factor)
            bk_data = np.true_divide(self.bk_cmap_mifiles[bk_lidx].Read(zRange=self.t_range, cropROI=self.cropROI, closeAfter=True), bk_norm_factor)
            
            if self.smooth_kernel_specs is not None:
                Kernel3D = self.LoadKernel(self.smooth_kernel_specs)
                fw_data = signal.convolve(fw_data, Kernel3D, mode='same')
                bk_data = signal.convolve(bk_data, Kernel3D, mode='same')
    
            # transform backscattered images
            if self.trans_bk is not None:
                bk_data = sp.ndimage.affine_transform(bk_data, self.trans_bk[0], offset=self.trans_bk[1],\
                                                      output_shape=fw_data.shape, order=1, mode='constant', cval=1.0)
        
            # sigma2 = ln(forward-scattering corr / backscattering corr) * 6 / (qz_bk^2 - qz_fw^2)
            sigma2 = np.log(np.true_divide(fw_data, bk_data)) * 6.0 / (self.qz_bk**2 - self.qz_fw**2)
            
            # For the first lagtime, generate and export metadata
            if (lidx==0):        
                out_meta = fw_cmap_mifiles[fw_lidx].GetMetadata().copy()
                out_meta['hdr_len'] = 0
                out_meta['gap_bytes'] = 0
                out_meta['shape'] = list(sigma2.shape)
                if ('fps' in out_meta):
                    val_tRange = fw_cmap_mifiles[fw_lidx].Validate_zRange(self.t_range)
                    out_meta['fps'] = float(out_meta['fps']) * 1.0 / val_tRange[2]
                exp_config = cf.Config()
                exp_config.Import(out_meta, section_name='MIfile')
                exp_config.Export(os.path.join(self.outFolder, 'NAffMap_metadata.ini'))
            
            # export data
            MI.MIfile(os.path.join(self.outFolder, 'NaffMap_d' + str(self.lagList[lidx]).zfill(4) + '.dat'), self.outMetaData).WriteData(sigma2)
