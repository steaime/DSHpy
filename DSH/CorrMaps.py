import os
import numpy as np
import scipy as sp
import scipy.special as ssp
from scipy import stats, signal
import time
import bisect
from DSH import Config as cf
from DSH import MIfile as MI
from DSH import SharedFunctions as sf

class CorrMaps():
    """ Class to compute correlation maps from a MIfile """
    
    def __init__(self, MIin, outFolder, lagList, KernelSpecs, imgRange=None, cropROI=None):
        """Initialize MIfile
        
        Parameters
        ----------
        MIin : input multi image file (MIfile class)
        outFolder : output folder path. If the directory doesn't exist, it will be created
        lagList : list of lagtimes (in image units, regardless the step specified in imgRange)
        KernelSpecs : dictionary with kernel specifications 
                    for Gaussian kernels: {'type':'Gauss', 'sigma':std_dev, 'cutoff':sigma_cutoff, 'padding':true/false}
                    if 'padding' is True the output will have same shape as image (or cropped ROI)
                    otherwise it will have trimmed boundaries with thickness given by kernel cutoff
        imgRange : range of images to be analyzed [start_idx, end_idx, step_idx]
                    if None, all images will be analyzed
        cropROI : ROI to be analyzed: [topleftx, toplefty, width, height]
                    if None, full images will be analyzed
        """
        self.MIinput = MIin
        self.outFolder = outFolder
        self.lagList = lagList
        self.numLags = len(self.lagList)
        if (imgRange is None):
            self.imgRange = [0, -1, 1]
        else:
            self.imgRange = imgRange
        if (self.imgRange[1] < 0):
            self.imgRange[1] = self.MIinput.ImageNumber()
        if (len(self.imgRange) < 2):
            self.imgRange.append(1)
        self.imgNumber = len(list(range(*self.imgRange)))
        self.CalcImageIndexes()
        self.cropROI = self.MIinput.ValidateROI(cropROI)
        if (self.cropROI is None):
            self.inputShape = [len(self.UniqueIdx), self.MIinput.ImageHeight(), self.MIinput.ImageWidth()]
        else:
            self.inputShape = [len(self.UniqueIdx), self.cropROI[3],self.cropROI[2]]
        self.Kernel = KernelSpecs
        if (self.Kernel['type']=='Gauss'):
            self.Kernel['size'] = int(self.Kernel['sigma']*self.Kernel['cutoff'])
        else:
            raise ValueError('Kernel type "' + str(self.Kernel['type']) + '" not supported')
        if (self.Kernel['padding']):
            self.convolveMode = 'same'
            self.outputShape = [self.imgNumber, self.inputShape[1], self.inputShape[2]]
        else:
            self.convolveMode = 'valid'
            self.outputShape = [self.imgNumber, self.inputShape[1] - 2*self.Kernel['size'], self.inputShape[2] - 2*self.Kernel['size']]
        self.outMetaData = {
                'hdr_len' : 0,
                'shape' : self.outputShape,
                'px_format' : 'f',
                'fps' : self.MIinput.GetFPS(),
                'px_size' : self.MIinput.GetPixelSize()
                }

    def __repr__(self):
        return '<CorrMaps class>'
    
    def __str__(self):
        str_res  = '\n|-----------------|'
        str_res += '\n| CorrMaps class: |'
        str_res += '\n|-----------------+---------------'
        str_res += '\n| MI Filename     : ' + str(self.MIinput.GetFilename())
        str_res += '\n| output folder   : ' + str(self.outFolder)
        str_res += '\n| lag times (' + str(self.numLags).zfill(2) + ')  : ' + str(self.lagList)
        str_res += '\n| image range     : ' + str(self.imgRange)
        str_res += '\n| crop ROI        : ' + str(self.cropROI)
        str_res += '\n| Kernel          : ' + str(self.Kernel['type']) + ' - '
        for key in self.Kernel:
            if key not in ['type', 'padw', 'padding', 'size']:
                str_res += str(key) + '=' + str(self.Kernel[key]) + ', '
        str_res += '\n| Kernel size     : ' + str(self.Kernel['size']) + ' '
        if (self.Kernel['padding']):
            str_res += 'PADDING (width=' + str(self.Kernel['size']) + ')'
        else:
            str_res += 'NO PADDING (trimming margin=' + str(self.Kernel['size']) + ')'
        str_res += '\n|-----------------+---------------'
        return str_res

    def CalcImageIndexes(self):
        """Populates the array of image indexes to be loaded (based on self.imgRange e self.lagList)
            and the array that associates each couple of image (t, t+tau) to a couple of indexes (i, j) in self.UniqueIdx
        """
        # This will contain the image positions in Intensity that need to be correlated at time t and lagtime tau
        self.imgIdx = np.empty([self.imgNumber, self.numLags, 2], dtype=np.int32)
        # This is the list of "reference" images
        t1_list = list(range(*self.imgRange))
        # This is the list of unique image times that will populate the Intensity array
        self.UniqueIdx = []
        # Let's set the first self.imgNumber elements of this list with t1 image indexes:
        for cur_t in t1_list:
            self.UniqueIdx.append(cur_t)
        # Now let's populate it with eventual t2 image indexes that are not already in there:
        for tidx in range(self.imgNumber):
            cur_t1 = t1_list[tidx]
            for lidx in range(self.numLags):
                self.imgIdx[tidx, lidx, 0] = tidx
                cur_t2 = cur_t1 + self.lagList[lidx]
                if cur_t2 < self.MIinput.ImageNumber():
                    if cur_t2 not in self.UniqueIdx:
                        self.UniqueIdx.append(cur_t2)
                    self.imgIdx[tidx, lidx, 1] = self.UniqueIdx.index(cur_t2)
                else:
                    self.imgIdx[tidx, lidx, 1] = -1
    
    def ExportConfiguration(self):
        dict_config = {'mi_input' : self.MIinput.GetMetadata(),
                       'mi_output' : self.outMetaData,
                       'parameters' : {'out_folder' : self.outFolder,
                                   'lags' : self.lagList,
                                   'img_range' : self.imgRange,
                                   'crop_roi' : self.cropROI
                                   },
                        'kernel' : self.Kernel
                       }
        conf = cf.Config()
        conf.Import(dict_config, section_name=None)
        conf.Export(os.path.join(self.outFolder, 'CorrMapsConfig.ini'))

    def LoadKernel(self, KernelSpecs):
        """Computes the convolution kernel for ROI computation
        """
        x = np.asarray(range(-KernelSpecs['size'], KernelSpecs['size']+1))
        y = np.asarray(range(-KernelSpecs['size'], KernelSpecs['size']+1))
        grid = np.meshgrid(x,y)
        if (KernelSpecs['type']=='Gauss'):
            ker2D = np.exp(np.divide(np.square(grid[0])+np.square(grid[1]),-np.square(KernelSpecs['sigma'])))
        else:
            raise ValueError('Kernel type "' + str(KernelSpecs['type']) + '" not supported')
        # Whatever kernel we are using, let's normalize so that weights has unitary integral
        ker2D = np.true_divide(ker2D, np.sum(ker2D))
        return ker2D

    def Compute(self, silent=True, return_maps=False):
        """Computes correlation maps
        
        Parameters
        ----------
        silent : bool. If set to False, procedure will print to output every time steps it goes through. 
                otherwise, it will run silently
        return_maps : bool. If set to True, procedure will return the array with correlation maps.
                Warning: no memory check is done when this happens, so be aware of memory consumption
                
        Returns
        -------
        res_4D : list of correlation maps (np.float32), if return_maps==True
        """
        
        if not silent:
            start_time = time.time()
            print('Computing correlation maps:')
        sf.CheckCreateFolder(self.outFolder)
        self.ExportConfiguration()
        
        if not silent:
            print('  STEP 1: Loading images and computing average intensity...')
        # This will contain image data, eventually zero-padded
        Intensity = np.empty(self.inputShape)
        # This will contain kernel-averaged intensity data
        AvgIntensity = np.empty([self.inputShape[0], self.outputShape[1], self.outputShape[2]])
        # This will contain autocorrelation data ("d0")
        AutoCorr = np.empty(self.outputShape)
        # 2D Kernel to convolve to spatially average images
        ker2D = self.LoadKernel(self.Kernel)
        # This is to properly normalize correlations at the edges
        ConvNorm = signal.convolve2d(np.ones_like(Intensity[0]), ker2D, mode=self.convolveMode, boundary='fill', fillvalue=0)
        # Now load all images we need
        self.MIinput.OpenForReading()
        for utidx in range(len(self.UniqueIdx)):  
            Intensity[utidx] = self.MIinput.GetImage(img_idx=self.UniqueIdx[utidx], cropROI=self.cropROI)
            AvgIntensity[utidx] = signal.convolve2d(Intensity[utidx], ker2D, mode=self.convolveMode, boundary='fill', fillvalue=0)
            if (self.convolveMode=='same'):
                AvgIntensity[utidx] = np.true_divide(AvgIntensity[utidx], ConvNorm)
        self.MIinput.Close()
        
        if not silent:
            print('  STEP 2: Computing contrast...')
        for tidx in range(self.outputShape[0]):
            AutoCorr[tidx] = signal.convolve2d(np.square(Intensity[self.imgIdx[tidx,0,0]]),\
                                               ker2D, mode=self.convolveMode, boundary='fill', fillvalue=0)
            if (self.Kernel['padding']):
                AutoCorr[tidx] = np.true_divide(AutoCorr[tidx], ConvNorm)
            AutoCorr[tidx] = np.subtract(np.true_divide(AutoCorr[tidx], np.square(AvgIntensity[tidx])),1)
        MI.MIfile(os.path.join(self.outFolder, 'CorrMap_d0.dat'), self.outMetaData).WriteData(AutoCorr)
        
        if not silent:
            print('  STEP 3: Computing correlations...')
        if return_maps:
            res_4D = [np.asarray(AutoCorr, dtype=np.float32)]
        for lidx in range(self.numLags):
            if not silent:
                print('     ...lag ' + str(self.lagList[lidx]))
            CorrMap = np.empty_like(AutoCorr)
            for tidx in range(self.imgNumber-self.lagList[lidx]):
                CorrMap[tidx] = signal.convolve2d(np.multiply(Intensity[self.imgIdx[tidx,lidx,0]], Intensity[self.imgIdx[tidx,lidx,1]]),\
                                                  ker2D, mode=self.convolveMode, boundary='fill', fillvalue=0)
                if (self.Kernel['padding']):
                    CorrMap[tidx] = np.true_divide(CorrMap[tidx], ConvNorm)
                CorrMap[tidx] = np.true_divide(np.subtract(np.true_divide(CorrMap[tidx],\
                                                                           np.multiply(AvgIntensity[self.imgIdx[tidx,lidx,0]],\
                                                                                       AvgIntensity[self.imgIdx[tidx,lidx,1]])),\
                                                            1),\
                                                AutoCorr[tidx])
            MI.MIfile(os.path.join(self.outFolder, 'CorrMap_d' + str(self.lagList[lidx]).zfill(4) + '.dat'), self.outMetaData).WriteData(CorrMap)
            if return_maps:
                res_4D.append(np.asarray(CorrMap, dtype=np.float32))

        if not silent:
            print('Procedure completed in {0:.1f} seconds!'.format(time.time()-start_time))

        if return_maps:
            return res_4D
        else:
            return None

    def _dr_g_relation(zProfile='Parabolic'):
        """Generate a lookup table for inverting correlation function
        """
        if (zProfile=='Parabolic'):
            dr_g = np.zeros([2,4500])
            dr_g[0] = np.linspace(4.5, 0.001, num=4500)
            norm = 4/np.pi
            dr_g[1] = (np.square(abs(ssp.erf(sp.sqrt(-dr_g[0]*1j))))/dr_g[0])/norm
            dr_g[1][4499] = 1 #to prevent correlation to be higher than highest value in array
            return dr_g
        else:
            raise ValueError(str(zProfile) + 'z profile not implemented yet')

    def _invert_monotonic(data, _lut):
        """Invert monotonic function based on a lookup table
        """
        res = np.zeros(len(data))
        for i in range(len(data)):
            index = bisect.bisect_left(_lut[1], data[i])
            res[i] = _lut[0][index]
        return res

    def ComputeVelocities(self, zProfile='Parabolic', useBuffer=True, silent=True):
        """Converts correlation data into velocity data assuming a given velocity profile along z
        
        Parameters
        ----------
        zProfile : 'Parabolic'|'Linear'. For the moment, only parabolic has been developed
        useBuffer : if True, first load all correlation data (not implemented yet)
                    otherwise, read images that you need one at a time
        """
        if (os.path.isdir(self.outFolder)):
            config_fname = os.path.join(self.outFolder, 'CorrMapsConfig.ini')
            if (os.path.isfile(config_fname)):
                conf_cmaps = cf.Config(config_fname)
            else:
                raise IOError('Configuration file CorrMapsConfig.ini not found in folder ' + str(self.outFolder))
        else:
            raise IOError('Correlation map folder ' + str(self.outFolder) + ' not found.')

        if not silent:
            start_time = time.time()
            print('Computing velocity maps:')
            cur_progperc = 0
            prog_update = 10
            
        all_cmap_fnames = sf.FindFileNames(self.outFolder, Prefix='CorrMap_d', Ext='.dat', Sort='ASC')
        cmap_mifiles = []
        for i in range(len(all_cmap_fnames)):
            cmap_mifiles.append(MI.MIfile(all_cmap_fnames[i], conf_cmaps.ToDict(section='mi_output')))

        # Check lagtimes for consistency
        all_lagtimes = sf.ExtractIndexFromStrings(all_cmap_fnames, index_pos=-1)
        print('These are all lagtimes. They should be already sorted:')
        print(all_lagtimes)
        for cur_lag in self.lagList:
            if (cur_lag not in all_lagtimes):
                print('WARNING: no correlation map found for lagtime ' + str(cur_lag))
        
        # Prepare memory
        cmap_shape = conf_cmaps.Get('mi_output', 'shape', None, int)
        dr_g = self._dr_g_relation(zProfile)
        cutoff_corr = 0.2
        vmap = np.zeros(cmap_shape)
        verr = np.zeros(cmap_shape)
        
        for tidx in range(cmap_shape[0]):
            
            # find compatible lag indexes
            lag_idxs = []  # index of lag in all_lagtimes list
            t1_idxs = []   # tidx if tidx is t1, tidx-lag if tidx is t2
            # From largest to smallest, 0 excluded
            for lidx in range(len(all_lagtimes)-1, 0, -1):
                if (all_lagtimes[lidx] <= tidx):
                    t1_idxs.append(tidx-all_lagtimes[lidx])
                    lag_idxs.append(lidx)
            # From smallest to largest, 0 included
            for lidx in range(len(all_lagtimes)):
                if (tidx+all_lagtimes[lidx] < cmap_shape[0]):
                    t1_idxs.append(tidx)
                    lag_idxs.append(lidx)
            
            # Populate arrays and masked arrays
            cur_cmaps = np.zeros([len(lag_idxs), cmap_shape[1], cmap_shape[2]])
            cur_lags = np.zeros([len(lag_idxs), cmap_shape[1], cmap_shape[2]])
            for lidx in range(len(lag_idxs)):
                cur_cmaps[lidx] = cmap_mifiles[lag_idxs[lidx]].GetImage(t1_idxs[lidx])
                cur_lags[lidx] = np.ones([cmap_shape[1], cmap_shape[2]])*all_lagtimes[lidx]
            cur_cmaps = np.ma.masked_less(cur_cmaps, cutoff_corr)
            cur_lags = np.ma.masked_array(cur_lags, cur_cmaps.mask)
            
            for ridx in range(cmap_shape[1]):
                for cidx in range(cmap_shape[2]):
                    dr = self._invert_monotonic(cur_cmaps[:,ridx,cidx].compressed(), dr_g)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(cur_lags[:,ridx,cidx].compressed(), dr)
                    vmap[tidx,ridx,cidx] = slope
                    verr[tidx,ridx,cidx] = std_err

            if not silent:
                cur_p = (tidx+1)*100/cmap_shape[0]
                if (cur_p > cur_progperc+prog_update):
                    cur_progperc = cur_progperc+prog_update
                    print('   {0}% completed...'.format(cur_progperc))

        if not silent:
            print('   ...done! Now exporting files'.format(cur_progperc))
        MI.MIfile(os.path.join(self.outFolder, '_vMap.dat'), self.outMetaData).WriteData(vmap)
        MI.MIfile(os.path.join(self.outFolder, '_vErr.dat'), self.outMetaData).WriteData(verr)
        
        if not silent:
            print('Procedure completed in {0:.1f} seconds!'.format(time.time()-start_time))
