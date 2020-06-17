import os
import numpy as np
import time
from scipy import signal
from DSH import Config as cf
from DSH import MIfile as MI
from DSH import SharedFunctions as sf

def LoadFromConfig(ConfigFile, outFolder=None):
    """Loads a CorrMaps object from a config file like the one exported with CorrMaps.ExportConfiguration()
    
    Parameters
    ----------
    ConfigFile : full path of the config file to read
    outFolder : folder containing correlation maps. 
                if None, the value from the config file will be used
                if not None, the value from the config file will be discarded
                
    Returns
    -------
    a CorrMaps object with an "empty" image MIfile (containing metadata but no actual image data)
    """
    config = cf.Config(ConfigFile)
    if (outFolder is None):
        outFolder = config.Get('corrmap_parameters', 'out_folder')
    kernel_specs = config.ToDict(section='kernel')
    kernel_specs['padding'] = config.Get('kernel', 'padding', True, bool)
    return CorrMaps(MI.MIfile(None,config.ToDict(section='imgs_metadata')),\
                            outFolder, config.Get('corrmap_parameters', 'lags', [], int),\
                            kernel_specs, config.Get('corrmap_parameters', 'img_range', None, int),\
                            config.Get('corrmap_parameters', 'crop_roi', None, int))

class CorrMaps():
    """ Class to compute correlation maps from a MIfile """
    
    def __init__(self, MIin, outFolder, lagList, KernelSpecs, imgRange=None, cropROI=None):
        """Initialize CorrMaps
        
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
            self.Kernel['sigma'] = float(self.Kernel['sigma'])
            self.Kernel['cutoff'] = float(self.Kernel['cutoff'])
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
                'fps' : self.MIinput.GetFPS()*1.0/self.imgRange[2],
                'px_size' : self.MIinput.GetPixelSize()
                }
    
        self._corrmaps_loaded = False

    def __repr__(self):
        return '<CorrMaps class>'
    
    def __str__(self):
        str_res  = '\n|-----------------|'
        str_res += '\n| CorrMaps class: |'
        str_res += '\n|-----------------+---------------'
        str_res += '\n| MI Filename     : ' + str(self.MIinput.GetFilename())
        str_res += '\n| output folder   : ' + str(self.outFolder)
        str_res += '\n| lag times (' + str(self.numLags).zfill(2) + ')  : ' 
        lag_per_row = 20
        if (self.numLags <= lag_per_row):
            str_res += str(self.lagList)
        else:
            str_res += '['
            for i in range(1, self.numLags):
                if (i % lag_per_row == 0):
                    str_res += '\n|                    '
                str_res += str(self.lagList[i]) + ', '
            str_res = str_res[:-2] + ']'
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
        cf.ExportDict({'imgs_metadata' : self.MIinput.GetMetadata(),
                       'corrmap_metadata' : self.outMetaData,
                       'corrmap_parameters' : {'out_folder' : self.outFolder,
                                               'lags' : self.lagList,
                                               'img_range' : self.imgRange,
                                               'crop_roi' : self.cropROI
                                               },
                        'kernel' : self.Kernel
                       }, os.path.join(self.outFolder, 'CorrMapsConfig.ini'))

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
    
    def GetCorrMaps(self, openMIfiles=True, getAutocorr=True, check_lagtimes=False):
        """Searches for MIfile correlation maps
        
        Parameters
        ----------
        openMIfiles: if true, it opens all MIfiles for reading.
        getAutocorr: if True, returns d0 in the list of correlation maps
                    otherwise, returns None instead of the autocorrelation map
        check_lagtimes: if true, checks that the lagtimes extracted from the filenames match with self.lagList
        
        Returns
        -------
        corr_config: configuration file for correlation maps
        corr_mifiles: list of correlation maps, one per time delay
        lag_list: list of lagtimes
        """
        
        if not self._corrmaps_loaded:

            assert os.path.isdir(self.outFolder), 'Correlation map folder ' + str(self.outFolder) + ' not found.'
            config_fname = os.path.join(self.outFolder, 'CorrMapsConfig.ini')
            assert os.path.isfile(config_fname), 'Configuration file ' + str(config_fname) + ' not found'
            self.conf_cmaps = cf.Config(config_fname)
    
            all_cmap_fnames = sf.FindFileNames(self.outFolder, Prefix='CorrMap_d', Ext='.dat', Sort='ASC', AppendFolder=True)
            self.cmap_mifiles = []
            self.all_lagtimes = []
            for i in range(len(all_cmap_fnames)):
                cur_lag = sf.LastIntInStr(all_cmap_fnames[i])
                self.all_lagtimes.append(cur_lag)
                self.cmap_mifiles.append(MI.MIfile(all_cmap_fnames[i], self.conf_cmaps.ToDict(section='corrmap_metadata')))
                self.cmap_mifiles[-1].OpenForReading()
    
            # Check lagtimes for consistency
            if (check_lagtimes):
                print('These are all lagtimes. They should be already sorted and not contain 0:')
                print(self.all_lagtimes)
                for cur_lag in self.lagList:
                    if (cur_lag not in self.all_lagtimes):
                        print('WARNING: no correlation map found for lagtime ' + str(cur_lag))
                        
            self._corrmaps_loaded = True
        
        if (self.all_lagtimes[0]==0 and getAutocorr==False):
            return self.conf_cmaps, [None] + self.cmap_mifiles[1:], self.all_lagtimes
        else:
            return self.conf_cmaps, self.cmap_mifiles, self.all_lagtimes
    
    def GetCorrMapsNumber(self):
        _, mifiles, _ = self.GetCorrMaps()
        return len(mifiles)
    
    def GetCorrTimetrace(self, pxLocs, zRange=None, lagList=None, excludeLags=[], lagFlip=False, returnCoords=False,\
                         squeezeResult=True, readConsecutive=1, skipGet=False):
        """Returns (t, tau) correlations for a given set of pixels
        
        Parameters
        ----------
        pxLocs :          list of pixel locations, each location being a tuple (row, col)
        zRange :          range of time (or z) slices to sample
        lagList :         list of lagtimes. If None, all available lagtimes will be loaded,
                          except the ones eventually contained in excludeLags
        excludeLags:      if lagList is None, list of lagtimes to be excluded
        lagFlip :         if False: standard correlations will be returned
                          if True: correlations having the given time as second correlated time
                                   will be returned if available
                          if 'BOTH': both signs will be stitched together
        returnCoords :    if True, also returns the list of times, lagtimes and lagsigns
        squeezeResult :   if True, squeeze the output numpy array to remove dimensions if size along axis is 1
        readConsecutive : positive integer. Number of consecutive pixels to read starting from each pxLoc
                          if >1 the output will still be a 3D array, but the first dimension will be
                          len(pxLoc) x readConsecutive
        lock :            lock semaphore for multiprocessing i/o
        
        Returns
        -------
        res:              If only one pixel was asked, single 2D array: one row per time delay
                          Otherwise, 3D array, one matrix per pixel
        if returnCoords is True:
        tvalues:          list of image times
        lagList:          list of lagtimes, in image units. All elements are also contained in self.all_lagtimes
                          if lagFlip=='BOTH' lags are duplicated, first descending and then ascending
        lagFlip:          boolean value or list of boolean values indicating whether the lagtime is flipped
                          if signle boolean values, all values are interpreted as equally flipped
        """
        
        if not skipGet:
            self.GetCorrMaps()
        if lagList is None:
            lagList = self.all_lagtimes
        lagSet = (set(lagList) & set(self.all_lagtimes)) - set(excludeLags)
        if (lagFlip=='BOTH'):
            lagList_pos = list(lagSet).copy()
            lagList_pos.sort()
            listFlip_pos = list(np.ones_like(lagList_pos, dtype=bool)*False)
            lagList_neg = list(lagSet).copy()
            lagList_neg.sort(reverse=True)
            # 0 is only counted once
            if 0 in lagList_neg:
                lagList_neg.remove(0)
            listFlip_neg = list(np.ones_like(lagList_neg, dtype=bool)*True)
            lagList = lagList_neg+lagList_pos
            lagFlip = listFlip_neg+listFlip_pos
        else:
            lagList = list(lagSet)
            lagList.sort(reverse=lagFlip)
        
        tvalues = list(range(*self.cmap_mifiles[1].Validate_zRange(zRange)))
        res = self.GetCorrValues(pxLocs, tvalues, lagList, lagFlip=lagFlip, do_squeeze=squeezeResult, readConsecutive=readConsecutive, skipGet=skipGet)
        
        if returnCoords:
            return res, tvalues, lagList, lagFlip
        else:
            return res
        
    def GetCorrValues(self, pxLocs, tList, lagList, lagFlip=None, do_squeeze=True, readConsecutive=1, skipGet=False):
        """Get correlation values relative to a bunch of pixel location, time points and lags
        
        Parameters
        ----------
        pxLocs:           pixel location [row, col] or list of pixel locations
        tList :           time, in image units, or list of times
        lagList:          lagtime, in image units (must be present in self.all_lagtimes)
                          or list of lagtimes
        lagFlip:          boolean or list of boolean with same shape as lagList
                          indicates whether the time specified in tList is the smallest (lagFlip==False)
                          or the larges (lagFlip==True) of the two times that are correlated
        do_squeeze :      if True, squeeze the output numpy array to remove dimensions if size along axis is 1
        readConsecutive : positive integer. Number of consecutive pixels to read starting from each pxLoc
                          if >1 the output will still be a 3D array, but the first dimension will be
                          len(pxLoc) x readConsecutive
        """
        if not skipGet:
            self.GetCorrMaps()
        if (type(pxLocs[0]) not in [list, tuple, np.ndarray]):
            pxLocs = [pxLocs]
        if (type(tList) not in [list, tuple, np.ndarray]):
            tList = [tList]
        if (type(lagList) not in [list, tuple, np.ndarray]):
            lagList = [lagList]
        if lagFlip is None:
            lagFlip = np.ones_like(lagList, dtype=bool)
        elif (type(lagFlip) not in [list, tuple, np.ndarray]):
            lagFlip = np.ones_like(lagList, dtype=bool)*lagFlip
        if readConsecutive>1:
            res_shape = (len(pxLocs), len(lagList), len(tList), readConsecutive)
        else:
            res_shape = (len(pxLocs), len(lagList), len(tList))
        res = np.ones(res_shape)*np.nan
        for lidx in range(res.shape[1]):
            cur_midx = self.all_lagtimes.index(lagList[lidx])
            cur_mifile = self.cmap_mifiles[cur_midx]
            if cur_mifile is not None:
                for tidx in range(res.shape[2]):
                    if lagFlip[lidx]:
                        if tList[tidx] >= lagList[lidx]:
                            img_idx = tList[tidx]-int(lagList[lidx]/self.imgRange[2])
                        else:
                            img_idx = None
                    else:
                        img_idx = tList[tidx]
                    if img_idx is not None:
                        for pidx in range(res.shape[0]):
                            res[pidx, lidx, tidx] = cur_mifile._read_pixels(px_num=readConsecutive,\
                                       seek_pos=cur_mifile._get_offset(img_idx=img_idx, row_idx=pxLocs[pidx][0], col_idx=pxLocs[pidx][1]))
        if readConsecutive>1:
            res = np.moveaxis(res, -1, 1)
            new_shape = (res.shape[0]*res.shape[1], res.shape[2], res.shape[3])
            res.reshape(new_shape)
            
        if do_squeeze:
            return np.squeeze(res)
        else:
            return res
       