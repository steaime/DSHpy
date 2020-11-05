import os
import numpy as np
import time
from scipy import signal
import DSH
from DSH import Kernel
from DSH import Config as cf
from DSH import MIfile as MI
from DSH import MIstack as MIs
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
    kernel_specs = DSH.Kernel.Kernel(config.ToDict(section='kernel'))
    return CorrMaps(MI.MIfile(None,config.ToDict(section='imgs_metadata')),\
                            outFolder, config.Get('corrmap_parameters', 'lags', [], int),\
                            kernel_specs, config.Get('corrmap_parameters', 'img_range', None, int),\
                            config.Get('corrmap_parameters', 'crop_roi', None, int))

class CorrMaps():
    """ Class to compute correlation maps from a MIfile """
    
    def __init__(self, MIin, outFolder, lagList, convKernel, imgRange=None, cropROI=None):
        """Initialize CorrMaps
        
        Parameters
        ----------
        MIin       : input multi image file (MIfile class) or stack of files (MIstack class)
                     both inputs can be used equivalently. If a stack is given, it will be considered
                     as a 't' type stack
        outFolder  : output folder path. If the directory doesn't exist, it will be created
        lagList    : list of lagtimes (in image units, regardless the step specified in imgRange)
        convKernel : DSH.Kernel object 
        imgRange   : range of images to be analyzed [start_idx, end_idx, step_idx]
                     if None, all images will be analyzed
        cropROI    : ROI to be analyzed: [topleftx, toplefty, width, height]
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
        self.Kernel = convKernel
        if (convKernel.Padding):
            self.outputShape = [self.imgNumber, self.inputShape[1], self.inputShape[2]]
        else:
            self.outputShape = [self.imgNumber, self.inputShape[1] - convKernel.Shape[0] + 1, self.inputShape[2] - convKernel.Shape[1] + 1]
        self.outMetaData = {
                'hdr_len' : 0,
                'shape' : self.outputShape,
                'px_format' : 'f',
                'fps' : self.MIinput.GetFPS()*1.0/self.imgRange[2],
                'px_size' : self.MIinput.GetPixelSize()
                }
    
        self.cmapStack = None

    def __repr__(self):
        return '<CorrMaps class>'
    
    def __str__(self):
        str_res  = '\n|-----------------|'
        str_res += '\n| CorrMaps class: |'
        str_res += '\n|-----------------+---------------'
        str_res += '\n| MI input        : ' + str(self.MIinput.__repr__())
        str_res += '\n| output folder   : ' + str(self.outFolder)
        str_res += '\n| lag times (' + str(self.numLags).zfill(2) + ')  : ' 
        lag_per_row = 20
        if (self.numLags <= lag_per_row):
            str_res += str(self.lagList)
        else:
            str_res += '['
            for i in range(0, self.numLags):
                if (i % lag_per_row == 0):
                    str_res += '\n|                    '
                str_res += str(self.lagList[i]) + ', '
            str_res = str_res[:-2] + ']'
        str_res += '\n| image range     : ' + str(self.imgRange)
        str_res += '\n| crop ROI        : ' + str(self.cropROI)
        str_res += '\n| Kernel          : ' + str(self.Kernel)
        str_res += '\n|-----------------+---------------'
        return str_res
    
    def __del__(self):
        self.CloseMaps()

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
        cf.ExportDict({'imgs_metadata' : self.MIinput.GetMetadata(section='MIfile'),
                       'corrmap_metadata' : self.outMetaData,
                       'corrmap_parameters' : {'out_folder' : self.outFolder,
                                               'lags' : self.lagList,
                                               'img_range' : self.imgRange,
                                               'crop_roi' : self.cropROI
                                               },
                        'kernel' : self.Kernel.ToDict(),
                       }, os.path.join(self.outFolder, 'CorrMapsConfig.ini'))

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
        ker2D = self.Kernel.ToMatrix()
        # This is to properly normalize correlations at the edges
        ConvNorm = signal.convolve2d(np.ones_like(Intensity[0]), ker2D, mode=self.Kernel.convolveMode, **self.Kernel.convolve_kwargs)
        # Now load all images we need
        self.MIinput.OpenForReading()
        for utidx in range(len(self.UniqueIdx)):  
            Intensity[utidx] = self.MIinput.GetImage(img_idx=self.UniqueIdx[utidx], cropROI=self.cropROI)
            AvgIntensity[utidx] = signal.convolve2d(Intensity[utidx], ker2D, mode=self.Kernel.convolveMode, **self.Kernel.convolve_kwargs)
            if (self.Kernel.convolveMode=='same'):
                AvgIntensity[utidx] = np.true_divide(AvgIntensity[utidx], ConvNorm)
        self.MIinput.Close()
        
        if not silent:
            print('  STEP 2: Computing contrast...')
        for tidx in range(self.outputShape[0]):
            AutoCorr[tidx] = signal.convolve2d(np.square(Intensity[self.imgIdx[tidx,0,0]]),\
                                               ker2D, mode=self.Kernel.convolveMode, **self.Kernel.convolve_kwargs)
            if (self.Kernel.Padding):
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
                                                  ker2D, mode=self.Kernel.convolveMode, **self.Kernel.convolve_kwargs)
                if (self.Kernel.Padding):
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
    
    def GetCorrMaps(self, openMIfiles=True):
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
        
        if (self.cmapStack is None):
            self.cmapStack = MIs.LoadFolder(self.outFolder, config_fname=os.path.join(self.outFolder, 'CorrMapsConfig.ini'),\
                                                config_section='corrmap_metadata', mi_prefix='CorrMap_d', mi_ext='.dat', mi_sort='ASC', open_mifiles=openMIfiles)
    
        return self.cmapStack
    
    def CloseMaps(self):
        if (self.cmapStack is not None):
            self.cmapStack.CloseAll()
            self.cmapStack = None
    
    def GetCorrMapsNumber(self):
        assert (self.cmapStack is not None), 'Correlation maps not loaded yet'
        return self.cmapStack.Count()
    
    def GetCorrTimetrace(self, pxLocs, zRange=None, lagList=None, excludeLags=[], lagFlip=False, returnCoords=False,\
                         squeezeResult=True, readConsecutive=1, skipGet=False):
        if not skipGet:
            self.GetCorrMaps()
        return self.cmapStack.GetTimetrace(pxLocs, zRange=zRange, idx_list=lagList, excludeIdxs=excludeLags, returnCoords=returnCoords,\
                         squeezeResult=squeezeResult, readConsecutive=readConsecutive, lagFlip=lagFlip, zStep=self.imgRange[2])
        
    def GetCorrValues(self, pxLocs, tList, lagList, lagFlip=None, do_squeeze=True, readConsecutive=1, skipGet=False):
        if not skipGet:
            self.GetCorrMaps()
        return self.cmapStack.GetValues(pxLocs, tList, idx_list=lagList, do_squeeze=do_squeeze, readConsecutive=readConsecutive, lagFlip=lagFlip, zStep=self.imgRange[2])
