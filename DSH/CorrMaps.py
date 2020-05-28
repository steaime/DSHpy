import numpy as np
if False:
    import SharedFunctions as sf
    import Config as cf
    import MIfile as MI
else:
    from DSH import SharedFunctions as sf
    from DSH import Config as cf
    from DSH import MIfile as MI

class CorrMaps():
    """ Class to compute correlation maps from a MIfile """
    
    def __init__(self, MIin, outFolder, lagList, KernelSpecs, imgRange=None, cropROI=None):
        """Initialize MIfile
        
        Parameters
        ----------
        MIin : input multi image file (MIfile class)
        outFolder : output folder path. If the directory doesn't exist, it will be created
        lagList : list of lagtimes (in image units)
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
        self.imgNumber = sf.CountRange(self.imgRange)
        self.cropROI = self.MIinput.ValidateROI(cropROI)
        self.Kernel = KernelSpecs
        if (self.cropROI is None):
            self.inputShape = [self.imgNumber, self.MIinput.ImageHeight(), self.MIinput.ImageWidth()]
        else:
            self.inputShape = [self.imgNumber, self.cropROI[3],self.cropROI[2]]
        self.Kernel['size'] = int(self.Kernel['sigma']*self.Kernel['cutoff'])
        if (self.Kernel['padding']):
            self.Kernel['padw'] = self.Kernel['size']
            self.trimMargin = 0
        else:
            self.Kernel['padw'] = 0
            self.trimMargin = self.Kernel['size']
        self.outputShape = [self.inputShape[0], self.inputShape[1] - 2*self.trimMargin, self.inputShape[2] - 2*self.trimMargin]
        self.outMetaData = {
                'hdr_len' : self.MIinput.HeaderSize(),
                'shape' : self.outputShape,
                'px_format' : 'f',
                'fps' : self.MIinput.GetFPS(),
                'px_size' : self.MIinput.GetPixelSize()
                }

    def Compute(self, silent=True):
        
        if not silent:
            print('Computing correlation maps:')
        sf.CheckCreateFolder(self.outFolder)
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
        conf.Export(sf.JoinPath(self.outFolder, 'CorrMapsConfig.ini'))
        
        if not silent:
            print('  STEP 1: Preparing memory...')
        # This will contain image data, eventually zero-padded
        Intensity = np.empty([self.inputShape[0], self.inputShape[1] + 2*self.Kernel['padw'], self.inputShape[2] + 2*self.Kernel['padw']], dtype=self.MIinput.DataType())
        # This contains ones eventually padded with zeros to properly average correlations
        # Without padding it is just ones
        IntensityMask = np.pad(np.ones(self.inputShape[1:], dtype=np.uint8), self.Kernel['padw'], 'constant')
        # This will contain kernel-averaged mask values
        # Without padding it is just ones
        MaskNorm = np.empty([self.outputShape[1],self.outputShape[2]], dtype=np.uint8)
        # This will contain cross product of image intensities:
        # CrossProducts[i,j,k,l] = Intensity[j,k,l]*Intensity[j+lag[i],k,l]
        if (self.MIinput.DataFormat() in ['c', 'b', 'B']):
            xprod_dtype = np.int16
        elif (self.MIinput.DataFormat() in ['h', 'H']):
            xprod_dtype = np.int32
        else:
            xprod_dtype = self.MIinput.DataType()
        CrossProducts = np.empty([self.numLags+1, self.inputShape[0], self.inputShape[1] + 2*self.Kernel['padw'], self.inputShape[2] + 2*self.Kernel['padw']], dtype=xprod_dtype)
        # This will contain autocorrelation data ("d0")
        AutoCorr = np.empty(self.outputShape, dtype=MI._data_types[self.outMetaData['px_format']])
        # This will contain kernel-averaged intensity data
        AvgIntensity = np.empty_like(AutoCorr, dtype=MI._data_types[self.outMetaData['px_format']])
        
        if not silent:
            print('  STEP 2: Loading images...')
        self.MIinput.OpenForReading()
        t_list = list(range(*self.imgRange))
        for tidx in range(self.imgNumber):  
            temp = self.MIinput.GetImage(img_idx=t_list[tidx], cropROI=self.cropROI)
            Intensity[tidx] = np.pad(temp, self.Kernel['padw'], 'constant')  
            CrossProducts[0][tidx] = np.square(Intensity[tidx])
        for lidx in range(self.numLags):
            for tidx in range(self.imgNumber-self.lagList[lidx]):
                CrossProducts[lidx+1][tidx] = np.multiply(Intensity[tidx], Intensity[tidx+self.lagList[lidx]])
        self.MIinput.Close()
                
        if not silent:
            print('  STEP 3: Calculating raw correlations...')
        x = np.asarray(range(-self.Kernel['size'], self.Kernel['size']+1))
        y = np.asarray(range(-self.Kernel['size'], self.Kernel['size']+1))
        grid = np.meshgrid(x,y)
        if (self.Kernel['type']=='Gauss'):
            weights = np.exp(np.divide(np.square(grid[0])+np.square(grid[1]),-np.square(self.Kernel['sigma'])))    
        else:
            raise ValueError('Kernel type "' + str(self.Kernel['type']) + '" not supported')
        for i in range(self.outputShape[1]):
            for j in range(self.outputShape[2]):
                MaskNorm[i][j] = np.sum(np.multiply(weights, IntensityMask[i:i+2*self.Kernel['size']+1, j:j+2*self.Kernel['size']+1]))
        
        if not silent:
            print('  STEP 4: Calculating and saving contrast...')
        for tidx in range(self.outputShape[0]):
            for ridx in range(self.outputShape[1]):
                for cidx in range(self.outputShape[2]):
                    temp_num = np.true_divide(np.sum(np.multiply(weights, CrossProducts[0][tidx][ridx:ridx+2*self.Kernel['size']+1,\
                                                                                        cidx:cidx+2*self.Kernel['size']+1])),\
                                              MaskNorm[ridx][cidx])
                    AvgIntensity[tidx][ridx][cidx] = np.true_divide(np.sum(np.multiply(weights, Intensity[tidx][ridx:ridx+2*self.Kernel['size']+1,\
                                                                                                         cidx:cidx+2*self.Kernel['size']+1])),\
                                                              MaskNorm[ridx][cidx])
                    AutoCorr[tidx][ridx][cidx]   = temp_num/np.square(AvgIntensity[tidx][ridx][cidx])-1 
        MI.MIfile(sf.JoinPath(self.outFolder, 'CorrMap_d0.dat'), self.outMetaData).WriteData(AutoCorr)
        
        if not silent:
            print('  STEP 5: Normalizing and saving correlations...')
        for lidx in range(self.numLags):
            cur_corr = np.empty_like(AutoCorr)
            for tidx in range(self.imgNumber-self.lagList[lidx]):
                for ridx in range(self.outputShape[1]):
                    for cidx in range(self.outputShape[2]):
                        temp_num = np.true_divide(np.sum(np.multiply(weights, CrossProducts[lidx+1][tidx][ridx:ridx+2*self.Kernel['size']+1,\
                                                                                                cidx:cidx+2*self.Kernel['size']+1])),\
                                                    MaskNorm[ridx][cidx])
                        cur_corr[tidx][ridx][cidx] = (temp_num/(AvgIntensity[tidx][ridx][cidx]*AvgIntensity[tidx+self.lagList[lidx]][ridx][cidx])-1)/AutoCorr[tidx][ridx][cidx]
            MI.MIfile(sf.JoinPath(self.outFolder, 'CorrMap_d%s.dat' % self.lagList[lidx]), self.outMetaData).WriteData(cur_corr)        
