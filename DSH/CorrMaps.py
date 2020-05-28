import numpy as np
import time
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
        lagList : list of lagtimes (in units of the image step specified in imgRange)
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
        if (self.Kernel['type']=='Gauss'):
            self.Kernel['size'] = int(self.Kernel['sigma']*self.Kernel['cutoff'])
        else:
            raise ValueError('Kernel type "' + str(self.Kernel['type']) + '" not supported')
        if (self.Kernel['padding']):
            self.Kernel['padw'] = self.Kernel['size']
            self.trimMargin = 0
        else:
            self.Kernel['padw'] = 0
            self.trimMargin = self.Kernel['size']
        self.outputShape = [self.inputShape[0], self.inputShape[1] - 2*self.trimMargin, self.inputShape[2] - 2*self.trimMargin]
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
        str_res += '\n|-----------------+------------'
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
            str_res += 'PADDING (width=' + str(self.Kernel['padw']) + ')'
        else:
            str_res += 'NO PADDING (trimming margin=' + str(self.trimMargin) + ')'
        str_res += '\n|----------------------------'
        return str_res

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
        Intensity = np.empty([self.inputShape[0], self.inputShape[1] + 2*self.Kernel['padw'], self.inputShape[2] + 2*self.Kernel['padw']])
        if (self.Kernel['padding']):
            # This contains ones eventually padded with zeros to properly average correlations
            # Without padding it would be just ones
            IntensityMask = np.pad(np.ones(self.inputShape[1:]), self.Kernel['padw'], 'constant')
            # This will contain kernel-averaged mask values
            # Without padding it would be just ones
            MaskNorm = np.empty([self.outputShape[1],self.outputShape[2]])
        # This will contain cross product of image intensities:
        # CrossProducts[i,j,k,l] = Intensity[j,k,l]*Intensity[j+lag[i],k,l]
        CrossProducts = np.empty([self.numLags+1, self.inputShape[0], self.inputShape[1] + 2*self.Kernel['padw'], self.inputShape[2] + 2*self.Kernel['padw']])
        # This will contain autocorrelation data ("d0")
        AutoCorr = np.empty(self.outputShape)
        # This will contain kernel-averaged intensity data
        AvgIntensity = np.empty_like(AutoCorr)
        
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
        if (self.Kernel['padding']):
            for i in range(self.outputShape[1]):
                for j in range(self.outputShape[2]):
                    MaskNorm[i][j] = np.sum(np.multiply(weights, IntensityMask[i:i+2*self.Kernel['size']+1, j:j+2*self.Kernel['size']+1]))
        
        if not silent:
            print('  STEP 4: Calculating and saving contrast...')
        for tidx in range(self.outputShape[0]):
            for ridx in range(self.outputShape[1]):
                for cidx in range(self.outputShape[2]):
                    temp_num = np.sum(np.multiply(weights, CrossProducts[0][tidx][ridx:ridx+2*self.Kernel['size']+1, cidx:cidx+2*self.Kernel['size']+1]))
                    AvgIntensity[tidx][ridx][cidx] = np.sum(np.multiply(weights, Intensity[tidx][ridx:ridx+2*self.Kernel['size']+1, cidx:cidx+2*self.Kernel['size']+1]))
                    if (self.Kernel['padding']):
                        temp_num = np.true_divide(temp_num, MaskNorm[ridx][cidx])
                        AvgIntensity[tidx][ridx][cidx] = np.true_divide(AvgIntensity[tidx][ridx][cidx], MaskNorm[ridx][cidx])
                    AutoCorr[tidx][ridx][cidx]   = temp_num/np.square(AvgIntensity[tidx][ridx][cidx])-1 
        MI.MIfile(sf.JoinPath(self.outFolder, 'CorrMap_d0.dat'), self.outMetaData).WriteData(AutoCorr)
        
        if not silent:
            print('  STEP 5: Normalizing and saving correlations...')
        if return_maps:
            res_4D = [np.asarray(AutoCorr, dtype=np.float32)]
        for lidx in range(self.numLags):
            cur_corr = np.empty_like(AutoCorr)
            for tidx in range(self.imgNumber-self.lagList[lidx]):
                for ridx in range(self.outputShape[1]):
                    for cidx in range(self.outputShape[2]):
                        temp_num = np.sum(np.multiply(weights, CrossProducts[lidx+1][tidx][ridx:ridx+2*self.Kernel['size']+1, cidx:cidx+2*self.Kernel['size']+1]))
                        if (self.Kernel['padding']):
                            temp_num = np.true_divide(temp_num, MaskNorm[ridx][cidx])
                        cur_corr[tidx][ridx][cidx] = np.true_divide(temp_num/(AvgIntensity[tidx][ridx][cidx]*AvgIntensity[tidx+self.lagList[lidx]][ridx][cidx])-1,\
                                                                    AutoCorr[tidx][ridx][cidx])
            MI.MIfile(sf.JoinPath(self.outFolder, 'CorrMap_d%s.dat' % self.lagList[lidx]), self.outMetaData).WriteData(cur_corr)        
            if return_maps:
                res_4D.append(np.asarray(cur_corr, dtype=np.float32))

        if not silent:
            print('Procedure completed in {0:.1f} seconds!'.format(time.time()-start_time))

        if return_maps:
            return res_4D
        else:
            return None