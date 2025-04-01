import os
import bisect
import math
import numpy as np
import logging

import DSH
from DSH import Config as cf
from DSH import MIfile as MI
from DSH import ROIproc as RP
from DSH import SharedFunctions as sf

def ValidatePolarSlices(ROI_specs, imgShape=None, rMax=None):
    """ Validate ROI specs for SALS analysis

    Parameters
    ----------
    ROI_specs : None, or raw mask with ROI numbers, or int, or couple [rSlices, aSlices].
                if raw mask:integer mask with ROI indexes for every pixel, 0-based.
                            Each pixel can only belong to one ROI.
                            Pixels belonging to no ROI are labeled with -1
                            Number of ROI is given by np.max(mask)
                None would correspond to [None, None]
                int would correspond to [int, None]
                rSlices :   2D float array of shape (N, 2), or 1D float array of length N+1, or int, or None. 
                            If 2D:  i-th element will be (rmin_i, rmax_i), where rmin_i and rmax_i 
                                     delimit i-th annulus (values in pixels)
                            If 1D:  i-th annulus will be delimited by i-th and (i+1)-th element of the list
                            If int: generate n annuli, linearly spaced
                            If None: one single annulus will be generated, comprising the whole image
                aSlices :   angular slices, same structure as for rSlices. 
                            Here, angles are measured clockwise (y points downwards) starting from the positive x axis
    rMax     :  maximum radius (only used if rSlices is None or int), or None
    imgShape :  shape of the binary image (num_rows, num_cols). Only used if rSlices is None or int, and rMax is None
    """
    if ROI_specs is None:
        ROI_specs = [None, None]
    elif not sf.IsIterable(ROI_specs):
        ROI_specs = [ROI_specs, None]
    rSlices, aSlices = ROI_specs
    if rSlices is None or not sf.IsIterable(rSlices):
        if rMax is None and imgShape is None:
            raise ValueError('Image shape or maximum radius must be specified to determine the range of radii')
        if rMax is None:
            rMax = np.hypot(imgShape[0], imgShape[1])
        if rSlices is None:
            rSlices = [1, rMax]
        else:
            rSlices = list(np.linspace(1, rMax, int(rSlices)+1, endpoint=True))
    if aSlices is None:
        aSlices = [-3.15, 3.15]
    return rSlices, aSlices

def GenerateROIs(ROI_specs, imgShape, centerPos, maskRaw=None):
    """ Generate ROI specs for SALS analysis

    Parameters
    ----------
    ROI_specs : None, or raw mask with ROI numbers, or int couple [rSlices, aSlices]. Check ValidatePolarSlices
    imgShape :  shape of the binary image (num_rows, num_cols)
    centerPos : center of polar coordinates.
    maskRaw :   2D binary array with same shape as MIin.ImageShape()
                True values (nonzero) denote pixels that will be included in the analysis,
                False values (zeroes) will be excluded
                If None, all pixels will be included.
                Disregarded if ROIs is already a raw mask
                
    Returns:
    ROI_masks: list of binary masks, one for each ROI and with same shape as image, marking image pixels belonging to the ROI
    ROIcoords: list of ROI coordinates in the form [r], [r,dr], [r,a], or [r, a, dr, da] depending on TODO
    """
    if ROI_specs is None:
        ROI_specs = [None, None]
    if not sf.IsIterable(ROI_specs):
        ROI_specs = [ROI_specs, None]
    if (len(ROI_specs)==2):
        pxCoords = sf.PixelCoordGrid(shape=imgShape, center=centerPos, coords='polar')
        rSlices, aSlices = ValidatePolarSlices(ROI_specs, rMax=np.max(pxCoords[0]))
        ROIcoords = RP.MaskCoordsFromBoundaries(rSlices, aSlices, flatten_res=True)
        ROIs = RP.GenerateMasks(ROIcoords, pxCoords, common_mask=maskRaw)
        
        return ROIs, ROIcoords
    else:
        ROIs = ROI_specs
        r_map, a_map = sf.PixelCoordGrid(ROI_specs.shape, center=centerPos, coords='polar')
        r_min, r_max = RP.ROIEval(r_map, ROI_specs, [np.min, np.max])
        a_min, a_max = RP.ROIEval(a_map, ROI_specs, [np.min, np.max])
        rSlices, aSlices = [[r_min[i], r_max[i]] for i in range(len(r_min))], [[a_min[i], a_max[i]] for i in range(len(a_min))]
        return GenerateROIs([rSlices, aSlices], imgShape, centerPos, maskRaw=maskRaw)
    
def RawSLS(image, centerPos, ROI_specs, maskRaw=None):
    """ compute average intensity of raw image on concentric rings 

    Parameters
    ----------
    image     : 2D image or 3D ndarray with list of images
    centerPos : center of polar coordinates.
    ROI_specs : None, or raw mask with ROI numbers, or int couple [rSlices, aSlices]. Check ValidatePolarSlices
    maskRaw :   2D binary array with same shape as MIin.ImageShape()
                
    Returns:
    ROI_avg: if image is 2D: 1D float array with ROI-averaged data
             if image is 3D: 2D float array, one image per row, one ROI per column.
    ROIcoords: 
    """
    im_shape = image.shape
    if len(im_shape) > 2:
        im_shape = im_shape[-2:]
    ROI_masks, ROIcoords = GenerateROIs(ROI_specs, imgShape=im_shape, centerPos=centerPos, maskRaw=maskRaw)
    ROI_avg, norm = RP.ROIAverage(image, ROI_masks, boolMask=True)
    return ROI_avg, ROIcoords
    
    
def LoadFromConfig(ConfigParams, runAnalysis=True, outputSubfolder='reproc', debugMode=False):
    """Loads a SALS object from a config file like the one exported in SALS.ExportConfiguration
    
    Parameters
    ----------
    ConfigParams : full path of the config file to read or dict or Config object
    runAnalysis  : if the config file has an Analysis section, 
                   set runAnalysis=True to run the analysis after initializing the object
    outputSubfolder : save analysis output in a subfolder of Analysis.out_folder from configuration
                    if None, directly save output in Analysis.out_folder
                
    Returns
    -------
    a SALS object
    """
    config = cf.LoadConfig(ConfigParams)
    folder_root = config.Get('General', 'folder', None, str)

    ROI_proc = RP.LoadFromConfig(config, runAnalysis=False)
    if config.HasSection('SALS'):
        exp_config = {'SALS':config.ToDict(section='SALS')}
        ctr_pos = config.Get('SALS', 'ctrpos', [0,0], int)
        r_slices = config.Get('SALS', 'rslices', None, float)
        a_slices = config.Get('SALS', 'aslices', None, float)
        roi_maskfile = sf.GetAbsolutePath(config.Get('SALS', 'raw_mask', None, str), root_path=folder_root)
        if roi_maskfile is None:
            mask_raw = None
        else:
            mask_raw = MI.ReadBinary(roi_maskfile, ROI_proc.MIinput.ImageShape(), 'B')
        SALSres = SALS(ROI_proc.MIinput, ctr_pos, [r_slices, a_slices], mask_raw, ROI_proc.imgTimes, ROI_proc.expTimes, BkgCorr=None)
    else:
        logging.warn('SALS.LoadFromConfig ERROR: no SALS section in configuration parameters. ROIproc object returned')
        exp_config = None
        SALSres = ROI_proc
    if debugMode:
        logging.info('SALS.LoadFromConfig running in debug mode')
        SALSres.DebugMode = debugMode
        
    if runAnalysis:
        SALSres.RunFromConfig(config, AnalysisSection='Analysis', OutputSubfolder=outputSubfolder, export_configparams=exp_config)
    return SALSres


class SALS(RP.ROIproc):
    """ Class to do small angle static and dynamic light scattering from a MIfile """
    
    def __init__(self, MIin, centerPos, ROIslices=None, maskRaw=None, imgTimes=None, expTimes=[1], PDdata=None, BkgCorr=None):
        """
        Initialize SALS

        Parameters
        ----------
        MIin : input MIfile or MIstack. It can be empty (i.e. initialized with metadata only)
        outFolder : output folder path. If the directory doesn't exist, it will be created
        centerPos : [float, float]. Position of transmitted beam [posX, posY], in pixels.
                    The center of top left pixel is [0,0], 
                    posX increases leftwards, posY increases downwards
        ROIslices : None, or int, or couple [rSlices, aSlices]. See ValidatePolarSlices for details
        maskRaw :   2D binary array with same shape as MIin.ImageShape()
                    True values (nonzero) denote pixels that will be included in the analysis,
                    False values (zeroes) will be excluded
                    If None, all pixels will be included.
                    Disregarded if ROIs is already a raw mask
        imgTimes :  None or float array of length Nimgs. i-th element will be the time of the image
                    if None, i-th time will be computed using the FPS from the MIfile Metadata
        expTimes :  list of floats
        PDdata :    None, 1D float array or list of 2 arrays [Iin, Itr], each of length equal to the number of images.
        BkgCorr :   Eventually, dict with data for background correction. A few keys:
                    ['Dark', 'Opt'] : dict with dark measurement and empty beam measurement. Each with a few keys: 
                                      - 'Iavg'     : 1D float array, i-th element is I averaged on i-th ROI, 
                                                     normalized the same way as the output of Iavg.dat
                                      - 'Iavg_norm': float, normalization factor that needs to be removed from Iavg
                                                     for it to be expressed as raw (8-bit grayscale) per unit exptime [ms]
                                                     If missing, Iavg will be interpreted as already expressed in these units
                                      - 'Iavg_raw' : 2D float array, element [i,j] is the average of i-th exposure time on j-th ROI
                                      - 'exptimes' : 1D float array with exposure times, in ms
                                      - 'PDdata'   : [Iin, Itr], each being a float value
        """
        
        # Initialize ROIproc
        RP.ROIproc.__init__(self, MIin, None, imgTimes=imgTimes, expTimes=expTimes, PDdata=PDdata, BkgCorr=BkgCorr)

        # Set ROIs
        self.ROIslices = ValidatePolarSlices(ROIslices, MIin.ImageShape())
        self.centerPos = centerPos
        self.px_mask = maskRaw
        ROImasks, ROIcoords = GenerateROIs(self.ROIslices, imgShape=MIin.ImageShape(), centerPos=self.centerPos, maskRaw=self.px_mask)
        self.SetROIs(ROImasks, ROIcoords)
        
    def __repr__(self):
        if (self.MIinput.IsStack()):
            return '<SALS object: MIstack >> ' + str(self.outFolder) + '>'
        else:
            return '<SALS object: MIfile (' + str(self.MIinput.FileName) + ') >> ' + str(self.outFolder) + '>'

    def __str__(self):
        str_res  = '\n|-----------------|'
        str_res += '\n|   SALS class:   |'
        str_res += '\n|-----------------+---------------'
        str_res += '\n| Input           : '
        if (self.MIinput.IsStack()):
            str_res += 'MIstack (' + self.MIinput.Count() + ' MIfiles)'
        else:
            str_res += 'MIfile (' + self.MIinput.FileName + ')'
        str_res += ', ' + str(self.MIinput.ImageNumber()) + ' images'
        str_res += '\n| Center position : ' + str(self.centerPos)
        str_res += '\n| ROIs            : ' + str(self.CountROIs()) + ' (' + str(self.CountValidROIs()) + ' valid, ' + str(self.CountEmptyROIs()) + ' empty)'
        str_res += '\n| Exposure times  : ' + str(self.NumExpTimes()) + ', from ' + str(self.expTimes[0]) + ' to ' + str(self.expTimes[-1])
        str_res += '\n|-----------------+---------------'
        return str_res
        
    def SetROIs(self, ROImasks, ROIcoords=None):
        if ROIcoords is not None:
            ROIcoords = [list(v) for v in ROIcoords]
        ROImetadata = {'coords' : ROIcoords, 'coord_names' : ['r[px]', 'theta[rad]'], 'box_margin' : 0}
        return RP.ROIproc.SetROIs(self, ROImasks, ROImetadata=ROImetadata)
    
    def GenerateROIs(self, ROI_specs, maskRaw=None):
        self.px_mask = maskRaw
        ROImasks, ROIcoords = GenerateROIs(ROI_specs, imgShape=self.MIinput.ImageShape(), centerPos=self.centerPos, maskRaw=maskRaw)
        return self.SetROIs(ROImasks, ROIcoords=ROIcoords)
    
    def GetSALSparams(self):
        res = {'ctrPos' : list(self.centerPos), 'rSlices' : list(self.ROIslices[0]), 'aSlices' : list(self.ROIslices[1])}
        return res
    
    def doDLS(self, saveFolder, lagtimes, export_configparams=None, **kwargs):
        sf.CheckCreateFolder(saveFolder)
        # save raw mask, get filename
        additional_params = {'SALS'    : self.GetSALSparams(), 
                             'General' : {'generated_by': 'SALS.doDLS'}}
        if self.px_mask is not None:
            raw_mask_filename = os.path.join(saveFolder, 'px_mask.raw')
            MI.WriteBinary(raw_mask_filename, self.px_mask, 'b')
            additional_params['SALS']['raw_mask'] = os.path.abspath(raw_mask_filename)

        
        if export_configparams is not None:
            additional_params = sf.UpdateDict(additional_params, export_configparams)
        return RP.ROIproc.doDLS(self, saveFolder, lagtimes=lagtimes, export_configparams=additional_params, **kwargs)
    
    def doSLS(self, saveFolder, export_configparams=None, **kwargs):
        sf.CheckCreateFolder(saveFolder)
        additional_params = {'SALS'    : self.GetSALSparams(), 
                             'General' : {'generated_by': 'SALS.doSLS'}}
        if self.px_mask is not None:
            raw_mask_filename = os.path.join(saveFolder, 'px_mask.raw')
            MI.WriteBinary(raw_mask_filename, self.px_mask, 'b')
            additional_params['SALS']['raw_mask'] = os.path.abspath(raw_mask_filename)

        if export_configparams is not None:
            additional_params = sf.UpdateDict(additional_params, export_configparams)
        return RP.ROIproc.doSLS(self, saveFolder, export_config=True, export_configparams=additional_params, **kwargs)