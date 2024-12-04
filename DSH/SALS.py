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

def ValidatePolarSlices(ROI_specs, imgShape=None):
    if ROI_specs is None:
        ROI_specs = [None, None]
    rSlices, aSlices = ROI_specs
    if rSlices is None:
        if imgShape is None:
            raise ValueError('Image shape must be specified to determine the range of radii')
        else:
            rSlices = [1, np.hypot(imgShape[0], imgShape[1])]
    if aSlices is None:
        aSlices = [-3.15, 3.15]
    return rSlices, aSlices

def GenerateROIs(ROI_specs, imgShape, centerPos, maskRaw=None):
    """ Generate ROI specs for SALS analysis

    Parameters
    ----------
    ROI_specs : None, or raw mask with ROI numbers, or couple [rSlices, aSlices].
                if raw mask:integer mask with ROI indexes for every pixel, 0-based.
                            Each pixel can only belong to one ROI.
                            Pixels belonging to no ROI are labeled with -1
                            Number of ROI is given by np.max(mask)
                None would correspond to [None, None]
                rSlices :   2D float array of shape (N, 2), or 1D float array of length N+1, or None. 
                            If 2D: i-th element will be (rmin_i, rmax_i), where rmin_i and rmax_i 
                                    delimit i-th annulus (values in pixels)
                            If 1D: i-th annulus will be delimited by i-th and (i+1)-th element of the list
                            If None: one single annulus will be generated, comprising the whole image
                aSlices :   angular slices, same structure as for rSlices. 
                            Here, angles are measured clockwise (y points downwards) starting from the positive x axis
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
    if (len(ROI_specs)==2):
        rSlices, aSlices = ValidatePolarSlices(ROI_specs, imgShape)
        ROIcoords = RP.MaskCoordsFromBoundaries(rSlices, aSlices, flatten_res=True)
        pxCoords = sf.PixelCoordGrid(shape=imgShape, center=centerPos, coords='polar')
        ROIs = RP.GenerateMasks(ROIcoords, pxCoords, common_mask=maskRaw)
        
        return ROIs, ROIcoords
    else:
        ROIs = ROI_specs
        r_map, a_map = sf.PixelCoordGrid(ROI_specs.shape, center=centerPos, coords='polar')
        r_min, r_max = RP.ROIEval(r_map, ROI_specs, [np.min, np.max])
        a_min, a_max = RP.ROIEval(a_map, ROI_specs, [np.min, np.max])
        rSlices, aSlices = [[r_min[i], r_max[i]] for i in range(len(r_min))], [[a_min[i], a_max[i]] for i in range(len(a_min))]
        return GenerateROIs([rSlices, aSlices], imgShape, centerPos, maskRaw=maskRaw)
    
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
    
    def __init__(self, MIin, centerPos, ROIslices=None, maskRaw=None, imgTimes=None, expTimes=[1], BkgCorr=None):
        """
        Initialize SALS

        Parameters
        ----------
        MIin : input MIfile or MIstack. It can be empty (i.e. initialized with metadata only)
        outFolder : output folder path. If the directory doesn't exist, it will be created
        centerPos : [float, float]. Position of transmitted beam [posX, posY], in pixels.
                    The center of top left pixel is [0,0], 
                    posX increases leftwards, posY increases downwards
        ROIslices : None, or couple [rSlices, aSlices].
                    None would correspond to [None, None]
                    rSlices :   2D float array of shape (N, 2), or 1D float array of length N+1, or None. 
                                If 2D: i-th element will be (rmin_i, rmax_i), where rmin_i and rmax_i 
                                        delimit i-th annulus (values in pixels)
                                If 1D: i-th annulus will be delimited by i-th and (i+1)-th element of the list
                                If None: one single annulus will be generated, comprising the whole image
                    aSlices :   angular slices, same structure as for rSlices. 
                                Here, angles are measured clockwise (y points downwards) starting from the positive x axis
                                None corresponds to full 2*pi angle
        maskRaw :   2D binary array with same shape as MIin.ImageShape()
                    True values (nonzero) denote pixels that will be included in the analysis,
                    False values (zeroes) will be excluded
                    If None, all pixels will be included.
                    Disregarded if ROIs is already a raw mask
        imgTimes :  None or float array of length Nimgs. i-th element will be the time of the image
                    if None, i-th time will be computed using the FPS from the MIfile Metadata
        expTimes :  list of floats
        BkgCorr :   Eventually, data for background correction: [DarkBkg, OptBkg, PDdata], where:
                    DarkBkg :    None, float array or MIfile with dark measurement from the camera. 
                                If None, dark subtraction will be skipped
                                If MIfile, all images from MIfile will be averaged and the raw result will be stored
                    OptBkg :     None, float array or MIfile with empty cell measurement. Sale as MIdark
                    PDdata :    2D float array of shape (Nimgs+2, 2), where Nimgs is the number of images.
                                i-th item is (PD0, PD1), where PD0 is the reading of the incident light, 
                                and PD1 is the reading of the transmitted light.
                                Only useful if DarkBkg or OptBkg is not None
        """
        
        # Initialize ROIproc
        RP.ROIproc.__init__(self, MIin, None, imgTimes=imgTimes, expTimes=expTimes)

        # Set ROIs
        self.ROIslices = ValidatePolarSlices(ROIslices, MIin.ImageShape())
        self.centerPos = centerPos
        self.px_mask = maskRaw
        ROImasks, ROIcoords = GenerateROIs(self.ROIslices, imgShape=MIin.ImageShape(), centerPos=self.centerPos, maskRaw=self.px_mask)
        self.SetROIs(ROImasks, ROIcoords)
        
        # SALS-specific initialization
        self._loadBkg(BkgCorr)
        
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

    def _loadBkg(self, BkgCorr):
        if BkgCorr is None:
            BkgCorr = [None, None, None]
        self.DarkBkg, self.OptBkg, self.PDdata = BkgCorr
        
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