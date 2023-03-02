import os
import bisect
import math
import numpy as np
import logging
import DSH
from DSH import ROIproc as RP
from DSH import SharedFunctions as sf

def ValidatePolarSlices(ROI_specs, imgShape=None):
    if ROI_specs is None:
        ROI_specs = [None, None]
    rSlices, aSlices = ROI_specs
    if rSlices is None:
        if imgShape is None:
            raise ValueError('Image shape must be specified to retermine the range of radii')
        else:
            rSlices = [0, np.hypot(imgShape[0], imgShape[1])]
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
    
def LoadFromConfig(ConfigFile, input_sect='input', outFolder=None):
    """Loads a SALS object from a config file like the one exported with VelMaps.ExportConfig()
    
    Parameters
    ----------
    ConfigFile : full path of the config file to read
    outFolder : folder containing velocity and correlation maps. 
                if None, the value from the config file will be used
                if not None, the value from the config file will be discarded
                
    Returns
    -------
    a SALS object, eventually with an "empty" image MIfile (containing metadata but no actual image data)
    """
    
    
    ###################
    ### TODO: CHANGE THIS
    ###################
    
    
    
    
    
    config = cf.Config(ConfigFile)
    froot = config.Get('global', 'root', '', str)
    miin_fname = config.Get(input_sect, 'mi_file', None, str)
    miin_meta_fname = config.Get(input_sect, 'meta_file', None, str)
    input_stack = False
    if (miin_fname is not None):
        # if miin_fname is a string, let's use a single MIfile as input.
        # otherwise, it can be a list: in that case, let's use a MIstack as input
        if (isinstance(miin_fname, str)):
            miin_fname = os.path.join(froot, miin_fname)
            input_stack = False
        else:
            input_stack = True
            for i in range(len(miin_fname)):
                miin_fname[i] = os.path.join(froot, miin_fname[i])
    if (miin_meta_fname is not None):
        miin_meta_fname = os.path.join(froot, miin_meta_fname)
    elif input_stack:
        logging.error('SALS.LoadFromConfig ERROR: medatada filename must be specified when loading a MIstack')
        return None
    else:
        miin_meta_fname = os.path.splitext(miin_fname)[0] + '_metadata.ini'
    if input_stack:
        mifile_info = 'MIstack ' + str(miin_fname)
        MIin = MIs.MIstack(miin_fname, miin_meta_fname, Load=True, StackType='t')
    else:
        mifile_info = 'MIfile ' + str(miin_fname)
        MIin = MI.MIfile(miin_fname, miin_meta_fname)
    logging.debug('SALS.LoadFromConfig loading ' + str(mifile_info) + ' (metadata filename: ' + str(miin_meta_fname) + ')')
    ctrPos = config.Get('SALS_parameters', 'center_pos', None, float)
    if (ctrPos is None):
        logging.error('SALS.LoadFromConfig ERROR: no SALS_parameters.center_pos parameter found in config file ' + str(ConfigFile))
        return None
    else:
        r_max = ppf.MaxRadius(MIin.ImageShape(), ctrPos)
        radRange = sf.ValidateRange(config.Get('SALS_parameters', 'r_range', None, float), r_max, MinVal=1, replaceNone=True)
        angRange = config.Get('SALS_parameters', 'a_range', None, float)
        rSlices = np.geomspace(radRange[0], radRange[1], int(radRange[2])+1, endpoint=True)
        aSlices = np.linspace(angRange[0], angRange[1], int(angRange[2])+1, endpoint=True)
        logging.debug(' > radial slices specs: ' + str(radRange) + ' (original input: ' + str(config.Get('SALS_parameters', 'r_range', None, float)) + '). ' + str(len(rSlices)) + ' slices generated: ' + str(rSlices))
        logging.debug(' > angular slices specs: ' + str(angRange) + ' (original input: ' + str(config.Get('SALS_parameters', 'a_range', None, float)) + '). ' + str(len(aSlices)) + ' slices generated: ' + str(aSlices))
        if (outFolder is None):
            outFolder = config.Get(input_sect, 'out_folder', None, str)
            if (outFolder is not None):
                outFolder = os.path.join(config.Get('global', 'root', '', str), outFolder)
        mask = config.Get('SALS_parameters', 'px_mask', None, str)
        logging.debug(' > pixel mask: ' + str(mask))
        mask = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_sect, 'px_mask', mask, str)),
                             MIin.ImageShape(), MIin.DataFormat(), 0)
        dark = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_sect, 'dark_bkg', None, str)), 
                             MIin.ImageShape(), MIin.DataFormat(), 0)
        opt = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_sect, 'opt_bkg', None, str)), 
                            MIin.ImageShape(), MIin.DataFormat(), 0)
        PD_data = sf.PathJoinOrNone(froot, config.Get(input_sect, 'pd_file', None, str))
        if (PD_data is not None):
            PD_data = np.loadtxt(PD_data, dtype=float)

        img_times = LoadImageTimes(config.Get(input_sect, 'img_times', None, str), root_folder=froot, 
                                   usecols=config.Get('format', 'img_times_colidx', 0, int), skiprows=1)
        exp_times = LoadImageTimes(config.Get(input_sect, 'exp_times', None, str), root_folder=froot, 
                                   usecols=config.Get('format', 'exp_times_colidx', 0, int), skiprows=0, default_value=[1])

        dlsLags = config.Get('SALS_parameters', 'dls_lags', None, int)
        tavgT = config.Get('SALS_parameters', 'timeavg_T', None, int)
        logging.debug('SALS.LoadFromConfig() returns SALS object with MIfile ' + str(mifile_info) + ', output folder ' + str(outFolder) + 
                    ', center position ' + str(ctrPos) + ', ' + str(len(rSlices)) + ' radial and ' + str(len(aSlices)) + ' angular slices')
        return SALS(MIin, outFolder, ctrPos, [rSlices, aSlices], mask, [dark, opt, PD_data], exp_times, dlsLags, img_times, tavgT)


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
        ROIcoords : None, or couple [rSlices, aSlices].
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
        self.ROIslices = ValidatePolarSlices(ROIslices, MIin.ImageShape())
        self.centerPos = centerPos
        self.px_mask = maskRaw
        ROImasks, ROIcoords = GenerateROIs(self.ROIslices, imgShape=MIin.ImageShape(), centerPos=self.centerPos, maskRaw=self.px_mask)
        RP.ROIproc.__init__(self, MIin, ROImasks, ROIcoords=ROIcoords, imgTimes=imgTimes, expTimes=expTimes)
        
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
        
    def SetROImasks(self, ROImasks, ROIcoords=None):
        ROImetadata = {'coords' : ROIcoords, 'coord_names' : ['r[px]', 'theta[rad]'], 'box_margin' : 0}
        return RP.ROIproc.SetROImasks(self, ROImasks, ROImetadata=ROImetadata)
    
    def GenerateROIs(self, ROI_specs, maskRaw=None):
        ROImasks, ROIcoords = GenerateROIs(ROI_specs, imgShape=self.MIinput.ImageShape(), centerPos=self.centerPos, maskRaw=maskRaw)
        return self.SetROImasks(ROImasks, ROIcoords=ROIcoords)
    
    def GetSALSparams(self):
        res = {'ctrPos' : list(self.centerPos), 'rSlices' : list(self.ROIslices[0]), 'aSlices' : list(self.ROIslices[1])}
        return res
    
    def doDLS(self, saveFolder, lagtimes, reftimes='all', no_buffer=False, force_SLS=True, save_transposed=False, export_configparams=None):
        additional_params = {'SALS' : self.GetSALSparams()}
        # save raw mask, get filename
        additional_params['SALS']['raw_mask'] = raw_mask_filename
        additional_params['General']['generated_by'] = 'SALS.doDLS'
        
        if export_configparams is not None:
            additional_params = sf.UpdateDict(additional_params, export_configparams)
        return RP.ROIproc.doDLS(self, saveFolder, lagtimes=lagtimes, reftimes=reftimes, no_buffer=no_buffer, 
                                force_SLS=force_SLS, save_transposed=save_transposed, export_configparams=additional_params)