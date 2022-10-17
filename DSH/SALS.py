import os
import bisect
import math
import numpy as np
import logging
import DSH
from DSH import Config as cf
from DSH import MIfile as MI
from DSH import MIstack as MIs
from DSH import SharedFunctions as sf
from DSH import PostProcFunctions as ppf

TXT_DELIMITER = '\t'
TXT_COMMENT = '#'
SLS_RAW_FNAME = 'I_r_raw.dat'
SLS_FNAME = 'I_r.dat'
ROICOORDS_FNAME = 'ROIcoords.dat'
EXPTIMES_FNAME = 'exptimes.dat'
CI_PREFIX = 'cI'
CUSTCI_PREFIX = 'custCI'
G2M1_PREFIX = 'g2m1'
ROI_PREFIX = 'ROI'
ROI_IDXLEN = 3
EXP_PREFIX = 'e'
EXP_IDXLEN = 2

SALS_DT_TOLERANCE = 1e-2
SALS_DT_TOLERANCE_ISREL = True

def OpenSLS(froot, open_raw=False):
    if open_raw:
        fname = os.path.join(froot, SLS_RAW_FNAME)
    else:
        fname = os.path.join(froot, SLS_FNAME)
    res_Ir, res_hdr, r_phi = sf.LoadResFile(fname, delimiter=TXT_DELIMITER, comments=TXT_COMMENT, 
                                               readHeader=True, isolateFirst=2)
    times = np.asarray([sf.FirstFloatInStr(hdr) for hdr in res_hdr])
    if open_raw:
        exptimes = np.asarray([sf.LastFloatInStr(hdr) for hdr in res_hdr])
        return np.squeeze(res_Ir.reshape((res_Ir.shape[0], -1, len(set(exptimes))))), r_phi[:,0], r_phi[:,1], times, exptimes
    else:
        return res_Ir, r_phi[:,0], r_phi[:,1], times

def ReadCIfile(fpath):
    fname = sf.GetFilenameFromCompletePath(fpath)
    roi_idx = sf.FirstIntInStr(fname)
    exp_idx = sf.LastIntInStr(fname)
    cur_cI = np.loadtxt(fpath, delimiter=TXT_DELIMITER, comments=TXT_COMMENT, skiprows=1)
    cur_times = cur_cI[:,0]
    cur_cI = cur_cI[:,1:] # skip first column with image times
    with open(fpath, "r") as file:
        hdr_line = file.readline().strip()
    cur_lagidx_list = sf.ExtractIndexFromStrings(hdr_line.split(TXT_DELIMITER)[1:])
    return cur_cI, cur_times, cur_lagidx_list, roi_idx, exp_idx
    
def OpenCIs(froot):
    res = []
    fnames_list = sf.FindFileNames(froot, Prefix=CI_PREFIX+'_', Ext='.dat', Sort='ASC')
    ROI_list = [sf.FirstIntInStr(name) for name in fnames_list]
    exptime_list = [sf.LastIntInStr(name) for name in fnames_list]
    lagtimes = []
    imgtimes = []
    for i, fname in enumerate(fnames_list):
        res_cI, res_hdr, col_times = sf.LoadResFile(os.path.join(froot, fname), delimiter=TXT_DELIMITER, comments=TXT_COMMENT, 
                                                   readHeader=True, isolateFirst=1)
        res.append(res_cI)
        lagtimes.append(np.asarray([sf.FirstIntInStr(hdr) for hdr in res_hdr]))
        imgtimes.append(col_times)
    return res, imgtimes, lagtimes, ROI_list, exptime_list

def OpenG2M1s(froot, expt_idx=None, roi_idx=None):
    res = []
    filter_str = ''
    if roi_idx is not None:
        filter_str += '_' + ROI_PREFIX + str(roi_idx).zfill(ROI_IDXLEN)
    if expt_idx is not None:
        filter_str += '_' + EXP_PREFIX + + str(expt_idx).zfill(EXP_IDXLEN)
    fnames_list = sf.FindFileNames(froot, Prefix=G2M1_PREFIX, Ext='.dat', FilterString=filter_str, Sort='ASC')
    ROI_list = [sf.FirstIntInStr(name) for name in fnames_list]
    exptime_list = [sf.LastIntInStr(name) for name in fnames_list]
    lagtimes = []
    imgtimes = []
    for i, fname in enumerate(fnames_list):
        res_g2m1, res_hdr = sf.LoadResFile(os.path.join(froot, fname), delimiter=TXT_DELIMITER, comments=TXT_COMMENT, 
                                                   readHeader=True, isolateFirst=0)
        res.append(res_g2m1[:,1::2].T)
        lagtimes.append(res_g2m1[:,::2].T)
        imgtimes.append(np.asarray([sf.FirstFloatInStr(res_hdr[j]) for j in range(1, len(res_hdr), 2)]))
    return res, lagtimes, imgtimes, ROI_list, exptime_list

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

def LoadImageTimes(img_times_source, root_folder=None, usecols=0, skiprows=1, default_value=None):
    '''
    Load image times from file or list of files
    '''
    if img_times_source is not None:
        # if img_times_source is a string, let's use a single text file as input.
        # otherwise, it can be a list: in that case, let's open each text file and append all results
        if (isinstance(img_times_source, str)):
            res = np.loadtxt(os.path.join(root_folder, img_times_source), dtype=float, usecols=usecols, skiprows=skiprows)
        else:
            res = np.empty(shape=(0,), dtype=float)
            for cur_f in img_times_source:
                res = np.append(res, np.loadtxt(os.path.join(root_folder, cur_f), dtype=float, usecols=usecols, skiprows=skiprows))
    else:
        res = default_value
    return res

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
    """
    if ROI_specs is None:
        ROI_specs = [None, None]
    if (len(ROI_specs)==2):
        rSlices, aSlices = ROI_specs
        if rSlices is None:
            rSlices = [0, np.hypot(imgShape[0], imgShape[1])]
        if aSlices is None:
            aSlices = [-3.15, 3.15]
        ROIcoords = ppf.PolarMaskCoords(rSlices, aSlices, flatten_res=True)
        ROIs = ppf.GenerateMasks(ROIcoords, imgShape, center=centerPos,
                                      common_mask=maskRaw, binary_res=False, coordsystem='polar')
    else:
        ROIs = ROI_specs
        r_map, a_map = ppf.GenerateGrid2D(ROI_specs.shape, center=centerPos, coords='polar')
        r_min, r_max = ppf.ROIEval(r_map, ROI_specs, [np.min, np.max])
        a_min, a_max = ppf.ROIEval(a_map, ROI_specs, [np.min, np.max])
        rSlices, aSlices = [[r_min[i], r_max[i]] for i in range(len(r_min))], [[a_min[i], a_max[i]] for i in range(len(a_min))]
        ROIcoords = ppf.PolarMaskCoords(rSlices, aSlices, flatten_res=True)
    return ROIs, ROIcoords

def FindTimelags(times, lags, subset_len=None):
    '''
    Find lagtimes given a list of time points and a list of lag indexes
    
    Parameters
    ----------
    times:          list of time points (float), not necessarily equally spaced
    lags:           list of lag indexes (int, >=0)
    subset_len:     int (>0) or None. Eventually divide the analysis in sections of subset_len datapoints each
                    If None, it will be set to the total number of datapoints (it will analyze the whole section)
                
    Returns
    -------
    allags:         2D list. allags[i][j] = time[j+lag[i]] - time[j]
                    NOTE: len(allags[i]) depends on i (no element is added to the list if j+lag[i] >= len(times))
    unique_laglist: 2D list. Element [i][j] is j-th lagtime of i-th analyzed subsection
    '''
    if subset_len is None:
        subset_len = len(times)
    logging.debug('FindTimelags: now finding lagtimes in time series (' + str(len(times)) + 
                  ' time points, divided into sections of ' + str(subset_len) + ' datapoints each) with ' + 
                  str(len(lags)) + ' lag indexes: ' + str(lags))
    alllags = []
    for lidx in range(len(lags)):
        if (lags[lidx]==0):
            alllags.append(np.zeros_like(times, dtype=float))
        elif (lags[lidx] < len(times)):
            alllags.append(np.subtract(times[lags[lidx]:], times[:-lags[lidx]]))
        else:
            alllags.append([])
    logging.debug('alllags list has {0} elements, with lengths ranging from {1} to {2}'.format(len(alllags), len(alllags[0]), len(alllags[-1])))
    unique_laglist = []
    for tavgidx in range(int(math.ceil(len(times)*1./subset_len))):
        cur_uniquelist = np.unique([alllags[i][j] for i in range(len(lags)) 
                                    for j in range(tavgidx*subset_len, min((tavgidx+1)*subset_len, len(alllags[i])))])
        cur_coarsenedlist = [cur_uniquelist[0]]
        for lidx in range(1, len(cur_uniquelist)):
            if not sf.IsWithinTolerance(cur_uniquelist[lidx], cur_coarsenedlist[-1], 
                                          tolerance=SALS_DT_TOLERANCE, tolerance_isrelative=SALS_DT_TOLERANCE_ISREL):
                cur_coarsenedlist.append(cur_uniquelist[lidx])
        unique_laglist.append(cur_coarsenedlist)
    logging.debug('unique_laglist has {0} elements. First line has {1} elements, ranging from {2} to {3}'.format(len(unique_laglist), len(unique_laglist[0]), unique_laglist[0][0], unique_laglist[0][-1]))
    return alllags, unique_laglist
    
def AverageG2M1(cI_file, average_T=None):
    cur_cI, cur_times, cur_lagidx_list, roi_idx, exp_idx = ReadCIfile(cI_file)
    
    g2m1, g2m1_lags = AverageCorrTimetrace(cur_cI, cur_times, cur_lagidx_list, average_T)
    
    str_hdr_g = str(TXT_DELIMITER).join(['dt'+TXT_DELIMITER+'t{0:.2f}'.format(cur_times[tavgidx*average_T]) for tavgidx in range(g2m1.shape[0])])
    g2m1_out = np.empty((g2m1.shape[1], 2*g2m1.shape[0]), dtype=float)
    g2m1_out[:,0::2] = g2m1_lags.T
    g2m1_out[:,1::2] = g2m1.T
    
    np.savetxt(os.path.join(os.path.dirname(cI_file), G2M1_PREFIX + sf.GetFilenameFromCompletePath(cI_file)[2:]), 
           g2m1_out, header=str_hdr_g, delimiter=TXT_DELIMITER, comments=TXT_COMMENT)

def AverageCorrTimetrace(CorrData, ImageTimes, Lagtimes_idxlist, average_T=None):
    '''
    Average correlation timetraces
    
    Parameters
    ----------
    - CorrData: 2D array. Element [i,j] is correlation between t[i] and t[i]+tau[j]
    - ImageTimes: 1D array, float. i-th element is the physical time at which i-th image was taken
    - Lagtimes_idxlist: 1D array, int. i-th element is the lagtime, in image units
    - average_T: int or None. When averaging over time, resolve the average on chunks of average_T images each
                 if None, result will be average on the whole stack
    
    Returns
    -------
    - g2m1: 2D array. Element [i,j] represents j-th lag time and i-th time-resolved chunk
    - g2m1_lags: 2D array. It contains the time delays, in physical units, of the respective correlation data
    '''
    
    if average_T is None:
        tavg_num = 1
        average_T = CorrData.shape[0] 
    else:
        tavg_num = int(math.ceil(CorrData.shape[0]*1.0/average_T))
            
    g2m1_alllags, g2m1_laglist = FindTimelags(times=ImageTimes, lags=Lagtimes_idxlist, subset_len=average_T)
    g2m1 = np.zeros((tavg_num, np.max([len(l) for l in g2m1_laglist])), dtype=float)
    g2m1_lags = np.nan * np.ones_like(g2m1, dtype=float)
    g2m1_avgnum = np.zeros_like(g2m1, dtype=int)
    logging.debug('AverageCorrTimetrace: cI time averages will be performed by dividing the {0} time points into {1} windows of {2} time points each'.format(CorrData.shape[0], tavg_num, average_T))
    logging.debug('original cI has shape ' + str(CorrData.shape) + '. Averaged g2m1 has shape ' + str(g2m1.shape) + ' (check: ' + str(g2m1_avgnum.shape) + ')')
    
    for tidx in range(CorrData.shape[0]):
        cur_tavg_idx = tidx // average_T
        if (cur_tavg_idx >= g2m1_lags.shape[0]):
            logging.warn('AverageCorrTimetrace: {0}-th time point should go to {1}-th subsection, but result has only {2} subsections'.format(tidx, cur_tavg_idx, g2m1_lags.shape[0]))
        g2m1_lags[cur_tavg_idx,:len(g2m1_laglist[cur_tavg_idx])] = g2m1_laglist[cur_tavg_idx]
        for lidx in range(CorrData.shape[1]):
            if (tidx < len(g2m1_alllags[lidx])):
                cur_lagidx = np.argmin(np.abs(np.subtract(g2m1_laglist[cur_tavg_idx], g2m1_alllags[lidx][tidx])))
                if (~np.isnan(CorrData[tidx,lidx])):
                    g2m1_avgnum[cur_tavg_idx,cur_lagidx] += 1
                    g2m1[cur_tavg_idx,cur_lagidx] += CorrData[tidx,lidx]
    g2m1 = np.divide(g2m1, g2m1_avgnum)
    
    return g2m1, g2m1_lags



class SALS():
    """ Class to do small angle static and dynamic light scattering from a MIfile """
    
    def __init__(self, MIin, outFolder, centerPos, ROIs=None, maskRaw=None, BkgCorr=None, expTimes=[1], dlsLags=None, imgTimes=None, timeAvg_T=None):
        """
        Initialize SALS

        Parameters
        ----------
        MIin : input MIfile or MIstack. It can be empty (i.e. initialized with metadata only)
        outFolder : output folder path. If the directory doesn't exist, it will be created
        centerPos : [float, float]. Position of transmitted beam [posX, posY], in pixels.
                    The center of top left pixel is [0,0], 
                    posX increases leftwards, posY increases downwards
        ROIs :      None, or raw mask with ROI numbers, or couple [rSlices, aSlices].
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
                                None corresponds to full 2*pi angle
        maskRaw :   2D binary array with same shape as MIin.ImageShape()
                    True values (nonzero) denote pixels that will be included in the analysis,
                    False values (zeroes) will be excluded
                    If None, all pixels will be included.
                    Disregarded if ROIs is already a raw mask
        BkgCorr :   Eventually, data for background correction: [DarkBkg, OptBkg, PDdata], where:
                    DarkBkg :    None, float array or MIfile with dark measurement from the camera. 
                                If None, dark subtraction will be skipped
                                If MIfile, all images from MIfile will be averaged and the raw result will be stored
                    OptBkg :     None, float array or MIfile with empty cell measurement. Sale as MIdark
                    PDdata :    2D float array of shape (Nimgs+2, 2), where Nimgs is the number of images.
                                i-th item is (PD0, PD1), where PD0 is the reading of the incident light, 
                                and PD1 is the reading of the transmitted light.
                                Only useful if DarkBkg or OptBkg is not None
        imgTimes :  None or float array of length Nimgs. i-th element will be the time of the image
                    if None, i-th time will be computed using the FPS from the MIfile Metadata
        expTimes :  list of floats
        """
        
        self.MIinput   = MIin
        self.outFolder = outFolder
        sf.CheckCreateFolder(self.outFolder)
        self.centerPos = centerPos
        self.SetROIs(ROIs, maskRaw=maskRaw)
        self._loadBkg(BkgCorr)
        self._loadTimes(imgTimes)
        self.SetExptimes(expTimes)
        self.SetLagtimes(dlsLags)
        self.timeAvg_T = timeAvg_T
        self._initConstants()
        
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
        str_res += '\n| Output folder   : ' + str(self.outFolder)
        str_res += '\n| Center position : ' + str(self.centerPos)
        str_res += '\n| ROIs            : ' + str(self.CountROIs()) + ' (' + str(self.CountValidROIs()) + ' valid, ' + str(self.CountEmptyROIs()) + ' empty)'
        str_res += '\n| Exposure times  : ' + str(self.NumExpTimes()) + ', from ' + str(self.expTimes[0]) + ' to ' + str(self.expTimes[-1])
        str_res += '\n| DLS lag times   : ' + str(self.NumLagtimes())
        if self.NumLagtimes() > 0:
            str_res += ' [' + str(self.dlsLags[0])
            if self.NumLagtimes() > 1:
                str_res += ', '  + str(self.dlsLags[1])
            if self.NumLagtimes() > 2:
                str_res += ', ...'
            if self.NumLagtimes() > 1:
                str_res += ', '  + str(self.dlsLags[-1])
            str_res += ']'
        str_res += '\n|-----------------+---------------'
        return str_res
    
    def SetROIs(self, ROI_specs, maskRaw=None):
        '''
        Sets ROI for SALS analysis
        
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
                                None corresponds to full 2*pi angle
        maskRaw :   2D binary array with same shape as MIin.ImageShape()
                    True values (nonzero) denote pixels that will be included in the analysis,
                    False values (zeroes) will be excluded
                    If None, all pixels will be included.
                    Disregarded if ROIs is already a raw mask

        '''
        self.ROIs, self.ROIcoords = GenerateROIs(ROI_specs, imgShape=self.MIinput.ImageShape(), centerPos=self.centerPos, maskRaw=maskRaw)
        if self.CountEmptyROIs() > 0:
            if self.CountValidROIs() > 0:
                logging.warning('There are {0} out of {1} empty masks'.format(self.CountEmptyROIs(), self.CountROIs()))
            else:
                logging.error('ROI mask is empty (no valid ROIs found)')
        else:
            logging.info('Set {0} valid ROIs'.format(self.CountROIs()))
            
    def SetExptimes(self, expTimes):
        if len(expTimes) > 0:
            _exps = np.unique(expTimes)
            # check that expTimes is sorted:
            #assert np.all(np.diff(expTimes) >= 0), 'Exposure times ' + str(expTimes) + ' must be sorted!'
            self.expTimes = np.asarray(sorted(_exps))
            if len(self.expTimes) > 1:
                logging.debug('Set {0} exptimes, sorted from {1} to {2}'.format(len(self.expTimes), self.expTimes[0], self.expTimes[-1]))
            else:
                logging.debug('Set one single exposure time: {0}'.format(self.expTimes[0]))
        else:
            logging.error('SALS.SetExptimes() called with empty expTimes list: ' + str(expTimes))
            
    def SetLagtimes(self, LagList):
        self.dlsLags = None
        if LagList is not None:
            _lags = np.unique(LagList)
            if len(_lags) > 0:
                # ensure that lags are sorted and include 0
                if _lags[0]>0:
                    _lags = np.append([0], sorted(_lags))
                self.dlsLags = _lags
            else:
                self.dlsLags = None

    def CountROIs(self):
        return self.ROIcoords.shape[0]
    def CountEmptyROIs(self):
        return self.CountROIs() - self.CountValidROIs()
    def CountValidROIs(self):
        return np.max(self.ROIs)+1
    def ImageNumber(self):
        return self.MIinput.ImageNumber()
    def NumTimes(self):
        return self.MIinput.ImageNumber() // len(self.expTimes)
    def NumExpTimes(self):
        return len(self.expTimes)
    def NumLagtimes(self):
        if self.dlsLags is None:
            return 0
        else:
            return len(self.dlsLags)
    def StackInput(self):
        return self.MIinput.IsStack()
    def GetOutFolder(self):
        return self.outFolder
        
    def SaveSLS(self, IofR, NormF, AllExpData=None, save_folder=None):
        """ Saves output of SLS analysis

        Parameters
        ----------
        IofR : 2D array of shape (NumTimes(), NumROIs())
        NormF : 1D array with ROI normalization factors
        AllExpData : None or [I, exptime], data with all exposure times
        """
        if save_folder is None:
            save_folder = self.outFolder
        roi_norms = np.zeros((IofR.shape[-1], 1))
        roi_norms[:len(NormF),0] = NormF
        np.savetxt(os.path.join(save_folder, ROICOORDS_FNAME), np.append(self.ROIcoords, roi_norms, axis=1), 
                   header='r[px]'+TXT_DELIMITER+'phi[rad]'+TXT_DELIMITER+'dr[px]'+TXT_DELIMITER+'dphi[rad]'+TXT_DELIMITER+'norm', **self.savetxt_kwargs)
        MI.WriteBinary(os.path.join(save_folder, 'ROI_mask.raw'), self.ROIs, 'i')
        str_hdr_Ir = 'r[px]'+TXT_DELIMITER+'phi[rad]' + ''.join([TXT_DELIMITER+'t{0:.2f}'.format(self.imgTimes[i]) for i in range(0, self.ImageNumber(), self.NumExpTimes())])
        np.savetxt(os.path.join(save_folder, SLS_FNAME), np.append(self.ROIcoords[:IofR.shape[1],:2], IofR.T, axis=1), 
                   header=str_hdr_Ir, **self.savetxt_kwargs)
        if AllExpData is not None:
            ROIavgs_allExp, BestExptime_Idx = AllExpData
            np.savetxt(os.path.join(save_folder, EXPTIMES_FNAME), np.append(self.ROIcoords[:,:2], BestExptime_Idx.T, axis=1), 
                       header=str_hdr_Ir, **self.savetxt_kwargs)
            str_hdr_raw = 'r[px]'+TXT_DELIMITER+'phi[rad]' + ''.join([TXT_DELIMITER+'t{0:.2f}_e{1:.3f}'.format(self.imgTimes[i], self.expTimes[i%len(self.expTimes)]) for i in range(len(self.imgTimes))])
            np.savetxt(os.path.join(save_folder, SLS_RAW_FNAME), np.append(self.ROIcoords[:,:2], ROIavgs_allExp.reshape((-1, ROIavgs_allExp.shape[-1])).T, axis=1), 
                       header=str_hdr_raw, **self.savetxt_kwargs)

    def LoadSLS(self):
        Ir_allexp, Ir, best_exptimes = None, None, None
        if os.path.isfile(os.path.join(self.outFolder, SLS_FNAME)):
            Ir = np.loadtxt(os.path.join(self.outFolder, SLS_FNAME), **self.savetxt_kwargs)
            if (Ir.shape[1] > 2):
                Ir = Ir[:,2:]
            else:
                Ir = None
        if Ir is not None:
            if os.path.isfile(os.path.join(self.outFolder, SLS_RAW_FNAME)):
                Ir_allexp = np.loadtxt(os.path.join(self.outFolder, EXPTIMES_FNAME), **self.savetxt_kwargs)
                if (Ir_allexp.shape[1] > 2):
                    Ir_allexp = Ir_allexp[:,2:]
                    Ir_allexp.reshape((Ir.shape[0], -1, Ir_allexp.shape[-1]))
                else:
                    Ir_allexp = None
        if os.path.isfile(os.path.join(self.outFolder, EXPTIMES_FNAME)):
            best_exptimes = np.loadtxt(os.path.join(self.outFolder, EXPTIMES_FNAME), **self.savetxt_kwargs)
            if (best_exptimes.shape[1] > 2):
                best_exptimes = best_exptimes[:,2:]
            else:
                best_exptimes = None
                
        return Ir_allexp, Ir, best_exptimes
                    
    def ReadCIfile(self, fname):
        cur_cI, cur_times, cur_lagidx_list, roi_idx, exp_idx = ReadCIfile(os.path.join(self.outFolder, fname))
        if cur_lagidx_list != self.dlsLags:
            logging.warning('cI result file {0} potentially corrupted: lagtimes are different from expected (read {1}, expected {2})').format(fname, cur_lagidx_list, self.dlsLags)
        if (cur_cI.shape != (self.NumTimes(), self.NumLagtimes())):
            logging.warning('cI result file {0} potentially corrupted: shape {1} does not match expected {2}'.format(fname, cur_cI.shape, (self.NumTimes(), self.NumLagtimes())))
        return cur_cI, cur_times, cur_lagidx_list, roi_idx, exp_idx

    def AverageG2M1(self):
        if self.timeAvg_T is None:
            self.timeAvg_T = self.NumTimes()
        cI_fnames = sf.FindFileNames(self.outFolder, Prefix=CI_PREFIX+'_', Ext='.dat')
        
        for cur_f in cI_fnames:
            AverageG2M1(os.path.join(self.outFolder, cur_f), average_T=self.timeAvg_T)
    
    def GetOrReadImage(self, img_idx, buffer=None):
        """ Retrieve image from buffer if present, otherwise read if from MIfile
        """
        if buffer is not None:
            if len(buffer) > img_idx:
                return buffer[img_idx]
        return self.MIinput.GetImage(img_idx)
    
    def ROIaverageIntensity(self, stack1, ROImasks=None, masks_isBool=False, no_buffer=False, imgs=None):
        return self.ROIaverageProduct(stack1, stack2=None, ROImasks=ROImasks, masks_isBool=masks_isBool, no_buffer=no_buffer, imgs=imgs)
    
    def ROIaverageProduct(self, stack1, stack2=None, ROImasks=None, masks_isBool=False, no_buffer=False, imgs=None):
        """ ROI average product of images

        Parameters
        ----------
        stack1 : list of indexes. Images will be either read by MIinput or retrieved from img_buffer
        stack2 : None, or list of indexes
                 - if None: function will return averages of single images (in stack1)
                 - if list: length should be the same as stack1
        ROImasks : 2D int array or 3D bool array: masks associating each pixel to a ROI
                 - if 2D int array: pixel value will denote which ROI the pixel belongs to
                 - if 3D bool array: i-th 2D array willbe True for pixels belonging to i-th ROI
                 shape of each 2D mask has to match the image shape
                 if None, self.ROIs will be used (which is of type int by default)
        masks_isBool : True or False to indicate that ROImasks is of bool or int type, respectively
        no_buffer : if True, avoid reading all images to a buffer, but read images one by one
                    (dumping them afterwards)
        imgs : None or 3D array with buffered images. If None, images will be read from MIinput

        Returns
        -------
        AvgRes : 2D array. Element [i,j] is the average of i-th image on j-th ROI
        NormList : 1D array. i-th element is the number of pixels in i-th ROI.
        """
        if ROImasks is None:
            ROImasks = self.ROIs
            masks_isBool = False
        if masks_isBool:
            num_ROI = len(ROImasks)
        else:
            num_ROI = np.max(ROImasks)+1
            
        if (self.StackInput() or no_buffer):
            AvgRes = np.nan*np.ones((len(stack1), num_ROI), dtype=float)
            for i in range(AvgRes.shape[0]):
                if stack2 is None:
                    AvgRes[i], NormList = ppf.ROIAverage(self.GetOrReadImage(stack1[i], imgs), ROImasks, boolMask=masks_isBool)
                else:
                    if (stack1[i]==stack2[i]):
                        AvgRes[i], NormList = ppf.ROIAverage(np.square(self.GetOrReadImage(stack1[i], imgs)), ROImasks, boolMask=masks_isBool)
                    else:
                        AvgRes[i], NormList = ppf.ROIAverage(np.multiply(self.GetOrReadImage(stack1[i], imgs), self.GetOrReadImage(stack2[i], imgs)), ROImasks, boolMask=masks_isBool)
                    if (self.DebugMode):
                        if (np.any(AvgRes[i]<0)):
                            min_idx = np.argmin(AvgRes[i])
                            logging.warn('Negative cross product value (image1: {0}, image2: {1}, ROI{2} avg: {3})'.format(stack1[i], stack2[i], min_idx, AvgRes[i][min_idx]))
                            logging.debug('   >>> Debug output for ROIAverage function:')
                            ppf.ROIAverage(np.multiply(self.GetOrReadImage(stack1[i], imgs), self.GetOrReadImage(stack2[i], imgs)), ROImasks, boolMask=masks_isBool, debug=True)
        else:
            if imgs is None:
                imgs = self.MIinput.Read()
            if stack2 is None:
                cur_stack = imgs[stack1]
            elif (stack1==stack2):
                cur_stack = np.square(imgs[stack1])
            else:
                cur_stack = np.multiply(imgs[stack1], imgs[stack2])
            AvgRes, NormList = ppf.ROIAverage(cur_stack, ROImasks, boolMask=masks_isBool)
            NormList = NormList[0]
            
        return AvgRes, NormList
    
    def CorrelateWithImage(self, ref_idx=[0], no_buffer=False):
        """ Compute correlations between a given (list of) reference image(s) and all other images
        """

        ROI_boolMasks = [self.ROIs==b for b in range(self.CountROIs())]
        ROIavgs_allExp, ROIavgs_best, BestExptime_Idx, buf_images = self.doSLS(ROImasks=ROI_boolMasks, saveFolder=self.outFolder, buf_images=None, 
                                                                               no_buffer=no_buffer, force_calc=force_SLS)
        if buf_images is None:
            if no_buffer:
                self.MIinput.OpenForReading()
                buf_images = None
            elif self.StackInput()==False:
                buf_images = self.MIinput.Read()
        
        

    
    def doSLS(self, ROImasks=None, saveFolder=None, buf_images=None, no_buffer=False, force_calc=True):
        """ Run SLS analysis
        
        Parameters
        ----------
        - ROImasks: list of binary masks, one element per ROI. 
                    Each (i-th) element is a binary image, pixel is 1 if it belongs to i-th ROI, else it is 0 
                    if None, they will be computed from self.ROIs
        - saveFolder: folder path, 'auto' or None. If not None, save analysis output in specified folder. 
                      If 'auto', the standard output folder for SALS analysis will be used
        - buf_images: 3D array, buffer with images to be processed. If None, images will be loaded
        - no_buffer: bool, used only if buf_images is None. If False, the entire image stack will be read in a buffer at once.
                     otherwise, images will be loaded one by one
        - force_calc: bool. If False, program will search for previously computed SLS results and load those.
        
        Returns
        -------
        - ROIavgs_allExp : 3D array. Element [i,j,k] is the average of j-th exposure time in i-th exposure time sweep, averaged on k-th ROI
        - ROIavgs_best : 2D array. Element [i,j] is the best average intensity taken from i-th exposure time sweep, averaged on j-th ROI
        - BestExptime_Idx : 2D array (int). Element [i,j] is the index of the optimum exposure time for j-th ROI
        - buf_images : 3D array. Buffer of images eventually read during the analysis
        """
        
        if saveFolder is not None:
            if saveFolder=='auto':
                saveFolder = self.outFolder
            sf.CheckCreateFolder(saveFolder)
        
        ROIavgs_allExp, ROIavgs_best, BestExptime_Idx = None, None, None
        if not force_calc:
            ROIavgs_allExp, ROIavgs_best, BestExptime_Idx = self.LoadSLS()
        
        if ROIavgs_allExp is None or ROIavgs_best is None or BestExptime_Idx is None:

            if buf_images is None:
                if no_buffer:
                    self.MIinput.OpenForReading()
                    buf_images = None
                elif self.StackInput()==False:
                    buf_images = self.MIinput.Read()

            if ROImasks is None:
                ROImasks = [self.ROIs==b for b in range(self.CountROIs())]

            # Compute average intensity for all images
            all_avg, NormList = self.ROIaverageProduct(stack1=list(range(self.ImageNumber())), stack2=None, ROImasks=ROImasks, 
                                                       masks_isBool=True, no_buffer=no_buffer, imgs=buf_images)
            if all_avg.shape[0] % (self.NumTimes() * self.NumExpTimes()) != 0:
                limit_len = self.NumTimes() * self.NumExpTimes()
                all_avg = all_avg[:limit_len]
                logging.warning('Number of images ({0}) is not a multiple of exposure times ({1}). '.format(self.ImageNumber(), self.NumExpTimes()) + 
                                'Average intensity output of shape {0} cannot be reshaped using number of times ({1}) and exposure times ({2}). '.format(all_avg.shape, self.NumTimes(), self.NumExpTimes()) +
                                'Restricting SLS analysis to first {0} images'.format(limit_len))
            ROIavgs_allExp = all_avg.reshape((self.NumTimes(), self.NumExpTimes(), -1))
            ROIavgs_best, BestExptime_Idx = self.FindBestExptimes(ROIavgs_allExp)
            if saveFolder is not None:
                self.SaveSLS(ROIavgs_best, NormList, [ROIavgs_allExp, BestExptime_Idx], save_folder=saveFolder)
            logging.debug('SLS output saved')

            # TODO: time average SLS
        
        return ROIavgs_allExp, ROIavgs_best, BestExptime_Idx, buf_images

    def FindBestExptimes(self, AverageIntensities):
        '''
        Find best exposure times based on ROI-averaged intensities
        
        Parameters
        ----------
        - AverageIntensities: 3D array. Element [i,j,k] is the average intensity of j-th ROI measured at k-th exposure time during i-th exposure time ramp
        
        Returns
        -------
        - ROIavgs_best: 2D array. Element [i,j] is the intensity of the best exposure time of j-th ROI during i-th time, normalized by the exposure time itself
        - BestExptime_Idx: 2D array, containing the index of the best exposure time selected for ROIavgs_best
        '''
        if AverageIntensities.ndim < 3:
            AverageIntensities = AverageIntensities.reshape((self.NumTimes(), self.NumExpTimes(), -1))
        ROIavgs_best = np.zeros((self.NumTimes(), AverageIntensities.shape[-1]), dtype=float)
        BestExptime_Idx = -1 * np.ones_like(ROIavgs_best, dtype=int)
        for idx, val in np.ndenumerate(ROIavgs_best):
            BestExptime_Idx[idx] = min(bisect.bisect(AverageIntensities[idx[0], :, idx[1]], self.MaxSafeAvgIntensity), len(self.expTimes)-1)
            ROIavgs_best[idx] = AverageIntensities[idx[0], BestExptime_Idx[idx], idx[1]] / self.expTimes[BestExptime_Idx[idx]]
        return ROIavgs_best, BestExptime_Idx
    
    def doDLS(self, no_buffer=False, force_SLS=True, reftimes='all', lagtimes='auto', save_transposed=False):
        """ Run SLS/DLS analysis

        Parameters
        ----------
        no_buffer : bool. If True, avoid reading full MIfile to RAM
        force_SLS : bool. If False, program will load previously computed SLS results if available.
        reftimes : 'auto', 'all' or list of int. 
                    - If 'auto' or 'all', all reference times will be used
                    - Otherwise, specialize the analysis to a subset of reference times
        lagtimes : 'auto', 'all' or list of int. 
                    - If 'auto' (default), lagtimes will be set to self.dlsLags
                    - If 'all', all available lagtimes will be processed
                    - Otherwise, only specified lagtimes will be processed
        save_transposed: bool. Format of correlation timetrace output
                    - if False, classic cI output: one line per reference time, one column per time delay
                    - if True, transposed output: one line per time delay, one column per 
        """
        if lagtimes=='auto':
            DLS_lags = np.asarray(self.dlsLags)
            DLS_lagnum = len(DLS_lags)
        elif lagtimes=='all':
            DLS_lags = []
            DLS_lagnum = self.ImageNumber()
        else:
            DLS_lags = np.asarray(lagtimes)
            DLS_lagnum = len(DLS_lags)
            
        if reftimes in ['auto', 'all']:
            DLS_reftimes = np.arange(self.NumTimes())
        else:
            DLS_reftimes = np.asarray(reftimes)
            
        logging.info('SALS Analysis started! Will analyze {0} images ({1} times, {2} exposure times)'.format(self.ImageNumber(), self.NumTimes(), self.NumExpTimes()))
        if (self.ImageNumber() != (self.NumTimes() * self.NumExpTimes())):
            logging.warn('WARNING: Number of images ({0}) should correspond to the number of times ({1}) times the number of exposure times ({2})'.format(self.ImageNumber(), self.NumTimes(), self.NumExpTimes()))
        logging.info('Analysis will resolve {0} ROIs and DLS will be performed on {1} reference times and {2} lagtimes. Output will be saved in folder {3}'.format(self.CountROIs(), len(DLS_reftimes), DLS_lagnum, self.outFolder))
        logging.info('Now starting with SLS...')
        
        ROI_boolMasks = [self.ROIs==b for b in range(self.CountROIs())]
        ROIavgs_allExp, ROIavgs_best, BestExptime_Idx, buf_images = self.doSLS(ROImasks=ROI_boolMasks, saveFolder=self.outFolder, buf_images=None, 
                                                                               no_buffer=no_buffer, force_calc=force_SLS)
        if buf_images is None:
            if no_buffer:
                self.MIinput.OpenForReading()
                buf_images = None
            elif self.StackInput()==False:
                buf_images = self.MIinput.Read()
        
        if DLS_lags is None:
            logging.warn('No lagtimes specified for DLS')
        elif len(DLS_reftimes)<=1:
            logging.warn('At least 1 reference time needed for DLS (' + str(len(DLS_reftimes)) + ' given)')
        else:
            logging.info('SLS analysis completed. Now doing DLS ({0} exposure times, {1} time points, {2} lagtimes)'.format(self.NumExpTimes(), len(DLS_reftimes), DLS_lagnum))
            for e in range(self.NumExpTimes()):
                readrange = self.MIinput.Validate_zRange([e, -1, self.NumExpTimes()])
                idx_list = list(range(*readrange))
                logging.info('Now performing DLS on {0}-th exposure time. Using image range {1} ({2} images)'.format(e, readrange, len(idx_list)))
                ISQavg, NormList = self.ROIaverageProduct(stack1=idx_list, stack2=idx_list, ROImasks=ROI_boolMasks, masks_isBool=True, no_buffer=no_buffer, imgs=buf_images)
                                                        
                if reftimes in ['auto', 'all'] and lagtimes=='auto':
                    cI = np.nan * np.ones((ISQavg.shape[1], ISQavg.shape[0], DLS_lagnum), dtype=float)
                    cI[:,:,0] = np.subtract(np.divide(ISQavg, np.square(ROIavgs_allExp[:,e,:])), 1).T
                    for lidx in range(1, DLS_lagnum):
                        if (DLS_lags[lidx]<ISQavg.shape[0]):
                        
                            IXavg, NormList = self.ROIaverageProduct(stack1=idx_list[:-DLS_lags[lidx]], stack2=idx_list[DLS_lags[lidx]:], 
                                                                     ROImasks=ROI_boolMasks, masks_isBool=True, no_buffer=no_buffer, imgs=buf_images)
                            # 'classic' cI formula
                            cI[:,:-DLS_lags[lidx],lidx] = np.subtract(np.divide(IXavg, np.multiply(ROIavgs_allExp[:-DLS_lags[lidx],e,:],
                                                                                                       ROIavgs_allExp[DLS_lags[lidx]:,e,:])), 1).T
                            # d0 normalization
                            cI[:,:-DLS_lags[lidx],lidx] = np.divide(cI[:,:-DLS_lags[lidx],lidx], 0.5 * np.add(cI[:,:-DLS_lags[lidx],0], cI[:,DLS_lags[lidx]:,0]))
                            logging.info('Lagtime {0}/{1} (d{2}) completed'.format(lidx, DLS_lagnum-1, DLS_lags[lidx]))
                    
                    
                else:
                    
                    cI = np.nan * np.ones((ISQavg.shape[1], len(DLS_reftimes), DLS_lagnum), dtype=float)
                    for t in DLS_reftimes:
                        cI[:,t,0] = np.subtract(np.divide(ISQavg[t,:], np.square(ROIavgs_allExp[t,e,:])), 1)
                        
                    for ref_tidx in range(len(DLS_reftimes)):
                        
                        cur_lagtimes = np.arange(self.NumTimes()-DLS_reftimes[ref_tidx])-DLS_reftimes[ref_tidx]
                        IXavg, NormList = self.ROIaverageProduct(stack1=[idx_list[DLS_reftimes[ref_tidx]]]*len(cur_lagtimes), stack2=idx_list[cur_lagtimes+DLS_reftimes[ref_tidx]], 
                                                                 ROImasks=ROI_boolMasks, masks_isBool=True, no_buffer=no_buffer, imgs=buf_images)
                        
                        for lidx in range(len(cur_lagtimes)):
                            # 'classic' cI formula
                            cI[:,ref_tidx,lidx] = np.subtract(np.divide(IXavg[DLS_reftimes[ref_tidx],:], np.multiply(ROIavgs_allExp[DLS_reftimes[ref_tidx],e,:],
                                                                                                       ROIavgs_allExp[DLS_reftimes[ref_tidx]+cur_lagtimes[lidx],e,:])), 1)
                            # d0 normalization
                            cI[:,ref_tidx,lidx] = np.divide(cI[:,ref_tidx,lidx], 0.5 * np.add(cI[:,ref_tidx,0], cI[:,DLS_reftimes[ref_tidx]+cur_lagtimes[lidx],0]))
                            logging.info('Lagtime {0}/{1} (tref={2}) completed'.format(ref_tidx, len(DLS_reftimes), DLS_reftimes[ref_tidx]))
                                
                                
                        
                # Save data to file
                for ridx in range(cI.shape[0]):
                    logging.info('Now saving ROI {0} to file'.format(ridx))
                    if reftimes in ['auto', 'all'] and lagtimes=='auto':
                        np.savetxt(os.path.join(self.outFolder, CI_PREFIX+'_'+ROI_PREFIX + str(ridx).zfill(ROI_IDXLEN) + '_'+EXP_PREFIX + str(e).zfill(EXP_IDXLEN) + '.dat'), 
                                   np.append(self.imgTimes[idx_list].reshape((-1, 1)), cI[ridx], axis=1), 
                                   header='t'+TXT_DELIMITER + str(TXT_DELIMITER).join(['d{0}'.format(l) for l in DLS_lags]), **self.savetxt_kwargs)
                    else:
                        np.savetxt(os.path.join(self.outFolder, CUSTCI_PREFIX+'_'+ROI_PREFIX + str(ridx).zfill(ROI_IDXLEN) + '_'+EXP_PREFIX + str(e).zfill(EXP_IDXLEN) + '.dat'), 
                                   np.append(self.imgTimes[idx_list].reshape((-1, 1)), cI[ridx], axis=1), 
                                   header='t'+TXT_DELIMITER + str(TXT_DELIMITER).join(['d{0}'.format(l) for l in DLS_lags]), **self.savetxt_kwargs)                    
                    

        logging.info('DLS analysis completed. Now averaging correlation functions g2-1')
        self.AverageG2M1()





        #ROIhistFit_allExp = np.zeros_like(ROIavgs_allExp, dtype=float)
        #ROIhistFit_weights = np.ones_like(ROIavgs_allExp, dtype=float)
        #self.MIinput.OpenForReading()
        #for i in range(self.MIinput.ImageNumber()):
            #tidx, expidx = i//self.NumExpTimes(), i%self.NumExpTimes()
            #curavg, NormList = ppf.ROIAverage(self.MIinput.GetImage(i), self.ROIs)
            #ROIavgs_allExp[tidx, expidx, :len(curavg)] = curavg
            #if (self.HistogramSLS):
                # TODO: implement this:
                # - compute intensity histogram
                # - select relevant part (ex: set max to min(max_val, 250) and min to max(hist)+1)
                # - fit log(pdf) vs I with a line
                # - think about how to relate the slope to the average intensity, it is straightforward
                # - think about how to estimate the uncertainty on the average (to properly average all exposure times afterwards)
                #pass
            
        # TODO: weighted average of histFits
        # TODO: dark and opt correction
        

    def _loadBkg(self, BkgCorr):
        if BkgCorr is None:
            BkgCorr = [None, None, None]
        self.DarkBkg, self.OptBkg, self.PDdata = BkgCorr
    def _loadTimes(self, imgTimes=None):
        if imgTimes is None:
            self.imgTimes = np.arange(0, self.ImageNumber() * 1./self.MIinput.GetFPS(), 1./self.MIinput.GetFPS())
            logging.debug('{0} image times automatically generated from MI metadata (fps={1}Hz)'.format(len(self.imgTimes), self.MIinput.GetFPS()))
        else:
            if (len(imgTimes) < self.ImageNumber()):
                raise ValueError('Image times ({0}) should at least be as many as the image number ({1}).'.format(len(imgTimes), self.ImageNumber()))
            elif (len(imgTimes) > self.ImageNumber()):
                logging.warn('More image times ({0}) than images ({1}). Only first {1} image times will be considered'.format(len(imgTimes), self.ImageNumber()))
                self.imgTimes  = imgTimes[:self.ImageNumber()]
            else:
                self.imgTimes  = imgTimes
                logging.debug('{0} image times loaded (Image number: {1})'.format(len(self.imgTimes), self.ImageNumber()))
    def _initConstants(self):
        #Constants
        self.HistogramSLS = False
        self.MaxSafeAvgIntensity = 40
        self.dt_tolerance = SALS_DT_TOLERANCE
        self.dt_tolerance_isrelative = SALS_DT_TOLERANCE_ISREL
        self.DebugMode = False
        self.savetxt_kwargs = {'delimiter':TXT_DELIMITER, 'comments':TXT_COMMENT}