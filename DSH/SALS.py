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
SLS_FNAME = 'I_r'
CI_PREFIX = 'cI'
G2M1_PREFIX = 'g2m1'
ROI_PREFIX = 'ROI'
ROI_IDXLEN = 3
EXP_PREFIX = 'e'
EXP_IDXLEN = 2

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
        else:
            input_stack = True
            for i in range(len(miin_fname)):
                miin_fname[i] = os.path.join(froot, miin_fname[i])
    if (miin_meta_fname is not None):
        miin_meta_fname = os.path.join(froot, miin_meta_fname)
    elif input_stack:
        logging.error('SALS.LoadFromConfig ERROR: medatada filename must be specified when loading a MIstack')
        return None
    if input_stack:
        MIin = MIs.MIstack(miin_fname, miin_meta_fname, Load=True, StackType='t')
    else:
        MIin = MI.MIfile(miin_fname, miin_meta_fname)
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
        mask = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_sect, 'px_mask', mask, str)),
                             MIin.ImageShape(), MIin.DataFormat(), 0)
        dark = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_sect, 'dark_bkg', None, str)), 
                             MIin.ImageShape(), MIin.DataFormat(), 0)
        opt = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_sect, 'opt_bkg', None, str)), 
                            MIin.ImageShape(), MIin.DataFormat(), 0)
        PD_data = sf.PathJoinOrNone(froot, config.Get(input_sect, 'pd_file', None, str))
        if (PD_data is not None):
            PD_data = np.loadtxt(PD_data, dtype=float)
        img_times = config.Get(input_sect, 'img_times', None, str)
        if img_times is not None:
            # if miin_fname is a string, let's use a single text file as input.
            # otherwise, it can be a list: in that case, let's open each text file and append all results
            if (isinstance(img_times, str)):
                img_times = np.loadtxt(os.path.join(froot, img_times), dtype=float, usecols=config.Get('format', 'img_times_colidx', 0, int), skiprows=1)
            else:
                tmp_times = np.empty(shape=(0,), dtype=float)
                for cur_f in img_times:
                    tmp_times = np.append(tmp_times, np.loadtxt(os.path.join(froot, cur_f), dtype=float, usecols=config.Get('format', 'img_times_colidx', 0, int), skiprows=1))
                img_times = tmp_times
        exp_times = sf.PathJoinOrNone(froot, config.Get(input_sect, 'exp_times', None, str))
        if (exp_times is not None):
            exp_times = np.unique(np.loadtxt(exp_times, dtype=float, usecols=config.Get('format', 'exp_times_colidx', 0, int)))
        dlsLags = config.Get('SALS_parameters', 'dls_lags', None, int)
        tavgT = config.Get('SALS_parameters', 'timeavg_T', None, int)
        return SALS(MIin, outFolder, ctrPos, [rSlices, aSlices], mask, [dark, opt, PD_data], exp_times, dlsLags, img_times, tavgT)

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
        self._genROIs(ROIs, maskRaw=maskRaw)
        self._loadBkg(BkgCorr)
        self._loadTimes(imgTimes)
        # check that expTimes is sorted:
        assert np.all(np.diff(expTimes) >= 0), 'Exposure times ' + str(expTimes) + ' must be sorted!'
        self.expTimes  = expTimes
        # ensure that lags are sorted and include 0
        if dlsLags is not None:
            dlsLags = np.unique(dlsLags)
            if dlsLags[0]>0:
                dlsLags = np.append([0], dlsLags)
        self.dlsLags = dlsLags
        self.timeAvg_T = timeAvg_T
        self._initConstants()
        
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
        return len(self.dlsLags)
    def StackInput(self):
        return self.MIinput.IsStack()
    def GetOutFolder(self):
        return self.outFolder
    def IsTimingConstant(self, times=None, tolerance=None, tolerance_isrelative=None):
        if times is None:
            times = self.imgTimes
        if len(times) <= 1:
            return True
        else:
            dt_arr = np.diff(times)
            return self.IsWithinTolerance(np.min(dt_arr), np.max(dt_arr), tolerance=tolerance, tolerance_isrelative=tolerance_isrelative)
    def IsWithinTolerance(self, t1, t2, tolerance=None, tolerance_isrelative=None):
        if tolerance is None:
            tolerance = self.dt_tolerance
        if tolerance_isrelative is None:
            tolerance_isrelative = self.dt_tolerance_isrelative
        if tolerance_isrelative:
            return (np.abs(t2 - t1) < tolerance * 0.5 * np.abs(t2 + t1))
        else:
            return (np.abs(t2 - t1) < tolerance)
        
    def SaveSLS(self, IofR, NormF, AllExpData=None):
        """ Saves output of SLS analysis

        Parameters
        ----------
        IofR : 2D array of shape (NumTimes(), NumROIs())
        NormF : 1D array with ROI normalization factors
        AllExpData : None or [I, exptime], data with all exposure times
        """
        roi_norms = np.zeros((IofR.shape[-1], 1))
        roi_norms[:len(NormF),0] = NormF
        np.savetxt(os.path.join(self.outFolder, 'ROIcoords.dat'), np.append(self.ROIcoords, roi_norms, axis=1), 
                   header='r[px]'+TXT_DELIMITER+'phi[rad]'+TXT_DELIMITER+'dr[px]'+TXT_DELIMITER+'dphi[rad]'+TXT_DELIMITER+'norm', **self.savetxt_kwargs)
        MI.WriteBinary(os.path.join(self.outFolder, 'ROI_mask.raw'), self.ROIs, 'i')
        str_hdr_Ir = 'r[px]'+TXT_DELIMITER+'phi[rad]' + ''.join([TXT_DELIMITER+'t{0:.2f}'.format(self.imgTimes[i]) for i in range(0, self.ImageNumber(), self.NumExpTimes())])
        np.savetxt(os.path.join(self.outFolder, SLS_FNAME), np.append(self.ROIcoords[:IofR.shape[1],:2], IofR.T, axis=1), 
                   header=str_hdr_Ir, **self.savetxt_kwargs)
        if AllExpData is not None:
            ROIavgs_allExp, BestExptime_Idx = AllExpData
            np.savetxt(os.path.join(self.outFolder, 'exptimes.dat'), np.append(self.ROIcoords[:,:2], BestExptime_Idx.T, axis=1), 
                       header=str_hdr_Ir, **self.savetxt_kwargs)
            str_hdr_raw = 'r[px]'+TXT_DELIMITER+'phi[rad]' + ''.join([TXT_DELIMITER+'t{0:.2f}_e{1:.3f}'.format(self.imgTimes[i], self.expTimes[i%len(self.expTimes)]) for i in range(len(self.imgTimes))])
            np.savetxt(os.path.join(self.outFolder, SLS_RAW_FNAME), np.append(self.ROIcoords[:,:2], ROIavgs_allExp.reshape((-1, ROIavgs_allExp.shape[-1])).T, axis=1), 
                       header=str_hdr_raw, **self.savetxt_kwargs)

    def ReadCIfile(self, fname):
        roi_idx = sf.FirstIntInStr(fname)
        exp_idx = sf.LastIntInStr(fname)
        cur_cI = np.loadtxt(os.path.join(self.outFolder, fname), **self.loadtxt_kwargs)
        cur_times = cur_cI[:,0]
        cur_cI = cur_cI[:,1:] # skip first row with image times
        if (cur_cI.shape != (self.NumTimes(), self.NumLagtimes())):
            logging.warning('cI result file {0} potentially corrupted: shape {1} does not match expected {2}'.format(fname, cur_cI.shape, (self.NumTimes(), self.NumLagtimes())))
        return cur_cI, cur_times, roi_idx, exp_idx
    
    def FindTimelags(self, times=None, lags=None, subset_len=None):
        if times is None:
            times = self.imgTimes
        if lags is None:
            lags = self.dlsLags
        if subset_len is None:
            subset_len = self.timeAvg_T
        alllags = [] # double list. Element [i][j] is time[j+lag[i]]-time[j]
        for lidx in range(len(lags)):
            if (lags[lidx]==0):
                alllags.append(np.zeros_like(times, dtype=float))
            elif (lags[lidx] < len(times)):
                alllags.append(np.subtract(times[lags[lidx]:], times[:-lags[lidx]]))
            else:
                alllags.append([])
        unique_laglist = [] # double list. Element [i][j] is j-th lagtime of i-th averaged time
        for tavgidx in range(int(math.ceil(len(times)*1./subset_len))):
            cur_uniquelist = np.unique([alllags[i][j] for i in range(len(lags)) 
                                        for j in range(tavgidx*subset_len, min((tavgidx+1)*subset_len, len(alllags[i])))])
            cur_coarsenedlist = [cur_uniquelist[0]]
            for lidx in range(1, len(cur_uniquelist)):
                if not self.IsWithinTolerance(cur_uniquelist[lidx], cur_coarsenedlist[-1]):
                    cur_coarsenedlist.append(cur_uniquelist[lidx])
            unique_laglist.append(cur_coarsenedlist)
        return alllags, unique_laglist

    def AverageG2M1(self):
        if self.timeAvg_T is None:
            self.timeAvg_T = self.NumTimes()
        cI_fnames = sf.FindFileNames(self.outFolder, Prefix=CI_PREFIX+'_', Ext='.dat')
        
        for cur_f in cI_fnames:
            cur_cI, cur_times, roi_idx, exp_idx = self.ReadCIfile(cur_f)
            tavg_num = cur_cI.shape[0] // self.timeAvg_T
            logging.debug('cI time averages will be performed by dividing the {0} images into {1} windows of {2} images each'.format(self.NumTimes(), tavg_num, self.timeAvg_T))
            
            if self.IsTimingConstant(cur_times):
                g2m1_lags = np.true_divide(self.dlsLags, self.MIinput.GetFPS())
                g2m1 = np.nan * np.ones((self.NumLagtimes(), tavg_num), dtype=float)
                for tavgidx in range(tavg_num):
                    g2m1[:,tavgidx] = np.nanmean(cur_cI[tavgidx*self.timeAvg_T:(tavgidx+1)*self.timeAvg_T,:], axis=0)
                str_hdr_g = self.txt_comment + 'dt' + ''.join([TXT_DELIMITER+'t{0:.2f}'.format(cur_times[tavgidx*self.timeAvg_T]) for tavgidx in range(tavg_num)])
                g2m1_out = np.append(g2m1_lags.reshape((-1, 1)), g2m1, axis=0).T
            else:
                g2m1_alllags, g2m1_laglist = self.FindTimelags(times=cur_times)
                g2m1 = np.zeros((tavg_num, np.max([len(l) for l in g2m1_laglist])), dtype=float)
                g2m1_lags = np.nan * np.ones_like(g2m1, dtype=float)
                g2m1_avgnum = np.zeros_like(g2m1, dtype=int)
                for tidx in range(cur_cI.shape[0]):
                    cur_tavg_idx = tidx // self.timeAvg_T
                    #print((tidx, self.timeAvg_T, cur_tavg_idx, len(cur_tavg_idx)))
                    g2m1_lags[cur_tavg_idx,:len(g2m1_laglist[cur_tavg_idx])] = g2m1_laglist[cur_tavg_idx]
                    for lidx in range(cur_cI.shape[1]):
                        if (tidx < len(g2m1_alllags[lidx])):
                            cur_lagidx = np.argmin(np.abs(np.subtract(g2m1_laglist[cur_tavg_idx], g2m1_alllags[lidx][tidx])))
                            if (~np.isnan(cur_cI[tidx,lidx])):
                                g2m1_avgnum[cur_tavg_idx,cur_lagidx] += 1
                                g2m1[cur_tavg_idx,cur_lagidx] += cur_cI[tidx,lidx]
                g2m1 = np.divide(g2m1, g2m1_avgnum)
    
                str_hdr_g = str(TXT_DELIMITER).join(['dt'+TXT_DELIMITER+'t{0:.2f}'.format(cur_times[tavgidx*self.timeAvg_T]) for tavgidx in range(tavg_num)])
                g2m1_out = np.empty((g2m1.shape[1], 2*tavg_num), dtype=float)
                g2m1_out[:,0::2] = g2m1_lags.T
                g2m1_out[:,1::2] = g2m1.T

            np.savetxt(os.path.join(self.outFolder, G2M1_PREFIX + cur_f[2:]), g2m1_out, header=str_hdr_g, **self.savetxt_kwargs)
    
    def GetOrReadImage(self, img_idx, buffer=None):
        """ Retrieve image from buffer if present, otherwise read if from MIfile
        """
        if buffer is not None:
            if len(buffer) > img_idx:
                return buffer[img_idx]
        return self.MIinput.GetImage(img_idx)
    
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
        NormList : 2D array. Element [i,j] is the average of j-th ROI (independent of i)
        imgs : buffer eventually filled with images read from MIfile

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
            NormList = np.empty_like(AvgRes)
            for i in range(AvgRes.shape[0]):
                if stack2 is None:
                    AvgRes[i], NormList[i] = ppf.ROIAverage(self.GetOrReadImage(stack1[i], imgs), ROImasks, boolMask=masks_isBool)
                else:
                    if (stack1[i]==stack2[i]):
                        AvgRes[i], NormList[i] = ppf.ROIAverage(np.square(self.GetOrReadImage(stack1[i], imgs)), ROImasks, boolMask=masks_isBool)
                    else:
                        AvgRes[i], NormList[i] = ppf.ROIAverage(np.multiply(self.GetOrReadImage(stack1[i], imgs), self.GetOrReadImage(stack2[i], imgs)), ROImasks, boolMask=masks_isBool)
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
            
        return AvgRes, NormList, imgs
    
    def Run(self, doDLS=False, no_buffer=False):
        """ Run SLS/DLS analysis

        Parameters
        ----------
        doDLS : bool. If True, perform DLS alongside SLS. Otherwise, only perform SLS
        no_buffer : bool. If True, avoid reading full MIfile to RAM
        """
        
        ROI_boolMasks = [self.ROIs==b for b in range(self.CountROIs())]
        
        logging.info('SALS Analysis started! Will analyze {0} images ({1} times, {2} exposure times)'.format(self.ImageNumber(), self.NumTimes(), self.NumExpTimes()))
        if (self.ImageNumber() != (self.NumTimes() * self.NumExpTimes())):
            logging.warn('WARNING: Number of images ({0}) should correspond to the number of times ({1}) times the number of exposure times ({2})'.format(self.ImageNumber(), self.NumTimes(), self.NumExpTimes()))
        logging.info('Analysis will resolve {0} ROIs and DLS will be performed on {1} lagtimes. Output will be saved in folder {2}'.format(self.CountROIs(), self.NumLagtimes(), self.outFolder))
        logging.info('Now starting with SLS...')
        
        if no_buffer:
            self.MIinput.OpenForReading()
        
        all_avg, NormList, buf_images = self.ROIaverageProduct(stack1=list(range(self.ImageNumber())), stack2=None, ROImasks=ROI_boolMasks, masks_isBool=True, no_buffer=no_buffer)
        ROIavgs_allExp = all_avg.reshape((self.NumTimes(), self.NumExpTimes(), -1))
        ROIavgs_best = np.zeros((self.NumTimes(), ROIavgs_allExp.shape[-1]), dtype=float)
        BestExptime_Idx = -1 * np.ones_like(ROIavgs_best, dtype=int)
        for idx, val in np.ndenumerate(ROIavgs_best):
            BestExptime_Idx[idx] = min(bisect.bisect(ROIavgs_allExp[idx[0], :, idx[1]], self.MaxSafeAvgIntensity), len(self.expTimes)-1)
            ROIavgs_best[idx] = ROIavgs_allExp[idx[0], BestExptime_Idx[idx], idx[1]] / self.expTimes[BestExptime_Idx[idx]]
        self.SaveSLS(ROIavgs_best, NormList[0], [ROIavgs_allExp, BestExptime_Idx])
        logging.debug('SLS output saved')
        # TODO: time average SLS
                
        if doDLS:
            if self.dlsLags is None:
                logging.warn('No lagtimes specified for DLS')
            elif self.NumTimes()<=1:
                logging.warn('At least 1 timepoints needed for DLS (' + str(self.NumTimes()) + ' given)')
            else:
                logging.info('SLS analysis completed. Now doing DLS ({0} exposure times, {1} time points, {2} lagtimes)'.format(self.NumExpTimes(), self.NumTimes(), self.NumLagtimes()))
                for e in range(self.NumExpTimes()):
                    readrange = self.MIinput.Validate_zRange([e, -1, self.NumExpTimes()])
                    idx_list = list(range(*readrange))
                    logging.info('Now performing DLS on {0}-th exposure time. Using image range {1} ({2} images)'.format(e, readrange, len(idx_list)))
                    ISQavg, NormList, buf_images = self.ROIaverageProduct(stack1=idx_list, stack2=idx_list, ROImasks=ROI_boolMasks, masks_isBool=True, no_buffer=no_buffer, imgs=buf_images)
                    cI = np.nan * np.ones((ISQavg.shape[1], ISQavg.shape[0], self.NumLagtimes()), dtype=float)
                    cI[:,:,0] = np.subtract(np.divide(ISQavg, np.square(ROIavgs_allExp[:,e,:])), 1).T
                    if (self.DebugMode):
                        if np.any(cI[:,:,0] < 0):
                            logging.warn('Negative values in d0 contrast!')
                    for lidx in range(1, self.NumLagtimes()):
                        if (self.dlsLags[lidx]<ISQavg.shape[0]):
                            IXavg, NormList, buf_images = self.ROIaverageProduct(stack1=idx_list[:-self.dlsLags[lidx]], stack2=idx_list[self.dlsLags[lidx]:], 
                                                                     ROImasks=ROI_boolMasks, masks_isBool=True, no_buffer=no_buffer, imgs=buf_images)
                            # 'classic' cI formula
                            cI[:,:-self.dlsLags[lidx],lidx] = np.subtract(np.divide(IXavg, np.multiply(ROIavgs_allExp[:-self.dlsLags[lidx],e,:],
                                                                                                       ROIavgs_allExp[self.dlsLags[lidx]:,e,:])), 1).T
                            # d0 normalization
                            cI[:,:-self.dlsLags[lidx],lidx] = np.divide(cI[:,:-self.dlsLags[lidx],lidx], 0.5 * np.add(cI[:,:-self.dlsLags[lidx],0], cI[:,self.dlsLags[lidx]:,0]))
                            logging.info('Lagtime {0}/{1} (d{2}) completed'.format(lidx, self.NumLagtimes()-1, self.dlsLags[lidx]))
                    for ridx in range(cI.shape[0]):
                        logging.info('Now saving ROI {0} to file'.format(ridx))
                        np.savetxt(os.path.join(self.outFolder, CI_PREFIX+'_'+ROI_PREFIX + str(ridx).zfill(ROI_IDXLEN) + '_'+EXP_PREFIX + str(e).zfill(EXP_IDXLEN) + '.dat'), 
                                   np.append(self.imgTimes[idx_list].reshape((-1, 1)), cI[ridx], axis=1), header='t'+TXT_DELIMITER + str(TXT_DELIMITER).join(['d{0}'.format(l) for l in self.dlsLags]), **self.savetxt_kwargs)

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
        
    def _genROIs(self, ROI_specs, maskRaw=None):
        if ROI_specs is None:
            ROI_specs = [None, None]
        self.ROIs, self.ROIcoords = GenerateROIs(ROI_specs, imgShape=self.MIinput.ImageShape(), centerPos=self.centerPos, maskRaw=maskRaw)
        if self.CountEmptyROIs() > 0:
            if self.CountValidROIs() > 0:
                logging.warning('There are {0} out of {1} empty masks'.format(self.CountEmptyROIs(), self.CountROIs()))
            else:
                logging.error('ROI mask is empty (no valid ROIs found)')
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
        self.dt_tolerance = 1e-2  #1e-4
        self.dt_tolerance_isrelative = True
        self.DebugMode = False
        self.savetxt_kwargs = {'delimiter':TXT_DELIMITER, 'comments':TXT_COMMENT}
        self.loadtxt_kwargs = {**self.savetxt_kwargs, 'skiprows':1}
