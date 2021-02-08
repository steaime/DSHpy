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

def LoadFromConfig(ConfigFile, input_key='input', outFolder=None):
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
    miin_fname = config.Get(input_key, 'mi_file', None, str)
    miin_meta_fname = config.Get(input_key, 'meta_file', None, str)
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
        angRange = sf.ValidateRange(config.Get('SALS_parameters', 'a_range', None, float), 2*np.pi, replaceNone=True)
        rSlices = np.geomspace(radRange[0], radRange[1], int(radRange[2])+1, endpoint=True)
        aSlices = np.linspace(angRange[0], angRange[1], int(angRange[2])+1, endpoint=True)
        if (outFolder is None):
            outFolder = config.Get(input_key, 'out_folder', None, str)
            if (outFolder is not None):
                outFolder = os.path.join(config.Get('global', 'root', '', str), outFolder)
        mask = config.Get('SALS_parameters', 'px_mask', None, str)
        mask = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_key, 'px_mask', mask, str)),
                             MIin.ImageShape(), MIin.DataFormat(), 0)
        dark = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_key, 'dark_bkg', None, str)), 
                             MIin.ImageShape(), MIin.DataFormat(), 0)
        opt = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get(input_key, 'opt_bkg', None, str)), 
                            MIin.ImageShape(), MIin.DataFormat(), 0)
        PD_data = sf.PathJoinOrNone(froot, config.Get(input_key, 'pd_file', None, str))
        if (PD_data is not None):
            PD_data = np.loadtxt(PD_data, dtype=float)
        img_times = sf.PathJoinOrNone(froot, config.Get(input_key, 'img_times', None, str))
        if (img_times is not None):
            img_times = np.loadtxt(img_times, dtype=float, usecols=config.Get('format', 'img_times_colidx', 0, int), skiprows=1)
        exp_times = sf.PathJoinOrNone(froot, config.Get(input_key, 'exp_times', None, str))
        if (exp_times is not None):
            exp_times = np.unique(np.loadtxt(exp_times, dtype=float, usecols=config.Get('format', 'exp_times_colidx', 0, int)))
        dlsLags = config.Get('SALS_parameters', 'dls_lags', None, int)
        tavgT = config.Get('SALS_parameters', 'timeavg_T', None, int)
        return SALS(MIin, outFolder, ctrPos, [rSlices, aSlices], mask, dark, opt, PD_data, img_times, exp_times, dlsLags, tavgT)

class SALS():
    """ Class to do small angle static and dynamic light scattering from a MIfile """
    
    def __init__(self, MIin, outFolder, centerPos, ROIs=None, maskRaw=None, DarkBkg=None, OptBkg=None, 
                 PDdata=None, imgTimes=None, expTimes=[1], dlsLags=None, timeAvg_T=None):
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
        
        # TODO: replace this with loading from ini config file, move initialization to separate function
        self.MIinput   = MIin
        self.outFolder = outFolder
        sf.CheckCreateFolder(self.outFolder)
        self.centerPos = centerPos
        if ROIs is None:
            ROIs = [None, None]
        if (len(ROIs)==2):
            rSlices, aSlices = ROIs
            self.ROIcoords = ppf.PolarMaskCoords(rSlices, aSlices, flatten_res=True)
            self.ROIs = ppf.GenerateMasks(self.ROIcoords, self.MIinput.ImageShape(), center=self.centerPos,
                                          common_mask=maskRaw, binary_res=False, coordsystem='polar')
            logging.debug('ROI mask with shape ' + str(self.ROIs.shape) + ' saved to file ' + os.path.join(outFolder, 'ROI_mask.raw'))
        else:
            self.ROIs = ROIs
            r_map, a_map = ppf.GenerateGrid2D(ROIs.shape, center=centerPos, coords='polar')
            r_min, r_max = ppf.ROIEval(r_map, ROIs, [np.min, np.max])
            a_min, a_max = ppf.ROIEval(a_map, ROIs, [np.min, np.max])
            rSlices, aSlices = [[r_min[i], r_max[i]] for i in range(len(r_min))], [[a_min[i], a_max[i]] for i in range(len(a_min))]
            self.ROIcoords = ppf.PolarMaskCoords(rSlices, aSlices, flatten_res=True)
        if self.CountEmptyROIs() > 0:
            if self.CountValidROIs() > 0:
                logging.warning('There are {0} out of {1} empty masks'.format(self.CountEmptyROIs(), self.CountROIs()))
            else:
                logging.error('ROI mask is empty (no valid ROIs found)')
        if (isinstance(DarkBkg, MI.MIfile)):
            self.DarkBkg = DarkBkg.zAverage()
        else:
            self.DarkBkg = DarkBkg
        if (isinstance(OptBkg, MI.MIfile)):
            self.OptBkg = OptBkg.zAverage()
        else:
            self.OptBkg = OptBkg
        self.PDdata    = PDdata
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
        self.HistogramSLS = False
        
        self.MaxSafeAvgIntensity = 40
        self.dt_tolerance = 1e-2  #1e-4
        self.dt_tolerance_isrelative = True
        self.DebugMode = False
        self.savetxt_kwargs = {'delimiter':'\t', 'comments':'#'}
        self.loadtxt_kwargs = {**self.savetxt_kwargs, 'skiprows':1}
        
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
                   header='r[px]\tphi[rad]\tdr[px]\tdphi[rad]\tnorm', **self.savetxt_kwargs)
        MI.WriteBinary(os.path.join(self.outFolder, 'ROI_mask.raw'), self.ROIs, 'i')
        str_hdr_Ir = 'r[px]\tphi[rad]' + ''.join(['\tt{0:.2f}'.format(self.imgTimes[i]) for i in range(0, self.ImageNumber(), self.NumExpTimes())])
        np.savetxt(os.path.join(self.outFolder, 'I_r.dat'), np.append(self.ROIcoords[:IofR.shape[1],:2], IofR.T, axis=1), 
                   header=str_hdr_Ir, **self.savetxt_kwargs)
        if AllExpData is not None:
            ROIavgs_allExp, BestExptime_Idx = AllExpData
            np.savetxt(os.path.join(self.outFolder, 'exptimes.dat'), np.append(self.ROIcoords[:,:2], BestExptime_Idx.T, axis=1), 
                       header=str_hdr_Ir, **self.savetxt_kwargs)
            str_hdr_raw = 'r[px]\tphi[rad]' + ''.join(['\tt{0:.2f}_e{1:.3f}'.format(self.imgTimes[i], self.expTimes[i%len(self.expTimes)]) for i in range(len(self.imgTimes))])
            np.savetxt(os.path.join(self.outFolder, 'I_r_raw.dat'), np.append(self.ROIcoords[:,:2], ROIavgs_allExp.reshape((-1, ROIavgs_allExp.shape[-1])).T, axis=1), 
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
        cI_fnames = sf.FindFileNames(self.outFolder, Prefix='cI_', Ext='.dat')
        
        for cur_f in cI_fnames:
            cur_cI, cur_times, roi_idx, exp_idx = self.ReadCIfile(cur_f)
            tavg_num = cur_cI.shape[0] // self.timeAvg_T
            logging.debug('cI time averages will be performed by dividing the {0} images into {1} windows of {2} images each'.format(self.NumTimes(), tavg_num, self.timeAvg_T))
            
            if self.IsTimingConstant(cur_times):
                g2m1_lags = np.true_divide(self.dlsLags, self.MIinput.GetFPS())
                g2m1 = np.nan * np.ones((self.NumLagtimes(), tavg_num), dtype=float)
                for tavgidx in range(tavg_num):
                    g2m1[:,tavgidx] = np.nanmean(cur_cI[tavgidx*self.timeAvg_T:(tavgidx+1)*self.timeAvg_T,:], axis=0)
                str_hdr_g = self.txt_comment + 'dt' + ''.join(['\tt{0:.2f}'.format(cur_times[tavgidx*self.timeAvg_T]) for tavgidx in range(tavg_num)])
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
    
                str_hdr_g = '\t'.join(['dt\tt{0:.2f}'.format(cur_times[tavgidx*self.timeAvg_T]) for tavgidx in range(tavg_num)])
                g2m1_out = np.empty((g2m1.shape[1], 2*tavg_num), dtype=float)
                g2m1_out[:,0::2] = g2m1_lags.T
                g2m1_out[:,1::2] = g2m1.T

            np.savetxt(os.path.join(self.outFolder, 'g2m1' + cur_f[2:]), g2m1_out, header=str_hdr_g, **self.savetxt_kwargs)
        
    
    
    def Run(self, doDLS=False, no_buffer=False):
        """ Run SLS/DLS analysis

        Parameters
        ----------
        doDLS : bool. If True, perform DLS alongside SLS. Otherwise, only perform SLS
        no_buffer : bool. If True, avoid reading full MIfile to RAM

        Returns
        -------
        None.

        """
        
        ROI_boolMasks = [self.ROIs==b for b in range(self.CountROIs())]
        
        if (self.StackInput() or no_buffer):
            buf_images = None
            all_avg = np.nan*np.ones((self.ImageNumber(), self.CountROIs), dtype=float)
            NormList = np.empty_like(all_avg)
            for i in range(self.ImageNumber()):
                all_avg[i], NormList[i] = ppf.ROIAverage(self.MIinput.GetImage(i), ROI_boolMasks, boolMask=True)
        else:
            buf_images = self.MIinput.Read()
            all_avg, NormList = ppf.ROIAverage(buf_images, ROI_boolMasks, boolMask=True)
            
        ROIavgs_allExp = all_avg.reshape((self.NumTimes(), self.NumExpTimes(), -1))
        ROIavgs_best = np.zeros((self.NumTimes(), ROIavgs_allExp.shape[-1]), dtype=float)
        BestExptime_Idx = -1 * np.ones_like(ROIavgs_best, dtype=int)
        for idx, val in np.ndenumerate(ROIavgs_best):
            BestExptime_Idx[idx] = min(bisect.bisect(ROIavgs_allExp[idx[0], :, idx[1]], self.MaxSafeAvgIntensity), len(self.expTimes)-1)
            ROIavgs_best[idx] = ROIavgs_allExp[idx[0], BestExptime_Idx[idx], idx[1]] / self.expTimes[BestExptime_Idx[idx]]
        self.SaveSLS(ROIavgs_best, NormList[0], [ROIavgs_allExp, BestExptime_Idx])
        # TODO: time average SLS
                
        if doDLS:
            if self.dlsLags is None:
                logging.warn('No lagtimes specified for DLS')
            elif self.NumTimes()<=1:
                logging.warn('At least 1 timepoints needed for DLS (' + str(self.NumTimes()) + ' given)')
            else:
                str_hdr_cI = 't' + '\t'.join(['d{0}'.format(l) for l in self.dlsLags])
                for e in range(self.NumExpTimes()):
                    readrange = self.MIinput.Validate_zRange([e, -1, self.NumExpTimes()])
                    idx_list = list(range(*readrange))
                    if (self.StackInput() or no_buffer):
                        ISQavg = np.nan*np.ones((len(idx_list), self.CountROIs), dtype=float)
                        NormList = np.empty_like(ISQavg)
                        for i in range(len(idx_list)):
                            ISQavg[i], NormList[i] = ppf.ROIAverage(self.MIinput.GetImage(idx_list[i]), ROI_boolMasks, boolMask=True)
                    else:
                        if (buf_images is None):
                            imgs = self.MIinput.Read(readrange, cropROI=None)
                        else:
                            imgs = buf_images[readrange[0]:readrange[1]:readrange[2]]
                        ISQavg, NormList = ppf.ROIAverage(np.square(imgs), ROI_boolMasks, boolMask=True)
                    cI = np.nan * np.ones((ISQavg.shape[1], ISQavg.shape[0], self.NumLagtimes()), dtype=float)
                    cI[:,:,0] = np.subtract(np.divide(ISQavg, np.square(ROIavgs_allExp[:,e,:])), 1).T
                    for lidx in range(1, self.NumLagtimes()):
                        if (self.dlsLags[lidx]<ISQavg.shape[0]):
                            if (self.StackInput() or no_buffer):
                                IXavg = np.nan*np.ones((ISQavg.shape[0]-self.dlsLags[lidx], self.CountROIs), dtype=float)
                                NormList = np.empty_like(IXavg)
                                for i in range(len(idx_list)):
                                    if (idx_list[i] + self.dlsLags[lidx] < ISQavg.shape[0]):
                                        IXavg[i], NormList[i] = ppf.ROIAverage(np.multiply(imgs[idx_list[i]], imgs[idx_list[i]+self.dlsLags[lidx]]), 
                                                                               ROI_boolMasks, boolMask=True)
                            else:
                                IXavg, NormList = ppf.ROIAverage(np.multiply(imgs[:-self.dlsLags[lidx]], imgs[self.dlsLags[lidx]:]), ROI_boolMasks, boolMask=True)
                            # 'classic' cI formula
                            cI[:,:-self.dlsLags[lidx],lidx] = np.subtract(np.divide(IXavg, np.multiply(ROIavgs_allExp[:-self.dlsLags[lidx],e,:],
                                                                                                       ROIavgs_allExp[self.dlsLags[lidx]:,e,:])), 1).T
                            # d0 normalization
                            cI[:,:-self.dlsLags[lidx],lidx] = np.divide(cI[:,:-self.dlsLags[lidx],lidx], 0.5 * np.add(cI[:,:-self.dlsLags[lidx],0], cI[:,self.dlsLags[lidx]:,0]))
                    for ridx in range(cI.shape[0]):
                        np.savetxt(os.path.join(self.outFolder, 'cI_ROI' + str(ridx).zfill(3) + '_e' + str(e).zfill(2) + '.dat'), 
                                   np.append(self.imgTimes.reshape((-1, 1)), cI[ridx], axis=1), header=str_hdr_cI, **self.savetxt_kwargs)

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