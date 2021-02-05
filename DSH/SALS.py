import os
import bisect
import numpy as np
import time
import logging
from scipy import signal
import DSH
from DSH import Config as cf
from DSH import MIfile as MI
from DSH import MIstack as MIs
from DSH import SharedFunctions as sf
from DSH import PostProcFunctions as ppf

def LoadFromConfig(ConfigFile, outFolder=None):
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
    miin_fname = config.Get('input', 'mi_file', None, str)
    miin_meta_fname = config.Get('input', 'meta_file', None, str)
    if (miin_fname is not None):
        miin_fname = os.path.join(froot, miin_fname)
    if (miin_meta_fname is not None):
        miin_meta_fname = os.path.join(froot, miin_meta_fname)
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
            outFolder = config.Get('input', 'out_folder', None, str)
            if (outFolder is not None):
                outFolder = os.path.join(config.Get('global', 'root', '', str), outFolder)
        mask = config.Get('SALS_parameters', 'px_mask', None, str)
        mask = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get('input', 'px_mask', mask, str)),
                             MIin.ImageShape(), MIin.DataFormat(), 0)
        dark = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get('input', 'dark_bkg', None, str)), 
                             MIin.ImageShape(), MIin.DataFormat(), 0)
        opt = MI.ReadBinary(sf.PathJoinOrNone(froot, config.Get('input', 'opt_bkg', None, str)), 
                            MIin.ImageShape(), MIin.DataFormat(), 0)
        PD_data = sf.PathJoinOrNone(froot, config.Get('input', 'pd_file', None, str))
        if (PD_data is not None):
            PD_data = np.loadtxt(PD_data, dtype=float)
        img_times = sf.PathJoinOrNone(froot, config.Get('input', 'img_times', None, str))
        if (img_times is not None):
            img_times = np.loadtxt(img_times, dtype=float, usecols=config.Get('format', 'img_times_colidx', 0, int))
        exp_times = sf.PathJoinOrNone(froot, config.Get('input', 'exp_times', None, str))
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
        self.dt_tolerance = 1e-4
        self.txt_comment = '#'
        self.DebugMode = False
        
    def CountROIs(self):
        return self.ROIcoords.shape[0]
    def CountEmptyROIs(self):
        return self.CountROIs() - self.CountValidROIs()
    def CountValidROIs(self):
        return np.max(self.ROIs)+1
    def ImageNumber(self):
        return self.MIinput.ImageNumber()
    def IsTimingConstant(self, tolerance=None):
        if len(self.imgTimes) <= 1:
            return True
        else:
            dt_arr = np.diff(self.imgTimes)
            if tolerance is None:
                tolerance = self.dt_tolerance
            return ((np.max(dt_arr) - np.min(dt_arr)) < tolerance)
    
    def Run(self, doDLS=False):
        t_num = self.MIinput.ImageNumber() // len(self.expTimes)
        r_num = self.ROIcoords.shape[0]
        ROIavgs_allExp = np.zeros((t_num, len(self.expTimes), r_num), dtype=float)
        ROIhistFit_allExp = np.zeros((t_num, len(self.expTimes), r_num), dtype=float)
        ROIhistFit_weights = np.ones((t_num, len(self.expTimes), r_num), dtype=float)
        self.MIinput.OpenForReading()
        for i in range(self.MIinput.ImageNumber()):
            tidx, expidx = i//len(self.expTimes), i%len(self.expTimes)
            curavg, NormList = ppf.ROIAverage(self.MIinput.GetImage(i), self.ROIs)
            ROIavgs_allExp[tidx, expidx, :len(curavg)] = curavg
            if (self.HistogramSLS):
                # TODO: implement this:
                # - compute intensity histogram
                # - select relevant part (ex: set max to min(max_val, 250) and min to max(hist)+1)
                # - fit log(pdf) vs I with a line
                # - think about how to relate the slope to the average intensity, it is straightforward
                # - think about how to estimate the uncertainty on the average (to properly average all exposure times afterwards)
                pass
        ROIavgs_best = np.zeros((t_num, r_num), dtype=float)
        BestExptime_Idx = -1 * np.ones_like(ROIavgs_best, dtype=int)
        for tidx in range(t_num):
            for ridx in range(r_num):
                BestExptime_Idx[tidx, ridx] = min(bisect.bisect(ROIavgs_allExp[tidx, :, ridx], self.MaxSafeAvgIntensity), len(self.expTimes)-1)
                ROIavgs_best[tidx, ridx] = ROIavgs_allExp[tidx, BestExptime_Idx[tidx, ridx], ridx] / self.expTimes[BestExptime_Idx[tidx, ridx]]
        # TODO: weighted average of histFits
        # TODO: dark and opt correction
        
        roi_norms = np.zeros((self.ROIcoords.shape[0], 1))
        roi_norms[:len(NormList),0] = NormList
        np.savetxt(os.path.join(self.outFolder, 'ROIcoords.dat'), np.append(self.ROIcoords, roi_norms, axis=1), header=self.txt_comment+'r[px]\tphi[rad]\tdr[px]\tdphi[rad]\tnorm', delimiter='\t', comments = '')
        MI.WriteBinary(os.path.join(self.outFolder, 'ROI_mask.raw'), self.ROIs, 'i')
        str_hdr_Ir = self.txt_comment + 'r[px]\tphi[rad]' + ''.join(['\tt{0:.2f}'.format(self.imgTimes[i]) for i in range(0, self.MIinput.ImageNumber(), len(self.expTimes))])
        np.savetxt(os.path.join(self.outFolder, 'I_r.dat'), np.append(self.ROIcoords[:,:2], ROIavgs_best.T, axis=1), header=str_hdr_Ir, delimiter='\t', comments='')
        np.savetxt(os.path.join(self.outFolder, 'exptimes.dat'), np.append(self.ROIcoords[:,:2], BestExptime_Idx.T, axis=1), header=str_hdr_Ir, delimiter='\t', comments='')
        str_hdr_raw = self.txt_comment + 'r[px]\tphi[rad]' + ''.join(['\tt{0:.2f}_e{1:.3f}'.format(self.imgTimes[i], self.expTimes[i%len(self.expTimes)]) for i in range(len(self.imgTimes))])
        np.savetxt(os.path.join(self.outFolder, 'I_r_raw.dat'), np.append(self.ROIcoords[:,:2], ROIavgs_allExp.reshape((-1, r_num)).T, axis=1), header=str_hdr_raw, delimiter='\t', comments='')
        
        # TODO: time average SLS
        if self.timeAvg_T is None:
            self.timeAvg_T = t_num
        tavg_num = t_num // self.timeAvg_T
        dt_iscst = self.IsTimingConstant()
        logging.debug('cI time averages will be performed by dividing the {0} images into {1} windows of {2} images each'.format(t_num, tavg_num, self.timeAvg_T))
        
        if doDLS:
            if self.dlsLags is None:
                logging.warn('No lagtimes specified for DLS')
            elif t_num<=1:
                logging.warn('At least 1 timepoints needed for DLS (' + str(t_num) + ' given)')
            else:
                for e in range(len(self.expTimes)):
                    readrange = self.MIinput.Validate_zRange([e, -1, len(self.expTimes)])
                    imgs = self.MIinput.Read(readrange, cropROI=None)
                    if self.DebugMode:
                        logging.debug('Now doing DLS on exposure time {0}/{1} ({2:.3f} ms) : reading range {3}'.format(e+1, len(self.expTimes), self.expTimes[e], readrange))
                        cIraw = np.nan * np.ones((r_num, t_num, len(self.dlsLags)), dtype=float)
                    cI = np.nan * np.ones((r_num, t_num, len(self.dlsLags)), dtype=float)
                    for tidx in range(t_num):
                        ISQavg, NormList = ppf.ROIAverage(np.square(imgs[tidx]), self.ROIs)
                        if self.DebugMode:
                            cIraw[:len(ISQavg),tidx,0] = ISQavg
                        cI[:len(ISQavg),tidx,0] = np.subtract(np.divide(ISQavg, np.square(ROIavgs_allExp[tidx,e,:len(ISQavg)])), 1)
                    for tidx in range(t_num):
                        for lidx in range(1, len(self.dlsLags)):
                            if self.DebugMode:
                                logging.debug('   t={0}, lag={1}: processing images {2} and {3}'.format(tidx, self.dlsLags[lidx], tidx, tidx+self.dlsLags[lidx]))
                            if (tidx+self.dlsLags[lidx]<imgs.shape[0]):
                                IXavg, NormList = ppf.ROIAverage(np.multiply(imgs[tidx], imgs[tidx+self.dlsLags[lidx]]), self.ROIs)
                                if self.DebugMode:
                                    cIraw[:len(IXavg),tidx,lidx] = IXavg
                                cI[:len(IXavg),tidx,lidx] = np.divide(np.subtract(np.divide(IXavg, np.multiply(ROIavgs_allExp[tidx,e,:len(ISQavg)],
                                                                                                               ROIavgs_allExp[tidx+self.dlsLags[lidx]*len(self.expTimes),e,:len(ISQavg)])), 1),
                                                                      np.multiply(0.5, np.add(cI[:len(IXavg),tidx,0], cI[:len(IXavg),tidx+self.dlsLags[lidx],0])))
                    str_hdr_cI = self.txt_comment + 't' + '\t'.join(['d{0}'.format(l) for l in self.dlsLags])
                    for ridx in range(cI.shape[0]):
                        np.savetxt(os.path.join(self.outFolder, 'cI_ROI' + str(ridx).zfill(3) + '_e' + str(e).zfill(2) + '.dat'), np.append(self.imgTimes.reshape((-1, 1)), cI[ridx], axis=1), header=str_hdr_cI, delimiter='\t', comments='')
                        if self.DebugMode:
                            np.savetxt(os.path.join(self.outFolder, 'cIraw_ROI' + str(ridx).zfill(3) + '_e' + str(e).zfill(2) + '.dat'), np.append(self.imgTimes.reshape((-1, 1)), cIraw[ridx], axis=1), header=str_hdr_cI, delimiter='\t', comments='')
        
                    if dt_iscst:
                        g2m1_lags = np.true_divide(self.dlsLags, self.MIinput.GetFPS())
                        g2m1 = np.nan * np.ones((r_num, tavg_num, len(self.dlsLags)), dtype=float)
                        for tavgidx in range(tavg_num):
                            g2m1[:,tavgidx,:] = np.nanmean(cI[:,tavgidx*self.timeAvg_T:(tavgidx+1)*self.timeAvg_T,:], axis=1)
                    else:
                        g2m1_alllags = np.nan * np.ones((t_num, len(self.dlsLags)), dtype=float)
                        for tidx in range(t_num):
                            for lidx in range(len(self.dlsLags)):
                                if (tidx+self.dlsLags[lidx]*len(self.expTimes) < t_num):
                                    g2m1_alllags[tidx, lidx] = self.imgTimes[tidx+self.dlsLags[lidx]*len(self.expTimes)] - self.imgTimes[tidx]
                        g2m1_laglist = []
                        g2m1_lenlags = np.zeros(tavg_num, dtype=int)
                        for tavgidx in range(tavg_num):
                            cur_uniquelist = np.unique(g2m1_alllags[tavgidx*self.timeAvg_T:(tavgidx+1)*self.timeAvg_T,:])
                            cur_coarsenedlist = [cur_uniquelist[0]]
                            for lidx in range(1, len(cur_uniquelist)):
                                if np.abs(cur_uniquelist[lidx]-cur_coarsenedlist[-1]) > self.dt_tolerance:
                                    cur_coarsenedlist.append(cur_uniquelist[lidx])
                            g2m1_laglist.append(cur_coarsenedlist)
                            g2m1_lenlags[tavgidx] = len(g2m1_laglist[-1])
                        logging.debug('Maximum number of unique lagtime values: {0}'.format(np.max(g2m1_lenlags)))
                        g2m1_lags = np.nan * np.ones((tavg_num, np.max(g2m1_lenlags)), dtype=float)
                        g2m1_lagidx = np.nan * np.ones_like(g2m1_alllags, dtype=int)
                        g2m1_avgnum = np.zeros_like(g2m1_lags, dtype=int)
                        g2m1 = np.zeros((r_num, tavg_num, g2m1_lags.shape[-1]), dtype=float)
                        for tavgidx in range(tavg_num):
                            g2m1_lags[tavgidx,:len(g2m1_laglist[tavgidx])] = g2m1_laglist[tavgidx]
                        for tidx in range(t_num):
                            cur_tavg_idx = tidx // self.timeAvg_T
                            for lidx in range(len(self.dlsLags)):
                                cur_lagidx = np.argmin(np.abs(g2m1_lags[cur_tavg_idx]-g2m1_alllags[tidx, lidx]))
                                cur_lagdiff = np.abs(g2m1_lags[cur_tavg_idx, cur_lagidx]-g2m1_alllags[tidx, lidx])
                                if (cur_lagdiff > self.dt_tolerance):
                                    logging.warn('Time #{0} - lagtime #{1} ({2}) WARNING: current lagtime ({3:.3f}) was not found in list of unique lagtimes. '.format(tidx, lidx, self.dlsLags[lidx], g2m1_alllags[tidx, lidx]) +
                                                 'Closest lagtime found (#{0}, dt={1:.3f}) exceeds expected tolerance ({2:.2e})'.format(cur_lagidx, cur_lagdiff, self.dt_tolerance))
                                g2m1_lagidx[tidx, lidx] = cur_lagidx
                                num_nan = np.count_nonzero(np.isnan(cI[:,tidx,lidx]))
                                if (num_nan==0):
                                    g2m1_avgnum[cur_tavg_idx,cur_lagidx] += 1
                                    g2m1[:,cur_tavg_idx,cur_lagidx] += cI[:,tidx,lidx]
                                elif (num_nan < len(cI[:,tidx,lidx])):
                                    logging.warn('Time #{0} - lagtime #{1} ({2}) WARNING: {3} NaN values found in cI out of {4}. This is unexpected (it should be either all or none). Lagtime discarded.'.format(tidx, lidx, self.dlsLags[lidx], num_nan, len(cI[:,tidx,lidx])))
                        for ridx in range(r_num):
                            g2m1[ridx] = np.divide(g2m1[ridx], g2m1_avgnum)
        
                    for ridx in range(r_num):
                        if dt_iscst:
                            str_hdr_g = self.txt_comment + 'dt' + ''.join(['\tt{0:.2f}'.format(self.imgTimes[tavgidx*self.timeAvg_T]) for tavgidx in range(tavg_num)])
                            g2m1_out = np.append(g2m1_lags.reshape((-1, 1)), g2m1[ridx], axis=0).T
                        else:
                            str_hdr_g = self.txt_comment + '\t'.join(['dt\tt{0:.2f}'.format(self.imgTimes[tavgidx*self.timeAvg_T]) for tavgidx in range(tavg_num)])
                            g2m1_out = np.empty((g2m1_lags.shape[1], 2*tavg_num), dtype=float)
                            g2m1_out[:,0::2] = g2m1_lags.T
                            g2m1_out[:,1::2] = g2m1[ridx].T
                        np.savetxt(os.path.join(self.outFolder, 'g2m1_ROI' + str(ridx).zfill(3) + '_e' + str(e).zfill(2) + '.dat'), g2m1_out, header=str_hdr_g, delimiter='\t', comments='')
                        