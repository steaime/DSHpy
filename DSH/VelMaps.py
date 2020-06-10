import os
import numpy as np
import bisect
import scipy.special as ssp
from scipy import stats
from scipy.ndimage import binary_opening
import time
from multiprocessing import Process

import emcee
import pandas as pd

from DSH import Config as cf
from DSH import MIfile as MI
from DSH import CorrMaps as CM
from DSH import SharedFunctions as sf

def log_prior_corr(param_guess, prior_avg_params, prior_std_params, reg_param):
    """
    returns log of prior probability distribution
    
    Parameters:
        param_guess: guess for parameters (array of length N+2):
                    param_guess[0] will be the guess for d0
                    param_guess[1] will be the guess for baseline 
                    param_guess[2:] will be v(t) (array of length N)
        prior_avg_params: prior average parameters
        prior_std_params: prior standard deviation on parameters
        reg_param : regularization parameter penalizing fluctuations on v(t)
    """
    if reg_param<=0:
        res = 0
    else:
        v_grad = np.diff(param_guess[2:])
        rel_grad = np.true_divide(v_grad, param_guess[2:-1])
        del_gradgrad = np.diff(rel_grad)
        res = -reg_param * np.sum(np.square(del_gradgrad))
    if prior_avg_params is not None:
        residual = np.square(np.subtract(param_guess, prior_avg_params))
        chi_square = np.nansum(np.true_divide(residual,np.square(prior_std_params)))
        constant = np.nansum(np.log(np.true_divide(1.0, np.sqrt(np.multiply(2.0*np.pi,np.square(prior_std_params))))))
        res += constant - 0.5*chi_square
    return res

def log_likelihood_corr(param_guess, corr_func, obs_corr, obs_err, lagtimes, q):
    """
    returns log of likelihood
    
    Parameters:
        param_guess: guess for parameters (array of length N+2):
                    param_guess[0] will be the guess for d0
                    param_guess[1] will be the guess for baseline 
                    param_guess[2:] will be v(t) (array of length N)
        corr_func: function to compute correlations from displacements
        obs_displ: observed g2(t, tau)-1 for M delays tau (matrix M x N)
        obs_err: uncertainties on correlations (matrix M x N)
        q: scattering vector (float)
    """
    # Displacement matrix
    cur_corr = corr_func(compute_displ_multilag(param_guess[2:], lagtimes), q, param_guess[0], param_guess[1])
    # Residual matrix
    residual = np.square(np.subtract(cur_corr, obs_corr))
    # Chi square = Residual/Variance (must use nansum, there are NaNs in the displacement matrix)
    chi_square = np.nansum(np.true_divide(residual,np.square(obs_err)))
    # This constant is 1/sqrt(2 pi sigma^2), the Gaussian normalization
    constant = np.nansum(np.log(np.true_divide(1.0, np.sqrt(np.multiply(2.0*np.pi,np.square(obs_err))))))
    return constant - 0.5*chi_square

def log_posterior_corr(param_guess, corr_func, obs_corr, obs_err, avg_params, std_params, lagtimes, q, reg_param=0):
    """
    returns log of posterior probability distribution
    """
    return log_prior_corr(param_guess, avg_params, std_params, reg_param) +\
            log_likelihood_corr(param_guess, corr_func, obs_corr, obs_err, lagtimes, q)

def corr_std_calc(corr, baseline=0.1, rolloff=0.15):
    """Function calculating uncertainty on correlation values
        To prevent low correlations (more affected by noise on baseline and
        spontaneous decorrelation) from dominating the signal, which is 
        only mildly dependent on low correlations, introduce an exponential
        increase of uncertainties beyond a rolloff correlation
    """
    # Need to clip here and there to avoid numerical overflow and runtime warnings
    corr_clip = np.clip(corr, a_min=1e-6, a_max=1-1e-6)
    return baseline*np.exp(np.clip(np.power(np.true_divide(rolloff, corr_clip), 1), a_min=-10, a_max=10))

def compute_displ_multilag(v, lagtimes, dt=1.0):
    """Compute discrete displacements from a velocity array and a list of lagtimes
    
    Parameters
    ----------
    v: array of instantaneous velocities (n timepoints, spaced by dt)
    lagtimes: (int) list of (positive) time delays, in units of dt (m delays)
    dt: time lag between points
    
    Returns
    -------
    dx: m x n matrix of finite displacements
    element [i,j] will be integral of v between j and j+lagtimes[i]
    if j+lagtimes[i]<len(v), it will be np.nan
    """
    res = np.ones((len(lagtimes), len(v)))*np.nan
    csums = np.cumsum(v)
    assert lagtimes.ndim==1
    for i in range(len(lagtimes)):
        res[i,:-lagtimes[i]] = dt*np.subtract(csums[lagtimes[i]:], csums[:-lagtimes[i]])
    return res

def g2m1_parab(displ, q, d0=1.0, baseline=0):
    """Simulates correlation data from given displacements
        under the assumption of parabolic velocity profile
    
    Parameters
    ----------
    q: scattering vector (float), in units of inverse displacements
    d0: correlation value limit at zero delay. Lower than 1.0 because of camera noise
    baseline: baseline for correlation function. Larger than 0.0 because of stray light
    """
    if (q == 1.0):
        dotpr = displ
    else:
        dotpr = np.multiply(q, displ)
    return (np.pi/4)*np.true_divide(np.square(np.abs(ssp.erf(np.sqrt(-1j*dotpr)))),np.abs(dotpr))

def g2m1_affine(displ, q, d0=1.0, baseline=0):
    """Simulates correlation data from given displacements
        under the assumption of affine velocity profile
    
    Parameters
    ----------
    q: scattering vector (float), in units of inverse displacements
    d0: correlation value limit at zero delay. Lower than 1.0 because of camera noise
    baseline: baseline for correlation function. Larger than 0.0 because of stray light
    """
    if (q == 1.0):
        dotpr = displ
    else:
        dotpr = np.multiply(q, displ)
    return np.square(np.true_divide(np.sin(dotpr),dotpr))


def g2m1_sample(zProfile='Parabolic', q=1.0, d0=1.0, baseline=0.0, sample_dr=None, max_dr=None, step_dr=0.001):
    """Generate a lookup table for inverting correlation function
        to give the dot product q*dr, where q is the scattering vector 
        and dt is the displacement cumulated over time delay tau and
        projected along q
        NOTE: by default displacements are sorted in descending order, and truncated so that
            correlations are increasing monotonically, to use _invert_monotonic
    
    Parameters
    ----------
    zProfile : velocity profile for which correlation have to be modeled
                available options: Parabolic | Affine
    q: scattering vector (float), in units of inverse displacements
    d0: correlation value limit at zero delay. Lower than 1.0 because of camera noise
    baseline: baseline for correlation function. Larger than 0.0 because of stray light
    sample_dr: if None, displacements will be linearly sampled and a 2D LUT will be returned
                otherwise, correlations will be computed using sample_dr as input displacements
                and a 1D result will be returned
    max_dr: sample until this maximum displacement (only used if sample_dr is None)
            if None, it will sample until the first correlation minimum
    step_dr: generate linearly spaced displacement points with this step (only used if sample_dr is None)
    """
    if (zProfile.upper()=='PARABOLIC'):
        model = g2m1_parab
        if max_dr is None:
            max_dr = 4.5
    elif (zProfile.upper()=='AFFINE'):
        model = g2m1_affine
        if max_dr is None:
            max_dr = 2.55
    else:
        raise ValueError(str(zProfile) + 'z profile not implemented yet')
    if sample_dr is None:
        dr_g = np.zeros([2,int(max_dr/step_dr)])
        dr_g[0] = np.linspace(max_dr, step_dr, num=dr_g.shape[1])
        dr_g[1] = model(dr_g[0], q, d0, baseline)
    else:
        dr_g = model(sample_dr, q, d0, baseline)
    return dr_g

def invert_monotonic(data, _lut, overflow_to_nan=False):
    """Invert monotonic function based on a lookup table
        NOTE: data in the lookup table are supposed to be sorted such that
        the second column (the y axis) is sorted in ascending order
    
    Parameters
    ----------
    data: 1D array of correlation to be converted to displacements
    _lut: lookup table for inversion
    overflow_to_nan : if true, set all elements outside the LUT range to nan
    """
    res = np.ones_like(data)*np.nan
    xmin, xmax = np.min(_lut[0]), np.max(_lut[0])
    for i in range(len(data)):
        if not np.isnan(data[i]):
            if not (overflow_to_nan and (data[i]<xmin or data[i]>xmax)):
                index = bisect.bisect_left(_lut[1], data[i])
                res[i] = _lut[0][min(index, len(_lut[0])-1)]
    return res

def _get_kw_from_config(conf=None):
    # Read options for velocity calculation from DSH.Config object
    def_kw = {'q_value':1.0,\
               't_range':None,\
               'lag_range':None,\
               'signed_lags':False,\
               'consec_only':True,\
               'max_holes':0,\
               'mask_opening':None,\
               'conservative_cutoff':0.3,\
               'generous_cutoff':0.15}
    if (conf is None):
        return def_kw
    else:
        return {'q_value':conf.Get('velmap_parameters', 'q_value', def_kw['q_value'], float),\
               't_range':conf.Get('velmap_parameters', 't_range', def_kw['t_range'], int),\
               'lag_range':conf.Get('velmap_parameters', 'lag_range', def_kw['lag_range'], int),\
               'signed_lags':conf.Get('velmap_parameters', 'signed_lags', def_kw['signed_lags'], bool),\
               'consec_only':conf.Get('velmap_parameters', 'consec_only', def_kw['consec_only'], bool),\
               'max_holes':conf.Get('velmap_parameters', 'max_holes', def_kw['max_holes'], int),\
               'mask_opening':conf.Get('velmap_parameters', 'mask_opening', def_kw['mask_opening'], int),\
               'conservative_cutoff':conf.Get('velmap_parameters', 'conservative_cutoff', def_kw['conservative_cutoff'], float),\
               'generous_cutoff':conf.Get('velmap_parameters', 'generous_cutoff', def_kw['generous_cutoff'], float)}

def LoadFromConfig(ConfigFile, outFolder=None):
    """Loads a VelMaps object from a config file like the one exported with VelMaps.ExportConfig()
    
    Parameters
    ----------
    ConfigFile : full path of the config file to read
    outFolder : folder containing velocity and correlation maps. 
                if None, the value from the config file will be used
                if not None, the value from the config file will be discarded
                
    Returns
    -------
    a VelMaps object with an "empty" image MIfile (containing metadata but no actual image data)
    """
    config = cf.Config(ConfigFile)
    vmap_kw = _get_kw_from_config(config)
    return VelMaps(CM.LoadFromConfig(ConfigFile, outFolder), **vmap_kw)

class VelMaps():
    """ Class to compute velocity maps from correlation maps """
    
    def __init__(self, corr_maps, z_profile='Parabolic', q_value=1.0, t_range=None, lag_range=None, signed_lags=False, consec_only=True,\
                          max_holes=0, mask_opening=None, conservative_cutoff=0.3, generous_cutoff=0.15):
        """Initialize VelMaps
        
        Parameters
        ----------
        corr_maps : CorrMaps object with information on available correlation maps, metadata and lag times
        z_profile :  'Parabolic'|'Linear'. For the moment, only parabolic has been developed
        q_value :    Scattering vector projected along the sample plane. Used to express velocities in meaningful dimensions
                    NOTE by Stefano : 4.25 1/um
        t_range :    restrict analysis to given time range [min, max, step].
                    if None, analyze full correlation maps
        lag_range :  restrict analysis to correlation maps with lagtimes in a given range (in image units)
                    if None, all available correlation maps will be used
                    if int, lagtimes in [-lagRange, lagRange] will be used
        signed_lags : if True, lagtimes will have a positive or negative sign according to whether
                    the current time is the first one or the second one correlated
                    In this case, displacements will also be assigned the same sign as the lag
                    otherwise, we will work with absolute values only
                    Working with absolute values will "flatten" the linear fits, reducing slopes and increasing intercepts
                    It does a much better job accounting for noise contributions to corr(tau->0)
                    if signed_lags==False, the artificial correlation value at lag==0 will not be processed
                    (it is highly recommended to set signed_lags to False)
        consec_only : if True only select sorrelation chunk with consecutive True value of the mask around tau=0
                    Note: it is highly recommended to use consec_only==True
        max_holes : integer, only used if consecutive_only==True.
                    Largest hole to be ignored before chunk is considered as discontinued
        mask_opening : None or integer > 1.
                    if not None, apply binary_opening to the mask for a given pixel as a function of lagtime
                    This removes thresholding noise by removing N-lag-wide unmasked domains where N=mask_opening_range
        conservative_cutoff : only consider correlation data above this threshold value
        generous_cutoff : when correlations above conservative_cutoff are not enough for linear fitting,
                    include first lagtimes provided correlation data is above this more generous threshold
                    Note: for parabolic profiles, the first correlation minimum is 0.083, 
                    the first maximum after that is 0.132. Don't go below that!
        """
        
        self.corr_maps = corr_maps
        self.outFolder = corr_maps.outFolder
        
        self.zProfile = z_profile
        self.qValue = q_value
        self.tRange = t_range
        self.lagRange = lag_range
        self.signedLags = signed_lags
        self.consecOnly = consec_only
        self.maxHoles = max_holes
        self.maskOpening = mask_opening
        self.conservative_cutoff = conservative_cutoff
        self.generous_cutoff = generous_cutoff
        
        if (lag_range is not None):
            if (isinstance(lag_range, int) or isinstance(lag_range, float)):
                self.lagRange = [-lag_range, lag_range]
            
        self._loaded_metadata = False
        self._velmaps_loaded = False

    def ExportConfig(self, FileName, tRange=None):
        # Export configuration
        if (tRange is None):
            tRange = self.tRange
        vmap_options = {
                        'z_profile' : self.zProfile,
                        'q_value' : self.qValue,
                        'signed_lags' : self.signedLags,
                        'consec_only' : self.consecOnly,
                        'max_holes' : self.maxHoles,
                        'conservative_cutoff' : self.conservative_cutoff,
                        'generous_cutoff' : self.generous_cutoff,
                        }
        if (self.tRange is not None):
            vmap_options['t_range'] = list(self.tRange)
        if (self.lagRange is not None):
            vmap_options['lag_range'] = list(self.lagRange)
        if (self.maskOpening is not None):
            vmap_options['mask_opening'] = self.maskOpening
        self.confParams.Import(vmap_options, section_name='velmap_parameters')
        self.confParams.Import(self.mapMetaData.ToDict(section='MIfile'), section_name='velmap_metadata')
        self.confParams.Export(FileName)

    def GetLagtimes(self):
        if not self._loaded_metadata:
            self._load_metadata_from_corr()
        return self.lagTimes.copy()
    def GetShape(self):
        return self.MapShape
    def ImageShape(self):
        return [self.MapShape[1], self.MapShape[2]]
    def ImageWidth(self):
        return self.MapShape[2]
    def ImageHeight(self):
        return self.MapShape[1]
    def ImageNumber(self):
        return self.MapShape[0]
    def GetMetadata(self):
        return self.mapMetaData.ToDict(section='MIfile').copy()
    def GetFPS(self):
        return self.mapMetaData.Get('MIfile', 'fps', 1.0, float)
    def GetPixelSize(self):
        return self.mapMetaData.Get('MIfile', 'px_size', 1.0, float)
    
    def GetMaps(self):
        """Searches for MIfile velocity maps
        
        Returns
        -------
        vmap_config: configuration file for velocity maps
        vmap_mifiles: velocity maps mifile
        """    
                
        if not self._velmaps_loaded:

            assert os.path.isdir(self.outFolder), 'Velocity map folder ' + str(self.outFolder) + ' not found.'
            config_fname = os.path.join(self.outFolder, 'VelMapsConfig.ini')
            assert os.path.isfile(config_fname), 'Configuration file ' + str(config_fname) + ' not found'
            self.conf_vmaps = cf.Config(config_fname)
            vmap_fname = os.path.join(self.outFolder, '_vMap.dat')
            assert os.path.isfile(vmap_fname), 'Velocity map file ' + str(config_fname) + ' not found'
            self.vmap_mifile = MI.MIfile(vmap_fname, self.conf_vmaps.ToDict(section='velmap_metadata'))
            self._velmaps_loaded = True
        
        return self.conf_vmaps, self.vmap_mifile

    def ComputeMultiproc(self, numProcesses, assemble_after=True):
        """Computes correlation maps in a multiprocess fashion
        
        Parameters
        ----------
        numProcesses : number of processes to split the computation
                        every process will be given a fraction of times
        assemble_after : if true, after the process assemble output in
                        one file with full correlations
        
        Returns
        -------
        assembled 3D velocity map, if assemble_after==True
        """
        
        if not self._loaded_metadata:
            self._load_metadata_from_corr()
            
        cur_trange = MI.Validate_zRange(self.tRange, self.corr_maps.outputShape[0], replaceNone=True)
        start_t = cur_trange[0]
        end_t = cur_trange[1]
        num_t = (end_t-start_t) // numProcesses
        step_t = cur_trange[2]
        all_tranges = []
        for pid in range(numProcesses):
            all_tranges.append([start_t, start_t+num_t, step_t])
            start_t = start_t + num_t
        all_tranges[-1][1] = end_t
        proc_list = []
        for pid in range(numProcesses):
            cur_p = Process(target=self.Compute, args=(all_tranges[pid], '_'+str(pid).zfill(2), True, False, False))
            cur_p.start()
            proc_list.append(cur_p)
        for cur_p in proc_list:
            cur_p.join()
        
        if assemble_after:
            return self.AssembleMultiproc()

    def AssembleMultiproc(self, outFileName, out_folder=None):
        """Assembles output from multiprocess calculation in one final output file
        
        Parameters
        ----------
        out_folder : folder where to search for partial velocity maps to assemble and where to save final output
                     if None, self.outFolder will be used
        """
        if (out_folder is None):
            out_folder = self.outFolder
        partial_vmap_fnames = sf.FindFileNames(out_folder, Prefix='_vMap_', Ext='.dat', Sort='ASC', AppendFolder=True)
        vmap_mi_files = []
        for fidx in range(len(partial_vmap_fnames)):
            pid = sf.LastIntInStr(partial_vmap_fnames[fidx])
            partial_vmap_config_fname = os.path.join(out_folder, 'VelMapsConfig_' + str(pid).zfill(2) + '.ini')
            if not os.path.isfile(partial_vmap_config_fname):
                raise IOError('vMap metadata file ' + str(partial_vmap_config_fname) + ' not found.')
            vmap_config = cf.Config(partial_vmap_config_fname)
            vmap_mi_files.append(MI.MIfile(partial_vmap_fnames[fidx], vmap_config.ToDict(section='velmap_metadata')))
        combined_corrmap = MI.MergeMIfiles(outFileName, vmap_mi_files, os.path.join(out_folder, '_vMap_metadata.ini'))
        return combined_corrmap

    def Compute(self, tRange=None, file_suffix='', silent=True, return_err=False):
        """Computes velocity maps
        
        Parameters
        ----------
        tRange : time range. If None, self.tRange will be used. Use None for single process computation
                Set it to subset of total images for multiprocess computation
        file_suffix : suffix to be appended to filename to differentiate output from multiple processes
        silent : bool. If set to False, procedure will print to output every time steps it goes through. 
                otherwise, it will run silently
        return_err : bool. if True, return mean squred error on the fit
                
        Returns
        -------
        res_3D : 3D velocity map
        """

        if not silent:
            start_time = time.time()
            print('Computing velocity maps:')
            cur_progperc = 0
            prog_update = 10
        
        if not self._loaded_metadata:
            self._load_metadata_from_corr()
        
        _MapShape = self.MapShape
        if (tRange is None):
            tRange = self.tRange
        else:
            tRange = self.cmap_mifiles[1].Validate_zRange(tRange)
            if (self.mapMetaData.HasOption('MIfile', 'fps')):
                self.mapMetaData.Set('MIfile', 'fps', str(self.GetFPS() * 1.0/self.tRange[2]))
        if (tRange is None):
            corrFrameIdx_list = list(range(self.MapShape[0]))
        else:
            corrFrameIdx_list = list(range(*tRange))
        _MapShape[0] = len(corrFrameIdx_list)
        
        self.ExportConfig(os.path.join(self.outFolder, 'VelMapsConfig' + str(file_suffix) + '.ini'))

        # Prepare memory
        qdr_g = g2m1_sample(zProfile=self.zProfile)
        vmap = np.zeros(_MapShape)
        write_vmap = MI.MIfile(os.path.join(self.outFolder, '_vMap' + str(file_suffix) + '.dat'), self.GetMetadata())
        if return_err:
            verr = np.zeros(_MapShape)
            write_verr = MI.MIfile(os.path.join(self.outFolder, '_vErr' + str(file_suffix) + '.dat'), self.GetMetadata())
            
        for tidx in range(_MapShape[0]):
            
            #corrframe_idx = corrFrameIdx_list[tidx]
            
            # find compatible lag indexes
            lag_idxs = []  # index of lag in all_lagtimes list
            t1_idxs = []   # tidx if tidx is t1, tidx-lag if tidx is t2
            sign_list = [] # +1 if tidx is t1, -1 if tidx is t2
            # From largest to smallest, 0 excluded
            for lidx in range(len(self.lagTimes)-1, 0, -1):
                if (self.lagTimes[lidx] <= corrFrameIdx_list[tidx]):
                    bln_add = True
                    if (self.lagRange is not None):
                        bln_add = (-1.0*self.lagTimes[lidx] >= self.lagRange[0])
                    if bln_add:
                        t1_idxs.append(corrFrameIdx_list[tidx]-self.lagTimes[lidx])
                        lag_idxs.append(lidx)
                        sign_list.append(-1)
            # From smallest to largest, 0 included
            for lidx in range(len(self.lagTimes)):
                bln_add = True
                if (self.lagRange is not None):
                    bln_add = (self.lagTimes[lidx] <= self.lagRange[1])
                if bln_add:
                    t1_idxs.append(corrFrameIdx_list[tidx])
                    lag_idxs.append(lidx)
                    sign_list.append(1)
            
            # Populate arrays
            cur_cmaps = np.ones([len(lag_idxs), self.ImageHeight(), self.ImageWidth()])
            cur_lags = np.zeros_like(cur_cmaps)
            cur_signs = np.ones_like(cur_cmaps, dtype=np.int8)
            zero_lidx = -1
            for lidx in range(len(lag_idxs)):
                if (lag_idxs[lidx] > 0):
                    cur_cmaps[lidx] = self.cmap_mifiles[lag_idxs[lidx]].GetImage(t1_idxs[lidx])
                    cur_lags[lidx] = np.ones([self.ImageHeight(), self.ImageWidth()])*self.lagTimes[lag_idxs[lidx]]*1.0/self.GetFPS()
                    cur_signs[lidx] = np.multiply(cur_signs[lidx], sign_list[lidx])
                else:
                    # if lag_idxs[lidx]==0, keep correlations equal to ones and lags equal to zero
                    # just memorize what this index is
                    zero_lidx = lidx
            cur_mask = cur_cmaps > self.conservative_cutoff
            
            
            for ridx in range(self.ImageHeight()):
                for cidx in range(self.ImageWidth()):
                    cur_try_mask = cur_mask[:,ridx,cidx]
                    if (self.maskOpening is not None and np.count_nonzero(cur_try_mask) > 2):
                        for cur_open_range in range(self.maskOpening, 2, -1):
                            # remove thresholding noise by removing N-lag-wide unmasked domains
                            cur_mask_denoise = binary_opening(cur_try_mask, structure=np.ones(cur_open_range))
                            if (np.count_nonzero(cur_try_mask) > 2):
                                cur_use_mask = cur_mask_denoise
                                break
                    if self.consecOnly:
                        cur_use_mask = np.zeros(len(lag_idxs), dtype=bool)
                        cur_hole = 0
                        for ilag_pos in range(zero_lidx+1, len(lag_idxs)):
                            if cur_try_mask[ilag_pos]:
                                cur_use_mask[ilag_pos] = True
                                cur_hole = 0
                            else:
                                cur_hole = cur_hole + 1
                            if (cur_hole > self.maxHoles):
                                break
                        cur_hole = 0
                        for ilag_neg in range(zero_lidx, -1, -1):
                            if cur_try_mask[ilag_neg]:
                                cur_use_mask[ilag_neg] = True
                                cur_hole = 0
                            else:
                                cur_hole = cur_hole + 1
                            if (cur_hole > self.maxHoles):
                                break
                    else:
                        cur_use_mask = cur_try_mask

                    # Only use zero lag correlation when dealing with signed lagtimes
                    cur_use_mask[zero_lidx] = self.signedLags
                        
                    num_nonmasked = np.count_nonzero(cur_use_mask)
                    if (num_nonmasked <= 1):
                        cur_use_mask[zero_lidx] = True
                        # If there are not enough useful correlation values, 
                        # check if the first available lagtimes can be used at least with a generous cutoff
                        # If they are, use them, otherwise just set that cell to nan
                        if (zero_lidx+1 < len(lag_idxs)):
                            if (cur_cmaps[zero_lidx+1,ridx,cidx] > self.generous_cutoff):
                                cur_use_mask[zero_lidx+1] = True
                        if (zero_lidx > 0):
                            if (cur_cmaps[zero_lidx-1,ridx,cidx] > self.generous_cutoff):
                                cur_use_mask[zero_lidx-1] = True
                        num_nonmasked = np.count_nonzero(cur_use_mask)
                        
                    if (num_nonmasked > 1):
                        cur_data = cur_cmaps[:,ridx,cidx][cur_use_mask]
                        if self.signedLags:
                            cur_signs_1d = cur_signs[:,ridx,cidx][cur_use_mask]
                            cur_dt = np.multiply(cur_lags[:,ridx,cidx][cur_use_mask], cur_signs_1d)
                            cur_dr = np.multiply(np.true_divide(invert_monotonic(cur_data, qdr_g), self.qValue), cur_signs_1d)
                            slope, intercept, r_value, p_value, std_err = stats.linregress(cur_dt, cur_dr)
                        else:
                            cur_dt = cur_lags[:,ridx,cidx][cur_use_mask]
                            cur_dr = np.true_divide(invert_monotonic(cur_data, qdr_g), self.qValue)
                            # Here there is the possibility to have only 2 datapoints with the same dt. We need to address that case
                            if (num_nonmasked == 2):
                                if (np.max(cur_dt)==np.min(cur_dt)):
                                    slope = np.mean(cur_dr) * 1.0 / cur_dt[0]
                                    intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan
                                else:
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(cur_dt, cur_dr)
                            else:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(cur_dt, cur_dr)
                    else:
                        slope, std_err = np.nan, np.nan
                        
                    vmap[tidx,ridx,cidx] = slope
                    if return_err:
                        verr[tidx,ridx,cidx] = std_err

            write_vmap.WriteData(vmap[tidx], closeAfter=False)
            if return_err:
                write_verr.WriteData(verr[tidx], closeAfter=False)

            if not silent:
                cur_p = (tidx+1)*100/_MapShape[0]
                if (cur_p > cur_progperc+prog_update):
                    cur_progperc = cur_progperc+prog_update
                    print('   {0}% completed...'.format(cur_progperc))
        
        if not silent:
            print('Procedure completed in {0:.1f} seconds!'.format(time.time()-start_time))
            
        if return_err:
            return vmap, verr
        else:
            return vmap

    def ProcessSinglePixel(self, pxLoc, tRange=None, debugPrint=False, debugFile=None):
        """Computes v(t) for one single pixel
        
        Parameters
        ----------
        pxLoc: [row_index, col_index] coordinates of pixel to be analyzed
        
        Returns
        -------
        vel : 1D array with velocity as a function of time
        verr : 1D array with linear fit errors
        """

        if not self._loaded_metadata:
            self._load_metadata_from_corr()

        if tRange is None:
            tRange = self.tRange

        # Load correlation data. Set central row (d0) to ones and set zero correlations to NaN
        corr_data, tvalues, lagList, lagFlip = self.corr_maps.GetCorrTimetrace(pxLoc, zRange=tRange, lagFlip='BOTH',\
                                                                               returnCoords=True, squeezeResult=False)
        lagList = np.asarray(lagList)
        zero_lidx = int((corr_data.shape[1]-1)/2)
        corr_data[:,zero_lidx,:] = np.ones_like(corr_data[:,0,:])
        corr_data[np.where(corr_data==0)]=np.nan
        try_mask = corr_data > self.conservative_cutoff
        if self.signedLags:
            lagsign = np.ones_like(lagList, dtype=np.int8)
            lagsign[lagFlip]=-1

        # Prepare memory
        qdr_g = g2m1_sample(zProfile=self.zProfile)
        vel = np.zeros(corr_data.shape[2])
        interc = np.zeros_like(vel)
        fiterr = np.zeros_like(vel)
        
        if debugFile is not None:
            fdeb = open(os.path.join(self.outFolder, debugFile), 'w')
            
        for tidx in range(len(vel)):
            
            # Fine tune selection of lags to include
            use_mask = self._tunemask_pixel(try_mask[0,:,tidx], zero_lidx, corr_data[0,:,tidx])
            if debugPrint:
                print('   ### ' + str(use_mask))

            # Perform linear fit
            cur_dt = lagList[use_mask]
            cur_dr = np.true_divide(invert_monotonic(corr_data[0,:,tidx][use_mask], qdr_g), self.qValue)
            if self.signedLags:
                cur_dt = np.multiply(cur_dt, lagsign[use_mask])
                cur_dr = np.multiply(cur_dr, lagsign[use_mask])
            if (np.max(cur_dt)==np.min(cur_dt)):
                slope = np.nanmean(cur_dr)*1.0/cur_dt[0]
                intercept, r_value, p_value, std_err = 0, np.nan, np.nan, np.nan
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(cur_dt, cur_dr)  
            
            # Save result
            vel[tidx] = slope
            interc[tidx] = intercept
            fiterr[tidx] = std_err
            
            if debugPrint:
                print('   *** ' + str(tidx) + ' - ' + str(np.count_nonzero(use_mask)) + ' points, dt=[' + str(np.min(cur_dt)) + ',' + str(np.max(cur_dt)) + ']' +\
                      ' - dr=[' + str(np.min(cur_dr)) + ',' + str(np.max(cur_dr)) + '] - fit result: ' + str([slope, intercept, r_value, p_value, std_err]))
            if debugFile is not None:
                strWrite = '\n*******************'
                strWrite += '\nt=' + str(tidx)
                strWrite += '\n-------------------'
                strWrite += '\norig_lag\torig_corr\tsign\tmask'
                for i in range(corr_data.shape[1]):
                    strWrite += '\n' + str(lagList[i]) + '\t' + str(corr_data[0,i,tidx]) + '\t' + str(lagsign[i]) + '\t' + str(use_mask[i])
                strWrite += '\n-------------------'
                strWrite += '\nfit_dt\tfit_dr'
                for i in range(len(cur_dt)):
                    strWrite += '\n' + str(cur_dt[i]) + '\t' + str(cur_dr[i])
                strWrite += '\n-------------------'
                strWrite += '\nFit results:'
                strWrite += '\n  slope=' + str(slope)
                strWrite += '\n  intercept=' + str(intercept)
                strWrite += '\n  r_value=' + str(r_value)
                strWrite += '\n  p_value=' + str(p_value)
                strWrite += '\n  std_err=' + str(std_err)
                strWrite += '\n*******************'                
                fdeb.write(strWrite)

        if debugFile is not None:
            fdeb.close()

        return vel, interc, fiterr
    
    def ProcessSinglePixelOLD(self, pxLoc, tRange=None, debugPrint=False, debugFile=None):
        """Computes v(t) for one single pixel
        
        Returns
        -------
        vel : 1D array with velocity as a function of time
        verr : 1D array with linear fit errors
        """

        if not self._loaded_metadata:
            self._load_metadata_from_corr()

        if tRange is None:
            tRange = self.tRange

        # Load correlation data. Set first row (d0) to ones and set zero correlations to NaN
        corr_data = self.corr_maps.GetCorrTimetrace(pxLoc, zRange=tRange)
        corr_data[0] = np.ones_like(corr_data[0])
        corr_data[np.where(corr_data==0)]=np.nan

        # Find list of compatible lagtimes
        all_lag_idxs, all_sign_list = self._find_compatible_lags(tRange)

        # Prepare memory
        qdr_g = g2m1_sample(zProfile=self.zProfile)
        vel = np.zeros(len(all_lag_idxs))
        interc = np.zeros_like(vel)
        fiterr = np.zeros_like(vel)
        
        if debugFile is not None:
            fdeb = open(os.path.join(self.outFolder, debugFile), 'w')
            
        for tidx in range(len(vel)):
            
            # Populate arrays
            lag_idxs = all_lag_idxs[tidx]
            cur_corr = np.ones(len(lag_idxs))
            cur_lags = np.zeros_like(cur_corr)
            zero_lidx = -1
            for lidx in range(len(lag_idxs)):
                if (lag_idxs[lidx] > 0):
                    if (all_sign_list[tidx][lidx] > 0):
                        cur_corr[lidx] = corr_data[lag_idxs[lidx],tidx]
                    else:
                        if tidx >= self.lagTimes[lag_idxs[lidx]]:
                            cur_corr[lidx] = corr_data[lag_idxs[lidx],tidx-self.lagTimes[lag_idxs[lidx]]]
                    cur_lags[lidx] = self.lagTimes[lag_idxs[lidx]]*1.0/self.GetFPS()
                else:
                    # if lag_idxs[lidx]==0, keep correlations equal to ones and lags equal to zero
                    # just memorize what this index is
                    zero_lidx = lidx

            # Fine tune selection of lags to include
            try_mask = cur_corr > self.conservative_cutoff
            use_mask = self._tunemask_pixel(try_mask, zero_lidx, cur_corr)

            # Perform linear fit
            cur_dt = cur_lags[use_mask]
            cur_dr = np.true_divide(invert_monotonic(cur_corr[use_mask], qdr_g), self.qValue)
            if self.signedLags:
                cur_dt = np.multiply(cur_dt, all_sign_list[tidx][use_mask])
                cur_dr = np.multiply(cur_dr, all_sign_list[tidx][use_mask])
            if (np.max(cur_dt)==np.min(cur_dt)):
                slope = np.nanmean(cur_dr)*1.0/cur_dt[0]
                intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(cur_dt, cur_dr)  
            
            # Save result
            vel[tidx] = slope
            interc[tidx] = intercept
            fiterr[tidx] = std_err
            
            if debugPrint:
                print('   *** ' + str(tidx) + ' - ' + str(np.count_nonzero(use_mask)) + ' points, dt=[' + str(np.min(cur_dt)) + ',' + str(np.max(cur_dt)) + ']' +\
                      ' - dr=[' + str(np.min(cur_dr)) + ',' + str(np.max(cur_dr)) + '] - fit result: ' + str([slope, intercept, r_value, p_value, std_err]))
            if debugFile is not None:
                strWrite = '\n*******************'
                strWrite += '\nt=' + str(tidx)
                strWrite += '\n-------------------'
                strWrite += '\norig_lag\torig_corr\tsign\tmask'
                for i in range(len(cur_corr)):
                    strWrite += '\n' + str(self.lagTimes[lag_idxs[i]]) + '\t' + str(cur_corr[i]) + '\t' + str(all_sign_list[tidx][i]) + '\t' + str(use_mask[i])
                strWrite += '\n-------------------'
                strWrite += '\nfit_dt\tfit_dr'
                for i in range(len(cur_dt)):
                    strWrite += '\n' + str(cur_dt[i]) + '\t' + str(cur_dr[i])
                strWrite += '\n-------------------'
                strWrite += '\nFit results:'
                strWrite += '\n  slope=' + str(slope)
                strWrite += '\n  intercept=' + str(intercept)
                strWrite += '\n  r_value=' + str(r_value)
                strWrite += '\n  p_value=' + str(p_value)
                strWrite += '\n  std_err=' + str(std_err)
                strWrite += '\n*******************'                
                fdeb.write(strWrite)

        if debugFile is not None:
            fdeb.close()

        return vel, interc, fiterr


    def GetMIfile(self):
        """Returns velocity map as MIfile, if found in folder
        """
        assert (os.path.isdir(self.outFolder)), 'Correlation map folder ' + str(self.outFolder) + ' not found.'
        config_fname = os.path.join(self.outFolder, '_vMap_metadata.ini')
        MI_fname = os.path.join(self.outFolder, '_vMap.dat')
        assert os.path.isfile(config_fname), 'Configuration file ' + str(config_fname) + ' not found'
        assert os.path.isfile(MI_fname), 'MI file ' + str(MI_fname) + ' not found'
        return MI.MIfile(MI_fname, config_fname)
    
    
    def RefineMC(self, cropROI=None, tRange=None, lagTimes=None, corrPrior=None,\
                 initGaussBall=1e-3, priorStd=None, corrStdfunc=corr_std_calc, corrStdfuncParams={},\
                 regParam=0.0, nwalkers=None, nsteps=1000, burnin=0, qErr=[0.16, 0.84], detailed_output=True, file_suffix=''):
        """Refine velocity map using generative modeling and MCMC sampling
        
        Parameters
        ----------
        cropROI :   region to be refined [topleftx (0-based), toplefty (0-based), width, height]
                    pixels will be refined one at the time, and bigger ROI sizes will requre
                    more time but not more memory
        tRange :    time range to be refined [idx_min, idx_max, 1].
                    Currently the timestep cannot be changed, it will mess up with the likelihood calculation
        lagTimes :  list of timelags to be used for refinement.
                    if None, all available timelags shorter than tRange will be used
        corrPrior : Prior on correlation function parameters: [[d0_avg, base_avg], [d0_std, base_std]]
                    if None, the default value [[1.0, 0.0], [0.01, 0.01]] will be used
        initGaussBall : start MC walkers from random positions centered on the unrefined velocities and with
                    a relative spread given by initGaussBall
        priorStd :  use Gaussian prior centered in the unrefined velocities and with this standard deviation
                    if None, will use a flat improper prior
                    this is equal to setting all elements of priorStd to NaN
        corrStdfunc : function calculating uncertainties on correlation data taking as parameters
                    the correlation themselves plus eventual additional parameters
        regParam :  float>=0. Regularization parameter penalizing velocity fluctuations.
                    Default is regParam=0.0, which means no regularization
        nwalkers :  number of MC walkers to run in parallel for MC sampling.
                    if None, it will be set to twice the degrees of freedom of the problem
        nsteps :    length of Markov chains to be generated by each walker
                    It has to be larger than burnin
        burnin :    burning length: number of initial MC steps that need to be discarded because the
                    walker was reminiscent of the initial conditions
        qErr :      [neg_err, pos_err]: quartiles of distributions defining standard errors to be saved.
                    Default is [0.16, 0.84], corresponding to +/- 1sigma for Gaussian distributions
                    if None, no error will be output
        detailed_output : if True, the procedure will save as well:
                    - d0 and baseline for every pixel
                    - best log_likelihood, log_prior and log_posterior for every pixel
                    - mean squared error for every pixel
        file_suffix : suffix of the filename of the refined velocity map to be saved
        """
            
        cropROI = MI.ValidateROI(cropROI, self.ImageShape(), replaceNone=True)
        tRange = MI.Validate_zRange(tRange, self.ImageNumber(), replaceNone=True)
        assert tRange[2]==1, 'MC refinement needs to have tRange[2]==1. ' + str(tRange[2]) + ' given.'
        if lagTimes is None:
            lagTimes = self.GetLagtimes()
        lags = lagTimes.copy() # Make a copy to avoid removing elements from the original
        for cur_lag in lagTimes:
            if (cur_lag == 0 or cur_lag > (tRange[1]-tRange[0])):
                lags.remove(cur_lag)
        if corrPrior is None:
            corrPrior = [[1.0, 0.0], [0.01, 0.01]]
        vel_config, vel_mifile = self.GetMaps()
        vel_prior = vel_mifile.Read(zRange=tRange, cropROI=cropROI, closeAfter=True)
        if priorStd is None:
            vel_std = np.ones_like(vel_prior) * np.nan
        else:
            vel_std = priorStd * np.abs(vel_prior)
        
        res_vavg = np.empty_like(vel_prior)
        vmap_MetaData = {
            'hdr_len' : 0,
            'shape' : list(res_vavg.shape),
            'px_format' : 'f',
            'fps' : self.GetFPS(),
            'px_size' : self.GetPixelSize()
            }
        if qErr is not None:
            res_verr = np.empty_like(res_vavg)
        if detailed_output:
            res_mse = np.empty_like(res_vavg[0])
            res_prior = np.empty_like(res_mse)
            res_posterior = np.empty_like(res_mse)
            res_likelihood = np.empty_like(res_mse)
            res_d0 = np.empty_like(res_mse)
            res_base = np.empty_like(res_mse)
            if qErr is not None:
                res_d0_err = np.empty_like(res_mse)
                res_base_err = np.empty_like(res_mse)
        # the model has N+2 parameters: 
        # - parameter 0 will be d0, 
        # - parameter 1 will be the baseline,
        # - parameters [2:] will be the speeds
        ndim_corr = vel_prior.shape[0]+2
        
        # Exports analysis configuration
        conf_MCparams = {'crop_roi' : cropROI,
                         't_range' : tRange,
                         'lags' : lags,
                         'corr_prior' : corrPrior,
                         'init_gaussball' : initGaussBall,
                         'prior_std' : priorStd,
                         'corr_func' : str(corrStdfunc),
                         'regularization_param' : regParam,
                         'num_walkers' : nwalkers,
                         'num_steps' : nsteps,
                         'burnin_steps' : burnin,
                         'quantiles' : qErr,
                         'detailed_output' : detailed_output,
                         'file_suffix' : file_suffix,
                         }
        vel_config.Import(conf_MCparams, section_name='refinemc_parameters')
        vel_config.Export(os.path.join(self.outFolder, 'VelMapsConfig_RefMC' + str(file_suffix) + '.ini'))
        
        for irow in range(cropROI[1], cropROI[1]+cropROI[3]):
            for icol in range(cropROI[0], cropROI[0]+cropROI[2]):
                corrTimetrace = self.corr_maps.GetCorrTimetrace([icol, irow], lagList=lags, lagFlip=False, returnCoords=False, squeezeResult=True)
                stdTimetrace = corrStdfunc(corrTimetrace, **corrStdfuncParams)
                cur_prior = vel_prior[:,irow,icol]
                    
                prior_avg_params = np.asarray(corrPrior[0] + cur_prior.tolist())
                prior_std_params = np.asarray(corrPrior[1] + vel_std[:,irow,icol].tolist())
                starting_positions = (1 + initGaussBall * np.random.randn(nwalkers, ndim_corr)) * prior_avg_params
                
                # set up the sampler object
                sampler_corr = emcee.EnsembleSampler(nwalkers, ndim_corr, log_posterior_corr,\
                                                args=(g2m1_parab, corrTimetrace, stdTimetrace,\
                                                      prior_avg_params, prior_std_params,\
                                                      np.asarray(lags), self.qValue, regParam))
                # run the sampler
                sampler_corr.run_mcmc(starting_positions, nsteps)

                samples_corr = sampler_corr.chain[:,burnin:,:]
                # reshape the samples into a 2D array where the colums are individual time points
                traces = samples_corr.reshape(-1, ndim_corr).T
                # create a pandas DataFrame with labels.  This will come in handy 
                # in a moment, when we start using seaborn to plot our results 
                # (among other things, it saves us the trouble of typing in labels
                # for our plots)
                builder_corr_dict = {}
                builder_corr_dict['d0'] = traces[0]
                builder_corr_dict['base'] = traces[1]
                for i in range(ndim_corr-2):
                    # Take absolute value because a priori DSH only probes |v|
                    builder_corr_dict['v'+str(i)] = np.absolute(traces[i+2])
                parameter_samples_corr = pd.DataFrame(builder_corr_dict)
                
                # calculating the MAP and values can be done concisely using pandas
                if qErr is None:
                    best_params = parameter_samples_corr.quantile(0.50, axis=0).to_numpy()
                else:
                    best_params = np.empty_like(prior_avg_params)
                    q_corr = parameter_samples_corr.quantile([qErr[0],0.50,qErr[1]], axis=0)
                    best_params[0] = q_corr['d0'][0.50]
                    best_params[1] = q_corr['base'][0.50]
                    for i in range(ndim_corr-2):
                        best_params[i+2] = q_corr['v'+str(i)][0.50]
                
                # Populate output arrays
                for i in range(ndim_corr-2):
                    res_vavg[i,irow,icol] = best_params[i+2]
                    if qErr is not None:
                        res_verr[i,irow,icol] = 0.5*(q_corr['v'+str(i)][qErr[1]]-q_corr['v'+str(i)][qErr[0]])
                if detailed_output:
                    res_mse[irow,icol] = np.nanmean(np.square(np.subtract(g2m1_parab(compute_displ_multilag(best_params[2:], np.asarray(lags)),\
                                                                                   self.qValue, best_params[0], best_params[1]),\
                                                                          corrTimetrace)))
                    res_prior[irow,icol] = log_prior_corr(best_params, prior_avg_params, prior_std_params, regParam)
                    res_likelihood[irow,icol] = log_likelihood_corr(best_params, g2m1_parab, corrTimetrace,\
                                                                  stdTimetrace, np.asarray(lags), self.qValue)
                    res_posterior[irow,icol] = log_posterior_corr(best_params, g2m1_parab, corrTimetrace, stdTimetrace, prior_avg_params,\
                                                                 prior_std_params, np.asarray(lags), self.qValue, regParam)
                    res_d0[irow,icol] = best_params[0]
                    res_base[irow,icol] = best_params[1]
                    if qErr is not None:
                        res_d0_err[irow,icol] = 0.5*(q_corr['d0'][qErr[1]]-q_corr['d0'][qErr[0]])
                        res_base_err[irow,icol] = 0.5*(q_corr['base'][qErr[1]]-q_corr['base'][qErr[0]])
                        
        # Save output files
        MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC' + str(file_suffix) + '.dat'), vmap_MetaData).WriteData(res_vavg)
        if qErr is not None:
            MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_err' + str(file_suffix) + '.dat'), vmap_MetaData).WriteData(res_verr)
        if detailed_output:
            singleimg_MetaData = {
                'hdr_len' : 0,
                'shape' : [1, res_vavg.shape[1], res_vavg.shape[2]],
                'px_format' : 'f',
                'px_size' : self.GetPixelSize()
                }
            MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_MSE' + str(file_suffix) + '.dat'), singleimg_MetaData).WriteData(res_mse)
            MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_prior' + str(file_suffix) + '.dat'), singleimg_MetaData).WriteData(res_prior)
            MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_likelihood' + str(file_suffix) + '.dat'), singleimg_MetaData).WriteData(res_likelihood)
            MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_post' + str(file_suffix) + '.dat'), singleimg_MetaData).WriteData(res_posterior)
            MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_d0' + str(file_suffix) + '.dat'), singleimg_MetaData).WriteData(res_d0)
            MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_base' + str(file_suffix) + '.dat'), singleimg_MetaData).WriteData(res_base)
            if qErr is not None:
                MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_d0err' + str(file_suffix) + '.dat'), singleimg_MetaData).WriteData(res_d0_err)
                MI.MIfile(os.path.join(self.outFolder, '_vMap_refMC_baseerr' + str(file_suffix) + '.dat'), singleimg_MetaData).WriteData(res_base_err)
        
        return res_vavg
        
    def _load_metadata_from_corr(self):
        
        self.confParams, self.cmap_mifiles, self.lagTimes = self.corr_maps.GetCorrMaps(getAutocorr=False)

        # NOTE: first element of self.cmap_mifiles will be None instead of d0 (we don't need d0 to compute velocity maps)
        self.mapMetaData = cf.Config()
        if (len(self.cmap_mifiles) > 1):
            self.mapMetaData.Import(self.cmap_mifiles[1].GetMetadata().copy(), section_name='MIfile')
            self.MapShape = self.cmap_mifiles[1].GetShape()
            self.mapMetaData.Set('MIfile', 'shape', str(list(self.MapShape)))
            self.tRange = self.cmap_mifiles[1].Validate_zRange(self.tRange)
            if (self.mapMetaData.HasOption('MIfile', 'fps')):
                self.mapMetaData.Set('MIfile', 'fps', str(self.GetFPS() * 1.0/self.tRange[2]))
        else:
            print('WARNING: no correlation maps found in folder ' + str(self.corr_maps.outFolder))
            
        self._loaded_metadata = True

    def _find_compatible_lags(self, tRange=None):
        """Finds a list of all (t, tau) couples needed to compute v(t*) at all times t*
        
        Parameters
        ----------
        tRange: range of times t* for which v(t*) needs to be computed
        
        Returns
        -------
        lag_idxs:  2D list of integers. For every t*, list of all lag indexes for which correlations are available
                   lag time in image units will be self.lagTimes[lag_idxs[t*][i]]
        sign_list: 2D list of integers. This list keeps track of positive and negative lagtimes by saving +1 and -1 respectively
        """
        
        if (tRange is None):
            tRange = self.tRange
        else:
            tRange = self.cmap_mifiles[1].Validate_zRange(tRange)
        if (tRange is None):
            fridxs = list(range(self.MapShape[0]))
        else:
            fridxs = list(range(*tRange))

        # find compatible lag indexes
        lag_idxs = []  # index of lag in all_lagtimes list
        sign_list = [] # +1 if tidx is t1, -1 if tidx is t2
        
        for tidx in range(len(fridxs)):
            cur_lag_idxs = []
            cur_sign_list = []
            # From largest to smallest, 0 excluded
            for lidx in range(len(self.lagTimes)-1, 0, -1):
                if (self.lagTimes[lidx] <= fridxs[tidx]):
                    bln_add = True
                    if (self.lagRange is not None):
                        bln_add = (-1.0*self.lagTimes[lidx] >= self.lagRange[0])
                    if bln_add:
                        cur_lag_idxs.append(lidx)
                        cur_sign_list.append(-1)
            # From smallest to largest, 0 included
            for lidx in range(len(self.lagTimes)):
                bln_add = True
                if (self.lagRange is not None):
                    bln_add = (self.lagTimes[lidx] <= self.lagRange[1])
                if bln_add:
                    cur_lag_idxs.append(lidx)
                    cur_sign_list.append(1)
            lag_idxs.append(cur_lag_idxs)
            sign_list.append(cur_sign_list)

        return lag_idxs, sign_list
    
    
    def _tunemask_pixel(self, try_mask, lagzero_idx, corrdata=None):
        """Tunes mask for pixel-by-pixel linear fit
        
        Parameters
        ----------
        try_mask : 1D boolean array. First guess for the mask
        lagzero_idx : index of zero lag (separating positive from negative time delays)
        corrdata : 1D float array, optional. Array of correlation data
                    it is used if final mask is too short, to slightly relax the correlation threshold
                    
        Returns
        -------
        use_mask : 1D boolean array. Mask to be used for linear fit
        """
        
        temp_mask = np.asarray(try_mask, dtype=bool)
        if (self.maskOpening is not None and np.count_nonzero(try_mask) > 2):
            for cur_open_range in range(self.maskOpening, 2, -1):
                # remove thresholding noise by removing N-lag-wide unmasked domains
                cur_mask_denoise = binary_opening(try_mask, structure=np.ones(cur_open_range))
                if (np.count_nonzero(try_mask) > 2):
                    temp_mask = cur_mask_denoise
                    break
        if self.consecOnly:
            use_mask = np.zeros(len(temp_mask), dtype=bool)
            cur_hole = 0
            for ilag_pos in range(lagzero_idx+1, len(temp_mask)):
                if temp_mask[ilag_pos]:
                    use_mask[ilag_pos] = True
                    cur_hole = 0
                else:
                    cur_hole = cur_hole + 1
                if (cur_hole > self.maxHoles):
                    break
            cur_hole = 0
            for ilag_neg in range(lagzero_idx, -1, -1):
                if temp_mask[ilag_neg]:
                    use_mask[ilag_neg] = True
                    cur_hole = 0
                else:
                    cur_hole = cur_hole + 1
                if (cur_hole > self.maxHoles):
                    break
        else:
            use_mask = temp_mask

        # Only use zero lag correlation when dealing with signed lagtimes
        use_mask[lagzero_idx] = self.signedLags
            
        if (np.count_nonzero(use_mask) <= 1):
            use_mask[lagzero_idx] = True
            # Eventually use correlation data, if available
            if corrdata is not None:
                # If there are not enough useful correlation values, 
                # check if the first available lagtimes can be used at least with a generous cutoff
                # If they are, use them, otherwise just set that cell to nan
                if (lagzero_idx+1 < len(try_mask)):
                    if (corrdata[lagzero_idx+1] > self.generous_cutoff):
                        use_mask[lagzero_idx+1] = True
                if (lagzero_idx > 0):
                    if (corrdata[lagzero_idx-1] > self.generous_cutoff):
                        use_mask[lagzero_idx-1] = True
        
        return use_mask










    def CalcDisplacements(self, outFilename, outMetadataFile, silent=True):
        """Integrate velocities to compute total displacements since the beginning of the experiment
        """
        if (os.path.isdir(self.outFolder)):
            vmap_fname = os.path.join(self.outFolder, '_vMap.dat')
            config_fname = os.path.join(self.outFolder, '_vMap_metadata.ini')
            if (os.path.isfile(config_fname) and os.path.isfile(vmap_fname)):
                vmap_mifile = MI.MIfile(vmap_fname, config_fname)
            else:
                raise IOError('MIfile ' + str(vmap_fname) + ' or metadata file ' + str(config_fname) + ' not found')
        else:
            raise IOError('Correlation map folder ' + str(self.outFolder) + ' not found.')

        # Read all velocity map to memory
        vmap_data = vmap_mifile.Read()
        
        displmap_data = np.empty_like(vmap_data)
        dt = 1.0/vmap_data.GetFPS()
        displmap_data[0] = np.multiply(vmap_data[0], dt)
        for tidx in range(1, displmap_data.shape[0]):
            displmap_data[tidx] = np.add(displmap_data[tidx-1], np.multiply(vmap_data[0], dt))

        MI.MIfile(outFilename, vmap_mifile.GetMetadata()).WriteData(displmap_data)
        cf.ExportDict(vmap_mifile.GetMetadata(), outMetadataFile, section_name='MIfile')
        
        return displmap_data