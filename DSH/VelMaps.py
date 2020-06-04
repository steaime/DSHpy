import os
import numpy as np
import bisect
import scipy as sp
import scipy.special as ssp
from scipy import stats
from scipy.ndimage import binary_opening
import time
from multiprocessing import Process

from DSH import Config as cf
from DSH import MIfile as MI
from DSH import SharedFunctions as sf

def _qdr_g_relation(zProfile='Parabolic'):
    """Generate a lookup table for inverting correlation function
    """
    if (zProfile=='Parabolic'):
        dr_g = np.zeros([2,4500])
        dr_g[0] = np.linspace(4.5, 0.001, num=4500)
        norm = 4/np.pi
        dr_g[1] = (np.square(abs(ssp.erf(sp.sqrt(-dr_g[0]*1j))))/dr_g[0])/norm
        dr_g[1][4499] = 1 #to prevent correlation to be higher than highest value in array
        return dr_g
    else:
        raise ValueError(str(zProfile) + 'z profile not implemented yet')

def _invert_monotonic(data, _lut):
    """Invert monotonic function based on a lookup table
    """
    res = np.zeros(len(data))
    for i in range(len(data)):
        index = bisect.bisect_left(_lut[1], data[i])
        res[i] = _lut[0][min(index, len(_lut[0])-1)]
    return res

class VelMaps():
    """ Class to compute velocity maps from correlation maps """
    
    def __init__(self, corr_maps, zProfile='Parabolic', qValue=1.0, tRange=None, lagRange=None, signedLags=False, consecOnly=False,\
                          maxHoles=0, maskOpening=None, conservative_cutoff=0.3, generous_cutoff=0.15):
        """Initialize VelMaps
        
        Parameters
        ----------
        corr_maps : CorrMaps object with information on available correlation maps, metadata and lag times
        zProfile :  'Parabolic'|'Linear'. For the moment, only parabolic has been developed
        qValue :    Scattering vector projected along the sample plane. Used to express velocities in meaningful dimensions
                    NOTE by Stefano : 4.25 1/um
        tRange :    restrict analysis to given time range [min, max, step].
                    if None, analyze full correlation maps
        lagRange :  restrict analysis to correlation maps with lagtimes in a given range (in image units)
                    if None, all available correlation maps will be used
                    if int, lagtimes in [-lagRange, lagRange] will be used
        signedLags : if True, lagtimes will have a positive or negative sign according to whether
                    the current time is the first one or the second one correlated
                    In this case, displacements will also be assigned the same sign as the lag
                    otherwise, we will work with absolute values only
                    This will "flatten" the linear fits, reducing slopes and increasing intercepts
                    It does a much better job accounting for noise contributions to corr(tau->0)
                    if signed_lags==False, the artificial correlation value at lag==0 will not be processed
                    (it is highly recommended to set signed_lags to False)
        consecOnly : only select sorrelation chunk with consecutive True value of the mask around tau=0
        maxHoles : integer, only used if consecutive_only==True.
                    Largest hole to be ignored before chunk is considered as discontinued
        maskOpening : None or integer > 1.
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
        
        self.zProfile = zProfile
        self.qValue = qValue
        self.tRange = tRange
        self.lagRange = lagRange
        self.signedLags = signedLags
        self.consecOnly = consecOnly
        self.maxHoles = maxHoles
        self.maskOpening = maskOpening
        self.conservative_cutoff = conservative_cutoff
        self.generous_cutoff = generous_cutoff
        
        self.confParams, self.cmap_mifiles, self.lagTimes = corr_maps.GetCorrMaps()
        
        if (lagRange is not None):
            if (isinstance(lagRange, int) or isinstance(lagRange, float)):
                self.lagRange = [-lagRange, lagRange]

        # NOTE: first element of self.cmap_mifiles will be None instead of d0 (we don't need d0 to compute velocity maps)
        self.mapMetaData = cf.Config()
        if (len(self.cmap_mifiles) > 1):
            self.mapMetaData.Import(self.cmap_mifiles[1].GetMetadata().copy(), section_name='MIfile')
            self.MapShape = self.cmap_mifiles[1].GetShape()
            self.mapMetaData.Set('MIfile', 'shape', str(list(self.MapShape)))
            if (self.mapMetaData.HasOption('MIfile', 'fps')):
                self.mapMetaData.Set('MIfile', 'fps', str(self.mapMetaData.Get('MIfile', 'fps', 1.0, float) * 1.0/self.tRange[2]))
        else:
            print('WARNING: no correlation maps found in folder ' + str(corr_maps.outFolder))


    def ExportConfig(self, FileName, tRange=None):
        # Export configuration
        if (tRange is None):
            tRange = self.tRange
        vmap_options = {
                        'zProfile' : self.zProfile,
                        'qValue' : self.qValue,
                        'signedLags' : self.signedLags,
                        'consecOnly' : self.consecOnly,
                        'maxHoles' : self.maxHoles,
                        'conservative_cutoff' : self.conservative_cutoff,
                        'generous_cutoff' : self.generous_cutoff,
                        }
        if (self.tRange is not None):
            vmap_options['tRange'] = list(self.tRange)
        if (self.lagRange is not None):
            vmap_options['lagRange'] = list(self.lagRange)
        if (self.maskOpening is not None):
            vmap_options['maskOpening'] = self.maskOpening
        self.confParams.Import(vmap_options, section_name='velmap_parameters')
        self.confParams.Import(self.mapMetaData.ToDict(section='MIfile'), section_name='velmap_metadata')
        self.confParams.Export(FileName)

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
    def GetValidFrameRange(self):
        ret_range = self.tRange
        if (ret_range is None):
            ret_range = [0, -1]
        if (ret_range[1] < 0):
            ret_range[1] = self.corr_maps.outputShape[0]
        if (len(ret_range) < 3):
            ret_range.append(1)
        return ret_range
    
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
        start_t = self.GetValidFrameRange()[0]
        end_t = self.GetValidFrameRange()[1]
        num_t = (end_t-start_t) // numProcesses
        step_t = self.GetValidFrameRange()[2]
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

    def Compute(self, tRange=None, file_suffix='', silent=True, return_err=False, debug=False):
        """Computes correlation maps
        
        Parameters
        ----------
        tRange : time range. If None, self.tRange will be used. Use None for single process computation
                Set it to subset of total images for multiprocess computation
        file_suffix : suffix to be appended to filename to differentiate output from multiple processes
        silent : bool. If set to False, procedure will print to output every time steps it goes through. 
                otherwise, it will run silently
        return_err : bool. if True, return mean squred error on the fit
        debug : bool. If set to True, procedure will save more detailed output, including
                fit intercept, errors and number of fitted datapoints
                
        Returns
        -------
        res_3D : 3D velocity map
        """

        if (silent==False or debug==True):
            start_time = time.time()
            print('Computing velocity maps:')
            cur_progperc = 0
            prog_update = 10
        
        _MapShape = self.MapShape
        if (tRange is None):
            tRange = self.tRange
        else:
            tRange = self.cmap_mifiles[1].Validate_zRange(tRange)
            if (self.mapMetaData.HasOption('MIfile', 'fps')):
                self.mapMetaData.Set('MIfile', 'fps', str(self.mapMetaData.Get('MIfile', 'fps', 1.0, float) * 1.0/self.tRange[2]))
        if (tRange is None):
            corrFrameIdx_list = list(range(self.MapShape[0]))
        else:
            corrFrameIdx_list = list(range(*tRange))
        _MapShape[0] = len(corrFrameIdx_list)
        
        self.ExportConfig(os.path.join(self.outFolder, 'VelMapsConfig' + str(file_suffix) + '.ini'))

        # Prepare memory
        qdr_g = _qdr_g_relation(zProfile=self.zProfile)
        vmap = np.zeros(_MapShape)
        write_vmap = MI.MIfile(os.path.join(self.outFolder, '_vMap' + str(file_suffix) + '.dat'), self.GetMetadata())
        if return_err:
            verr = np.zeros(_MapShape)
            write_verr = MI.MIfile(os.path.join(self.outFolder, '_vErr' + str(file_suffix) + '.dat'), self.GetMetadata())
        if debug:
            write_interc = MI.MIfile(os.path.join(self.outFolder, '_interc' + str(file_suffix) + '.dat'), self.GetMetadata())
            write_pval = MI.MIfile(os.path.join(self.outFolder, '_pval' + str(file_suffix) + '.dat'), self.GetMetadata())
            write_nvals = MI.MIfile(os.path.join(self.outFolder, '_nvals' + str(file_suffix) + '.dat'), self.GetMetadata())
            
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
            cur_lags = np.zeros([len(lag_idxs), self.ImageHeight(), self.ImageWidth()])
            cur_signs = np.ones([len(lag_idxs), self.ImageHeight(), self.ImageWidth()], dtype=np.int8)
            zero_lidx = -1
            for lidx in range(len(lag_idxs)):
                if (lag_idxs[lidx] > 0):
                    cur_cmaps[lidx] = self.cmap_mifiles[lag_idxs[lidx]].GetImage(t1_idxs[lidx])
                    cur_lags[lidx] = np.ones([self.ImageHeight(), self.ImageWidth()])*self.lagTimes[lag_idxs[lidx]]*1.0/self.outMetaData['fps']
                    cur_signs[lidx] = np.multiply(cur_signs[lidx], sign_list[lidx])
                else:
                    # if lag_idxs[lidx]==0, keep correlations equal to ones and lags equal to zero
                    # just memorize what this index is
                    zero_lidx = lidx
            cur_mask = cur_cmaps > self.conservative_cutoff
            
            if debug:
                cur_nvals = np.empty(self.ImageShape())
                cur_interc = np.empty(self.ImageShape())
                cur_pval = np.empty(self.ImageShape())
            
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
                            cur_dr = np.multiply(np.true_divide(self._invert_monotonic(cur_data, qdr_g), self.qValue), cur_signs_1d)
                            slope, intercept, r_value, p_value, std_err = stats.linregress(cur_dt, cur_dr)
                        else:
                            cur_dt = cur_lags[:,ridx,cidx][cur_use_mask]
                            cur_dr = np.true_divide(self._invert_monotonic(cur_data, qdr_g), self.qValue)
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
                        if debug:
                            intercept, p_value = np.nan, np.nan
                        
                    vmap[tidx,ridx,cidx] = slope
                    if return_err:
                        verr[tidx,ridx,cidx] = std_err
                    if debug:
                        cur_nvals[ridx,cidx] = len(cur_dr)
                        cur_interc[ridx,cidx] = intercept
                        cur_pval[ridx,cidx] = p_value

            write_vmap.WriteData(vmap[tidx], closeAfter=False)
            if return_err:
                write_verr.WriteData(verr[tidx], closeAfter=False)
            if debug:
                write_interc.WriteData(cur_interc, closeAfter=False)
                write_pval.WriteData(cur_pval, closeAfter=False)
                write_nvals.WriteData(cur_nvals, closeAfter=False)
                print('t={0} -- slope range:[{1},{2}], interc range:[{3},{4}], elapsed: {5:.1f}s'.format(tidx, np.nanmin(vmap[tidx]), np.nanmax(vmap[tidx]),\
                                                                                      np.nanmin(cur_interc), np.nanmax(cur_interc), time.time()-start_time))

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