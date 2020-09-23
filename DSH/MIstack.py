import os
import numpy as np
from DSH import Config as cf
from DSH import MIfile as MI
from DSH import SharedFunctions as sf

def LoadFolder(mi_folder, config_fname, config_section='MIfile', mi_prefix='', mi_ext='.dat', mi_sort='ASC', open_mifiles=True):
    """Searches for MIfiles in a folder and loads them
        
    Parameters
    ----------
    mi_folder : folder to search.
    config_fname : full path of configuration filename
    config_section : title of the configuration section
    mi_prefix : filter only files in folder whose filename begins with mi_prefix
    mi_ext :    filter only files in folder whose filename ends with mi_ext
    mi_sort :   sort filenames in ascending (ASC) or descending (DESC) order
    open_mifiles : if True, open all MIfiles for reading
    
    Returns
    -------
    conf_cmaps: configuration file for correlation maps
    cmap_mifiles: list of correlation maps, one per time delay
    all_lagtimes: list of lagtimes
    """
    assert os.path.isdir(mi_folder), 'MIfile folder ' + str(mi_folder) + ' not found.'
    assert os.path.isfile(config_fname), 'Configuration file ' + str(config_fname) + ' not found'
    all_mi_fnames = sf.FindFileNames(mi_folder, Prefix=mi_prefix, Ext=mi_ext, Sort=mi_sort, AppendFolder=True)
    mi_stack = MIstack(MIfiles=all_mi_fnames, MetaData=config_fname, MetaDataSection=config_section, Load=True, OpenFiles=open_mifiles)
    return mi_stack

class MIstack():
    """ Class containing a stack of MIfile sharing the same folder and configuration file """
    
    def __init__(self, MIfiles=[], MetaData=None, MetaDataSection=None, Load=False, OpenFiles=True):
        """Initialize MIstack
        
        Parameters
        ----------
        MIfiles : list of MIfiles. Can be a list of filenames, a list of MIfile objects
        MetaData : metadata common to all MIfiles. string or dict. 
                    if string: filename of metadata file
                    if dict: dictionary with metadata.
        MetaDataSection : load section of the configuration file
        Load : if True, load metadata and MIfiles directly upon initialization.
        OpenFiles : if loading MIfiles, eventually open them for reading
        """
        
        self.MIfiles = MIfiles
        self.MetaData = MetaData
        self.IdxList = []
        self._loaded = False
        if Load:
            self.LoadMetadata(MetaDataSection=MetaDataSection)
            if (len(MIfiles)>0):
                if (isinstance(MIfiles[0],str)):
                    self.LoadFiles(MIfiles, metadata_section=MetaDataSection, open_mifiles=OpenFiles, replace_previous=True)

    def __del__(self):
        self.CloseAll()

    def LoadMetadata(self, MetaData=None, MetaDataSection=None):
        """Load metadata
        
        Parameters
        ----------
        MetaData : string or dict. If None, self.MetaData_init will be used, if available
        MetaDataSection : if self.MetaData is a string, load section of the configuration file
        """
        if (MetaData is not None):
            self.MetaData = MetaData
        assert (self.MetaData is not None), 'No Metadata to be loaded'
        self.MetaData = cf.LoadMetadata(self.MetaData, MetaDataSection)

    def LoadFiles(self, mi_fnames=None, metadata_section='MIfile', open_mifiles=True, replace_previous=False):
        """Load list of filenames
        
        Parameters
        ----------
        mi_fnames : list of filenames (full path)
        open_mifiles : if True, open each MIfile for reading
        replace_previous : if True, replace eventual preexisting list of MIfile
        """
        if (replace_previous or self.MIfiles is None):
            self.MIfiles = []
            self.IdxList = []
        for i in range(len(mi_fnames)):
            self.IdxList.append(sf.LastIntInStr(mi_fnames[i]))
            self.MIfiles.append(MI.MIfile(mi_fnames[i], self.MetaData.ToDict(section=metadata_section)))
            if open_mifiles:
                self.MIfiles[-1].OpenForReading()
        self._loaded = True
    
    def MIindexes(self):
        return self.IdxList
    
    def GetMIfiles(self):
        return self.MIfiles
    
    def Count(self):
        return len(self.MIfiles)
    
    def GetMetaData(self, section=None):
        assert isinstance(self.MetaData, cf.Config), 'MetaData not loaded yet: ' + str(self.MetaData)
        return self.MetaData.ToDict(section=section)
    
    def CloseAll(self):
        for midx in range(len(self.MIfiles)):
            if isinstance(self.MIfiles[midx], MI.MIfile):
                self.MIfiles[midx].Close()
                
                
    def GetTimetrace(self, pxLocs, zRange=None, idx_list=None, excludeIdxs=[], returnCoords=False,\
                         squeezeResult=True, readConsecutive=1, lagFlip=False, zStep=1):
        """Returns (t, tau) data for a given set of pixels
        
        Parameters
        ----------
        pxLocs :          list of pixel locations, each location being a tuple (row, col)
        zRange :          range of time (or z) slices to sample
        idx_list :         list of lagtimes. If None, all available lagtimes will be loaded,
                          except the ones eventually contained in excludeLags
        excludeIdxs:      if idx_list is None, list of lagtimes to be excluded
        returnCoords :    if True, also returns the list of times, lagtimes and lagsigns
        squeezeResult :   if True, squeeze the output numpy array to remove dimensions if size along axis is 1
        readConsecutive : positive integer. Number of consecutive pixels to read starting from each pxLoc
                          if >1 the output will still be a 3D array, but the first dimension will be
                          len(pxLoc) x readConsecutive
        lagFlip :     boolean parameter (or string=='BOTH'), useful when MI index is a lagtime (e.g. for correlation maps)
                          if False: standard correlations will be returned
                          if True: correlations having the given time as second correlated time
                                   will be returned if available
                          if 'BOTH': both signs will be stitched together
        
        Returns
        -------
        res:              If only one pixel was asked, single 2D array: one row per time delay
                          Otherwise, 3D array, one matrix per pixel
        if returnCoords is True:
        tvalues:          list of image times
        idx_list:          list of lagtimes, in image units. All elements are also contained in self.cmapStack.IdxList
                          if lagFlip=='BOTH' lags are duplicated, first descending and then ascending
        lagFlip:          boolean value or list of boolean values indicating whether the lagtime is flipped
                          if signle boolean values, all values are interpreted as equally flipped
        """
        
        assert self._loaded, 'MIstack needs to be loaded first'
        if idx_list is None:
            idx_list = self.IdxList
        lagSet = (set(idx_list) & set(self.IdxList)) - set(excludeIdxs)
        if (lagFlip=='BOTH'):
            lagList_pos = list(lagSet).copy()
            lagList_pos.sort()
            listFlip_pos = list(np.ones_like(lagList_pos, dtype=bool)*False)
            lagList_neg = list(lagSet).copy()
            lagList_neg.sort(reverse=True)
            # 0 is only counted once
            if 0 in lagList_neg:
                lagList_neg.remove(0)
            listFlip_neg = list(np.ones_like(lagList_neg, dtype=bool)*True)
            idx_list = lagList_neg+lagList_pos
            lagFlip = listFlip_neg+listFlip_pos
        else:
            idx_list = list(lagSet)
            idx_list.sort(reverse=lagFlip)
        
        tvalues = list(range(*self.MIfiles[1].Validate_zRange(zRange)))
        res = self.GetValues(pxLocs, tvalues, idx_list, do_squeeze=squeezeResult, readConsecutive=readConsecutive, lagFlip=lagFlip, zStep=zStep)
        
        if returnCoords:
            return res, tvalues, idx_list, lagFlip
        else:
            return res
        
    def GetValues(self, pxLocs, tList, idx_list, do_squeeze=True, readConsecutive=1, lagFlip=None, zStep=1):
        """Get values relative to a bunch of pixel location, time points and lags
        
        Parameters
        ----------
        pxLocs:           pixel location [row, col] or list of pixel locations
        tList :           time, in image units, or list of times
        idx_list:          MI index (must be present in self.IdxList)
                          or list of lagtimes
        do_squeeze :      if True, squeeze the output numpy array to remove dimensions if size along axis is 1
        readConsecutive : positive integer. Number of consecutive pixels to read starting from each pxLoc
                          if >1 the output will still be a 3D array, but the first dimension will be
                          len(pxLoc) x readConsecutive
        lagFlip:          boolean or list of boolean with same shape as lagList
                          indicates whether the time specified in tList is the smallest (lagFlip==False)
                          or the largest (lagFlip==True) of the two times that are correlated
        zStep:            Multiplicative factor for converting MIindex into lagtime (in image units)
                          Useful whenever the current MIfile was generated processing every N images
        """
        
        assert self._loaded, 'MIstack needs to be loaded first'
        if (type(pxLocs[0]) not in [list, tuple, np.ndarray]):
            pxLocs = [pxLocs]
        if (type(tList) not in [list, tuple, np.ndarray]):
            tList = [tList]
        if (type(idx_list) not in [list, tuple, np.ndarray]):
            idx_list = [idx_list]
        if lagFlip is None:
            lagFlip = np.ones_like(idx_list, dtype=bool)
        elif (type(lagFlip) not in [list, tuple, np.ndarray]):
            lagFlip = np.ones_like(idx_list, dtype=bool)*lagFlip
        if readConsecutive>1:
            res_shape = (len(pxLocs), len(idx_list), len(tList), readConsecutive)
        else:
            res_shape = (len(pxLocs), len(idx_list), len(tList))
        res = np.ones(res_shape)*np.nan
        for lidx in range(res.shape[1]):
            cur_midx = self.IdxList.index(idx_list[lidx])
            cur_mifile = self.MIfiles[cur_midx]
            if cur_mifile is not None:
                for tidx in range(res.shape[2]):
                    if lagFlip[lidx]:
                        if tList[tidx] >= idx_list[lidx]:
                            img_idx = tList[tidx]-int(idx_list[lidx]/zStep)
                        else:
                            img_idx = None
                    else:
                        img_idx = tList[tidx]
                    if img_idx is not None:
                        for pidx in range(res.shape[0]):
                            res[pidx, lidx, tidx] = cur_mifile._read_pixels(px_num=readConsecutive,\
                                       seek_pos=cur_mifile._get_offset(img_idx=img_idx, row_idx=pxLocs[pidx][0], col_idx=pxLocs[pidx][1]))
        if readConsecutive>1:
            res = np.moveaxis(res, -1, 1)
            new_shape = (res.shape[0]*res.shape[1], res.shape[2], res.shape[3])
            res = res.reshape(new_shape)
            
        if do_squeeze:
            return np.squeeze(res)
        else:
            return res