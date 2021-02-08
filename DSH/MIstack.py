import os
import numpy as np
import logging
from DSH import Config as cf
from DSH import MIfile as MI
from DSH import SharedFunctions as sf

def LoadFolder(mi_folder, config_fname, config_section='MIfile', mi_prefix='', mi_ext='.dat', mi_sort='ASC', open_mifiles=True, stack_type='tau'):
    """Searches for MIfiles in a folder and loads them
        
    Parameters
    ----------
    mi_folder      : folder to search.
    config_fname   : full path of configuration filename
    config_section : title of the configuration section
    mi_prefix      : filter only files in folder whose filename begins with mi_prefix
    mi_ext         : filter only files in folder whose filename ends with mi_ext
    mi_sort        : sort filenames in ascending (ASC) or descending (DESC) order
    open_mifiles   : if True, open all MIfiles for reading
    stack_type     : {'tau','t'}, sets MIstack.StackType
    
    Returns
    -------
    conf_cmaps: configuration file for correlation maps
    cmap_mifiles: list of correlation maps, one per time delay
    all_lagtimes: list of lagtimes
    """
    assert os.path.isdir(mi_folder), 'MIfile folder ' + str(mi_folder) + ' not found.'
    assert os.path.isfile(config_fname), 'Configuration file ' + str(config_fname) + ' not found'
    all_mi_fnames = sf.FindFileNames(mi_folder, Prefix=mi_prefix, Ext=mi_ext, Sort=mi_sort, AppendFolder=True)
    mi_stack = MIstack(MIfiles=all_mi_fnames, MetaData=config_fname, MetaDataSection=config_section, Load=True, OpenFiles=open_mifiles, StackType=stack_type)
    return mi_stack

class MIstack():
    """ Class containing a stack of MIfile sharing the same folder and configuration file """
    
    def __init__(self, MIfiles=[], MetaData=None, MetaDataSection='MIfile', Load=False, OpenFiles=True, StackType='tau'):
        """Initialize MIstack
        
        Parameters
        ----------
        MIfiles         : list of MIfiles. Can be a list of filenames, a list of MIfile objects
                          NOTE: MIfiles must have identical image shape (num_imgs, num_rows, num_cols)
        MetaData        : metadata common to all MIfiles. string or dict. 
                          if string: filename of metadata file
                          if dict: dictionary with metadata.
        MetaDataSection : load subsection of the configuration parameters
                          it only used to load self.MetaData if MetaData is dict.
                          By contrast, it is used to load the single MIfile.MetaData: for that, the typical choice is 'MIfile'
        Load            : if True, load metadata and MIfiles directly upon initialization.
        OpenFiles       : if loading MIfiles, eventually open them for reading
        StackType       : {'tau','t'} 
                          if 't', interpret MIfiles as consecutive parts of a long experiment
                          if 'tau' (default), interpret different MIfiles as videos of the same time range
                          where what varies is a fourth dimension (e.g. z or tau).
                          Note: in practice, all functions are accessible independently of the stack type
                          but the distinction is made for the sake of clarity and for future implementations
        """
        
        self.MIfiles = MIfiles
        self.MetaData = MetaData
        self.IdxList = []
        self.MIshape = [0,0,0]
        self.StackType = StackType
        self._loaded = False
        if Load:
            self.LoadMetadata(MetaDataSection=MetaDataSection)
            if (len(MIfiles)>0):
                if (isinstance(MIfiles[0],str)):
                    self.LoadFiles(MIfiles, metadata_section=MetaDataSection, open_mifiles=OpenFiles, replace_previous=True)
    
    def __repr__(self):
        return '<MIstack [%s]: %sx%sx%sx%sx%s bytes>' % (self.StackType, self.Count(), self.ImgsPerMIfile, self.ImgHeight, self.ImgWidth, self.PixelDepth)
    
    def __str__(self):
        str_res  = '\n|----------------|'
        str_res += '\n| MIstack class: |'
        str_res += '\n|----------------+---------------'
        str_res += '\n| MIfile number  : ' + str(self.Count())
        str_res += '\n| MIshape        : ' + str(self.MIshape) + ' px'
        str_res += '\n| Image number   : ' + str(self.ImageNumber())
        str_res += '\n| Pixel format   : ' + str(self.PixelFormat) + ' (' + str(self.PixelDepth) + ' bytes/px)'
        str_res += '\n| Stack type     : ' + str(self.StackType)
        str_res += '\n|----------------+---------------'
        return str_res

    def __del__(self):
        self.CloseAll()
        
    def IsStack(self):
        return True

    def LoadMetadata(self, MetaData=None, MetaDataSection=None):
        """Load metadata from dict or filename
        
        Parameters
        ----------
        MetaData : string or dict.
        MetaDataSection : if self.MetaData is a dictionnary, load subsection of the configuration parameters
        """
        if (MetaData is not None):
            self.MetaData = MetaData
        assert (self.MetaData is not None), 'No Metadata to be loaded'
        self.MetaData = cf.LoadMetadata(self.MetaData, MetaDataSection)
        if 'MIfile' not in self.MetaData.GetSections():
            logging.warn('No MIfile section found in MIstack metadata (available sections: ' + str(self.MetaData.GetSections()) + ')')
        else:
            logging.debug('Now loading MIstack.MetaData from Config object. Available sections: ' + str(self.MetaData.GetSections()) +
                          ' -- MIfile keys: ' + str(self.MetaData.ToDict(section='MIfile')))
        self.MIshape = self.MetaData.Get('MIfile', 'shape', [0,0,0], int)
        self.hdrSize = self.MetaData.Get('MIfile', 'hdr_len', 0, int)
        self.gapBytes = self.MetaData.Get('MIfile', 'gap_bytes', 0, int)
        self.ImgsPerMIfile = self.MIshape[0]
        self.ImgHeight = self.MIshape[1]
        self.ImgWidth = self.MIshape[2]
        self.PxPerImg = self.ImgHeight * self.ImgWidth
        self.PixelFormat = self.MetaData.Get('MIfile', 'px_format', 'B', str)
        self.PixelDepth = MI._data_depth[self.PixelFormat]
        self.PixelDataType = MI._data_types[self.PixelFormat]
        self.FPS = self.MetaData.Get('MIfile', 'fps', 1.0, float)
        self.PixelSize = self.MetaData.Get('MIfile', 'px_size', 1.0, float)


    def LoadFiles(self, mi_fnames, metadata_section='MIfile', open_mifiles=True, replace_previous=False):
        """Load list of filenames
        
        Parameters
        ----------
        mi_fnames        : list of filenames (full path, str)
        open_mifiles     : if True, open each MIfile for reading
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
    def GetMIfile(self, MI_idx):
        if (MI_idx>0 and MI_idx<len(self.MIfiles)):
            return self.MIfiles[MI_idx]
        else:
            return None
    def Count(self):
        return len(self.MIfiles)    
    def GetMetadata(self, section=None):
        assert isinstance(self.MetaData, cf.Config), 'MetaData not loaded yet: ' + str(self.MetaData)
        return self.MetaData.ToDict(section=section)
    def ImageShape(self):
        return (self.MIshape[1], self.MIshape[2])
    def ImageHeight(self):
        return int(self.ImgHeight)
    def ImageWidth(self):
        return int(self.ImgWidth)    
    def ImageNumber(self):
        return len(self.MIfiles)*self.ImgsPerMIfile    
    def DataFormat(self):
        return self.PixelFormat
    def ValidateROI(self, ROI):
        return MI.ValidateROI(ROI, self.ImageShape())
    def Validate_zRange(self, zRange, replaceNone=True):
        return MI.Validate_zRange(zRange, self.ImageNumber(), replaceNone)
    def GetFPS(self):
        return float(self.FPS)
    def GetPixelSize(self):
        return float(self.PixelSize)

    def OpenForReading(self):
        if (self.MIfiles is not None):
            for midx in range(len(self.MIfiles)):
                if isinstance(self.MIfiles[midx], MI.MIfile):
                    self.MIfiles[midx].OpenForReading()
    def Close(self):
        self.CloseAll()
    def CloseAll(self):
        if (self.MIfiles is not None):
            for midx in range(len(self.MIfiles)):
                if isinstance(self.MIfiles[midx], MI.MIfile):
                    self.MIfiles[midx].Close()
    
    def GetImage(self, img_idx, MI_idx=None, cropROI=None):
        """Read single image from MIfile
        
        Parameters
        ----------
        img_idx : index of the image, 0-based. If -N, it will get the Nth last image
        MI_idx  : index of the MIfile from which the image has to be read.
                  if MI_idx is None, the stack is considered as a 't' stack, and 
                  img_idx ranges from 0 to ImageNumber() and is interpreted as the 
                  index of the image since the beginning of the stack (MI #0).
                  In this case, the MIfile containing the image is calculated at runtime
                  Otherwise, if MI_idx is given, img_idx ranges from 0 to ImgsPerMIfile
        cropROI : if None, full image is returned
                  otherwise, [topleftx (0-based), toplefty (0-based), width, height]
                  width and/or height can be -1 to signify till the end of the image
        """
        if MI_idx is None:
            MI_idx = img_idx // self.ImgsPerMIfile
            img_in_mi = img_idx % self.ImgsPerMIfile
        else:
            img_in_mi = img_idx
        return self.MIfiles[MI_idx].GetImage(img_in_mi, cropROI=cropROI)
                
    def GetTimetrace(self, pxLocs, zRange=None, idx_list=None, excludeIdxs=[], returnCoords=False,\
                         squeezeResult=True, readConsecutive=1, lagFlip=False, zStep=1, mask_cropROI=None):
        """Returns (t, tau) data for a given set of pixels (assumes self.StackType=='tau')
        
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
        res = self.GetValues(pxLocs, tvalues, idx_list, do_squeeze=squeezeResult, readConsecutive=readConsecutive, lagFlip=lagFlip, zStep=zStep, mask_cropROI=mask_cropROI)
        
        if returnCoords:
            return res, tvalues, idx_list, lagFlip
        else:
            return res
        
    def GetValues(self, pxLocs, tList, idx_list, do_squeeze=True, readConsecutive=1, lagFlip=None, zStep=1, mask_cropROI=None):
        """Get values relative to a bunch of pixel location, time points and lags (assumes self.StackType=='tau')
        
        Parameters
        ----------
        pxLocs:           pixel location [row, col] or list of pixel locations or list of 2D binary masks
                          In the case of binary mask, the whole image is read and averaged on the nonzero mask values
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
        mask_cropROI:     if working with masks, specify ROI to be loaded. Is shape should correspond to the mask
        """
        
        assert self._loaded, 'MIstack needs to be loaded first'
        if (type(pxLocs[0]) not in [list, tuple, np.ndarray]):
            pxLocs = [pxLocs]
        if (len(np.array(pxLocs[0]).shape)>1):
            use_mask=True
            mask_avg = []
            for midx in range(len(pxLocs)):
                mask_avg.append(np.mean(pxLocs[midx]))
            logging.debug('Average ROIs: ' + str(mask_avg))
        else:
            use_mask=False
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
                            if use_mask:
                                res[pidx, lidx, tidx] = np.nanmean(np.multiply(cur_mifile.GetImage(img_idx=img_idx, cropROI=mask_cropROI), pxLocs[pidx])) * 1.0/mask_avg[pidx]
                            else:
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
