import sys
import os
import logging
import numpy as np

from DSH import SharedFunctions as sf

def LoadResFile(fname, readHeader=True, isolateFirst=0, delimiter=',', comments='#'):
    if (readHeader):
        f = open(fname)
        header = f.readline()[len(comments):]
        hdr_list = header.strip().split(delimiter)
    res_arr = np.loadtxt(fname, comments=comments, delimiter=delimiter)
    if (isolateFirst>0):
        firstcol = np.squeeze(res_arr[:,:isolateFirst])
        res_arr = np.squeeze(res_arr[:,isolateFirst:])
        if (readHeader):
            hdr_list = hdr_list[isolateFirst:]
            return res_arr, hdr_list, firstcol
        else:
            return res_arr, firstcol
    else:
        if (readHeader):
            return res_arr, hdr_list
        else:
            return res_arr

def OpenRawSLS(fname, roi_numcoords=2, delimiter='\t', comments='#'):
    res_Iav, res_hdr, roi_coord = LoadResFile(fname, delimiter=delimiter, comments=comments, readHeader=True, isolateFirst=roi_numcoords)
    exptimes = np.asarray([sf.LastFloatInStr(hdr) for hdr in res_hdr])
    num_exptimes = len(set(exptimes))
    times = np.squeeze(np.asarray([sf.FirstFloatInStr(hdr) for hdr in res_hdr]).reshape((-1, num_exptimes)))
    exptimes = np.squeeze(exptimes.reshape((-1, num_exptimes)))
    return np.squeeze(res_Iav.reshape((res_Iav.shape[0], -1, num_exptimes))), roi_coord, times, exptimes

def OpenSLS(fname, roi_numcoords=2, delimiter='\t', comments='#'):
    res_Iavg, res_hdr, roi_coord = LoadResFile(fname, delimiter=delimiter, comments=comments, readHeader=True, isolateFirst=roi_numcoords)
    times = np.asarray([sf.FirstFloatInStr(hdr) for hdr in res_hdr])
    return res_Iavg, roi_coord, times

def ReadCIfile(fpath, time_colidx=1, delimiter='\t', comments='#'):
    """Loads a CI file
    
    Parameters
    ----------
    fpath :         full path of the CI file to read
                    a CI file is assumed to have:
                    - (time_colidx+1) initial columns. The last one contains image times, the following ones are data columns
                    - a one-line header. Header of data columns has to report the lagtime, in image units (integer)
    time_colidx :   column index of image times.
                
    Returns
    -------
    cI_data :      2D array. Element [i,j] is correlation between image i and image i+lags[j]
    times :        array of image times in physical units (float)
    lagidx_list :  list of lagtimes, in image units (int)
    """
    data = np.loadtxt(fpath, delimiter=delimiter, comments=comments, skiprows=1)
    times = data[:,time_colidx]
    cI_data = data[:,time_colidx+1:]
    with open(fpath, "r") as file:
        hdr_line = file.readline().strip()
    lagidx_list = sf.ExtractIndexFromStrings(hdr_line.split(delimiter)[time_colidx+1:])
    return cI_data, times, lagidx_list
    
def OpenCIs(froot, fname_prefix='cI_', time_colidx=1, delimiter='\t', comments='#'):
    res = []
    fnames_list = sf.FindFileNames(froot, Prefix=fname_prefix, Ext='.dat', Sort='ASC')
    ROI_list = [sf.FirstIntInStr(name) for name in fnames_list]
    exptime_list = [sf.LastIntInStr(name) for name in fnames_list]
    lagtimes = []
    imgtimes = []
    for i, fname in enumerate(fnames_list):
        res_cI, res_hdr, col_times = LoadResFile(os.path.join(froot, fname), delimiter=delimiter, comments=comments, 
                                                   readHeader=True, isolateFirst=time_colidx+1)
        if col_times.ndim>1: #this happens if time_colidx>0
            col_times = col_times[:,time_colidx]
        res.append(res_cI)
        lagtimes.append(np.asarray([sf.FirstIntInStr(hdr) for hdr in res_hdr]))
        imgtimes.append(col_times)
    return res, imgtimes, lagtimes, ROI_list, exptime_list

def OpenG2M1s(froot, expt_idx=None, roi_idx=None, fname_prefix='g2m1_', time_colidx=1, delimiter='\t', comments='#'):
    res = []
    filter_str = ''
    if roi_idx is not None:
        filter_str += '_ROI' + str(roi_idx).zfill(3)
    if expt_idx is not None:
        filter_str += '_e' + str(expt_idx).zfill(2)
    fnames_list = sf.FindFileNames(froot, Prefix=fname_prefix, Ext='.dat', FilterString=filter_str, Sort='ASC')
    ROI_list = [sf.AllIntInStr(name)[-2] for name in fnames_list]
    exptime_list = [sf.LastIntInStr(name) for name in fnames_list]
    lagtimes = []
    imgtimes = []
    for i, fname in enumerate(fnames_list):
        res_g2m1, res_hdr = LoadResFile(os.path.join(froot, fname), delimiter=delimiter, comments=comments, 
                                                   readHeader=True, isolateFirst=0)
        res.append(res_g2m1[:,1::2].T)
        lagtimes.append(res_g2m1[:,::2].T)
        imgtimes.append(np.asarray([sf.FirstFloatInStr(res_hdr[j]) for j in range(1, len(res_hdr), 2)]))
    return res, lagtimes, imgtimes, ROI_list, exptime_list

def LoadImageTimes(img_times_source, usecols=0, skiprows=1, root_folder=None, return_unique=False):
    '''
    Load image times from file or list of files
    
    Parameters
    ----------
    - img_times_source : file name or list of filenames
    - usecols          : index of column containing image times in file
    - skiprows         : number of rows to be skipped at the beginning of the file
    - root_folder      : root folder path. If specified, img_times_source will be interpreted as a relative path
    - return_unique    : if True, remove duplicates before returning result
    '''
    if img_times_source is not None:
        if usecols is None:
            max_col = 0
        elif np.isscalar(usecols):
            max_col = usecols
        else:
            max_col = np.max(usecols)
        # if img_times_source is a string, let's use a single text file as input.
        # otherwise, it can be a list: in that case, let's open each text file and append all results
        if (isinstance(img_times_source, str)):
            if root_folder is None:
                fpath = img_times_source
            else:
                fpath = os.path.join(root_folder, img_times_source)
            if sf.CountFileColumns(fpath, firstlineonly=False) > max_col:
                res = np.loadtxt(fpath, dtype=float, usecols=usecols, skiprows=skiprows, ndmin=2)
            else:
                res = None
                logging.warning('DSH.SALS.LoadImageTimes(): file ' + str(fpath) + ' incompatible with usecols ' + str(usecols) + '. Returning default value ' + str(res))
        else:
            res = np.empty(shape=(0,), dtype=float)
            for cur_f in img_times_source:
                if root_folder is None:
                    fpath = cur_f
                else:
                    fpath = os.path.join(root_folder, cur_f)
                if sf.CountFileColumns(fpath, firstlineonly=False) > max_col:
                    res = np.append(res, np.loadtxt(fpath, dtype=float, usecols=usecols, skiprows=skiprows, ndmin=2))
                else:
                    logging.warning('DSH.SALS.LoadImageTimes(): file ' + str(fpath) + ' incompatible with usecols ' + str(usecols) + '. Skipping file from list')
            if len(res)<=0:
                res = None
                logging.warning('DSH.SALS.LoadImageTimes(): no ok file in list ' + str(img_times_source) + ' returning default value ' + str(res))
    else:
        res = None
        logging.debug('DSH.SALS.LoadImageTimes(): no file specified. Returning default value ' + str(res))
    if res is not None and return_unique:
        res = np.unique(res)
    return res

def LoadROIcoords(file_name, delimiter='\t', comments='#'):
    data, hdr = LoadResFile(file_name, readHeader=True, isolateFirst=0, delimiter=delimiter, comments=comments)
    num_coords = data.shape[1]-5
    if num_coords <= 0:
        logging.error('IOfunctions.LoadROIcoords() : error loading file ' + str(file_name) + ' at least 6 columns expected in data (shape=' + str(data.shape) + ')')
        return None, None, None, None
    else:
        ROIcoords = np.squeeze(data[:,:num_coords])
        ROInames = hdr[:num_coords]
        NormFact = data[:,num_coords]
        ROIbb = data[:,num_coords+1:]
    return ROIcoords, ROInames, NormFact, ROIbb