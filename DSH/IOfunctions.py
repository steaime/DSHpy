import sys
import os
import logging
import numpy as np

from DSH import SharedFunctions as sf

def LoadResFile(fname, readHeader=True, isolateFirst=0, delimiter=',', comments='#'):
    if (readHeader):
        f = open(fname)
        header = f.readline()
        hdr_list = header.split(delimiter)
    res_arr = np.loadtxt(fname, comments=comments, delimiter=delimiter)
    if (isolateFirst>0):
        firstcol = np.squeeze(res_arr[:,:isolateFirst])
        res_arr = np.squeeze(res_arr[:,isolateFirst:])
        hdr_list = hdr_list[isolateFirst:]
        if (readHeader):
            return res_arr, hdr_list, firstcol
        else:
            return res_arr, firstcol
    else:
        if (readHeader):
            return res_arr, hdr_list
        else:
            return res_arr

def OpenRawSLS(fname, delimiter='\t', comments='#')
    res_Ir, res_hdr, roi_coord = LoadResFile(fname, delimiter=delimiter, comments=comments, readHeader=True, isolateFirst=2)
    times = np.asarray([sf.FirstFloatInStr(hdr) for hdr in res_hdr])
    exptimes = np.asarray([sf.LastFloatInStr(hdr) for hdr in res_hdr])
    return np.squeeze(res_Ir.reshape((res_Ir.shape[0], -1, len(set(exptimes))))), roi_coord[:,0], roi_coord[:,1], times, exptimes

def OpenSLS(fname, delimiter='\t', comments='#'):
    res_Ir, res_hdr, roi_coord = LoadResFile(fname, delimiter=delimiter, comments=comments, readHeader=True, isolateFirst=2)
    times = np.asarray([sf.FirstFloatInStr(hdr) for hdr in res_hdr])
    return res_Ir, roi_coord[:,0], roi_coord[:,1], times

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
    ROI_list = [sf.FirstIntInStr(name) for name in fnames_list]
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