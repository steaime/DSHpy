import os
import re
import inspect
import logging
import datetime
import math
import numpy as np
import collections
import collections.abc
import ast
    

def AllIntInStr(my_string):
    arr_str = re.findall(r'\d+', my_string)
    res_int = []
    for item in arr_str:
        res_int.append(int(item))
    return res_int

def FirstIntInStr(my_string):
    arr = AllIntInStr(my_string)
    if (len(arr) > 0):
        return arr[0]
    else:
        return None

def LastIntInStr(my_string):
    arr = AllIntInStr(my_string)
    if (len(arr) > 0):
        return arr[len(arr)-1]
    else:
        return None
    
def AllFloatInStr(my_string):
    arr_str = re.findall(r"[-+]?\d*\.\d+|\d+", my_string)
    res_float = []
    for item in arr_str:
        res_float.append(float(item))
    return res_float

def FirstFloatInStr(my_string):
    arr = AllFloatInStr(my_string)
    if (len(arr) > 0):
        return arr[0]
    else:
        return None
    
def LastFloatInStr(my_string):
    arr = AllFloatInStr(my_string)
    if (len(arr) > 0):
        return arr[-1]
    else:
        return None
    
def ValidateRange(Range, MaxVal, MinVal=None, replaceNone=True):
    """ Validates a range of positive quantities (Note: must be positive!)
    """
    if Range is None:
        if replaceNone:
            return [MinVal, MaxVal, 1]
        else:
            return None
    if MinVal is not None:
        if (Range[0] < MinVal):
            Range[0] = MinVal
    if (Range[1] < 0):
        Range[1] = MaxVal
    if (len(Range) < 3):
        Range.append(1)
    return Range

def PathJoinOrNone(root, folder):
    if (folder is None):
        return None
    else:
        return os.path.join(root, folder)
    
def ReportCyclic(var, start_val=-np.pi, period=2*np.pi):
    """ Reports a given cyclic variable (e.g. an angle) 
    in the range [start_val, start_val+period)

    Parameters
    ----------
    var : float, array-like. Variable to be reportedd
    start_val : float, array-like. Minimum value the variable can take
    period : float, array-like. Winding period

    Returns
    -------
    reported variable
    """
    return np.add(np.mod(np.subtract(var, start_val), period), start_val)

def ReportAngle(angle, start_angle=-np.pi):
    return ReportCyclic(angle, start_val=start_angle, period=2*np.pi)

def StrParse(my_string, cast_type=None):
    """ Casts a string into the desired type, 
    taking care of lists, tuples, None values and dictionnaries
    
    Parameters
    ----------
    my_string : str, input string
    cast_type : data type. if None, just cast string into None, list, tuple or dict, if relevant
                If data type is specified (e.g. int, float, bool), variable 
                (or every element in variable, if applicable) will be cast to given type

    Returns
    -------
    res : list, tuple, dict or variable type depending on the input string and cast_type
    """
    res = my_string
    if str(res).strip().lower() in ['none', '']:
        return None
    elif (str(res)[0] in ['[','(', '{']):
        res = ast.literal_eval(str(res))
    if (type(res) in [list]):
        for i in range(len(res)):
            if (cast_type is not None):
                if (type(res[i]) in [list]):
                    for j in range(len(res[i])):
                        res[i][j] = cast_type(res[i][j])
                else:
                    res[i] = cast_type(res[i])
        return res
    elif (type(res) in [tuple]):
        if (cast_type is not None):
            return tuple(map(cast_type, res))
        else:
            return res
    elif (cast_type is bool):
        if (type(res) is bool):
            return res
        else:
            return str(my_string).strip().lower() in ['true', '1', 't', 'y', 'yes']
    elif (cast_type is float):
        if (type(res) is float):
            return res
        elif (res == 'nan'):
            return np.nan
        else:
            return float(my_string)
    else:
        if (cast_type is None):
            return res
        elif (type(res) is cast_type):
            return res
        else:
            return cast_type(res)
        
def GetFilenameFromCompletePath(my_string):
    res = None
    if len(my_string)>0:
        split1 = my_string.split('\\')
        if len(split1)>0:
            split2 = split1[-1].split('/')
            if len(split2)>0:
                res = split2[-1]
    return res

def GetFolderFromCompletePath(my_string):
    return os.path.dirname(my_string)

def GetAbsolutePath(path, root_path=None):
    if os.path.isabs(path):
        return path
    else:
        if root_path is None:
            return os.path.abspath(path)
        else:
            return os.path.abspath(os.path.join(root_path, path))
            # for the relative path: relative_path = os.path.relpath(absolute_path, root_path)

def CheckCreateFolder(folderPath):
    if (os.path.isdir(folderPath)):
        return True
    else:
        os.makedirs(folderPath)
        print("Created folder: {0}".format(folderPath))
        return False
    
def IsIterable(var):
    return isinstance(var, collections.abc.Iterable)

def IterableShape(var):
    if isinstance(var, np.ndarray):
        return np.prod(var.shape)
    else:
        res = 1
        cur_slice = var
        while(IsIterable(cur_slice)):
            res *= len(cur_slice)
            cur_slice = cur_slice[0]
        return res
    
def UpdateDict(dict_orig, dict_update):
    if dict_orig is None:
        return dict_update
    else:
        if dict_update is not None:
            for k, v in dict_update.items():
                if isinstance(v, collections.abc.Mapping):
                    dict_orig[k] = UpdateDict(dict_orig.get(k, {}), v)
                else:
                    dict_orig[k] = v
        return dict_orig

def CheckIterableVariable(var, n_dim, force_length=True, cast_type=None):
    """ Checks if a variable is an iterable with given dimensions (n_dim). 
    If so, returns the variable itself. Otherwise, returns a list of 
    the variable replicated n_dim times
    if cast_type is not None, variable will be cast into cast_type before being replicated
    """
    
    assert n_dim > 0, 'Number of dimensions must be strictly positive'
    
    if IsIterable(var):
        if force_length==True and len(var)!=n_dim:
            if cast_type is None:
                return [var] * n_dim
            else:
                return [cast_type(var)] * n_dim
        else:
            return var
    else:
        if cast_type is None:
            return [var] * n_dim
        else:
            return [cast_type(var)] * n_dim

'''
Sort: None not to sort, or: 'ASC' | 'DESC' to sort output list ascending | descending
'''
def FilterStringList(my_list, Prefix='', Ext='', Step=-1, FilterString='', ExcludeStrings=[], Verbose=0, Sort=None):
    if Verbose>0:
        print('before filter: {0} files'.format(len(my_list)))
    if (len(Prefix) > 0):
        my_list = [i for i in my_list if str(i).find(Prefix) == 0]
    if (len(Ext) > 0):
        my_list = [i for i in my_list if i[-len(Ext):] == Ext]
    if (len(FilterString) > 0):
        my_list = [i for i in my_list if FilterString in i]
    if (len(ExcludeStrings) > 0):
        for excl_str in ExcludeStrings:
            my_list = [i for i in my_list if excl_str not in i]
    if Verbose>0:
        print('after filter: {0} files'.format(len(my_list)))
    if (Sort=='ASC' or Sort=='asc'):
        my_list.sort(reverse=False)
    elif (Sort=='DESC' or Sort=='desc'):
        my_list.sort(reverse=True)
    if (Step > 0):
        resList = []
        for idx in range(len(my_list)):
            if (idx % Step == 0):
                resList.append(my_list[idx])
        return resList
    else:
        return my_list
    
def FindFileNames(FolderPath, Prefix='', Ext='', Step=-1, FilterString='', ExcludeStrings=[], Verbose=0, AppendFolder=False, Sort=None):
    if Verbose>0:
        print('Sarching {0}{1}*{2}'.format(FolderPath, Prefix, Ext))
    FilenameList = []
    for (dirpath, dirnames, filenames) in os.walk(FolderPath):
        FilenameList.extend(filenames)
        break
    FilenameList = FilterStringList(FilenameList, Prefix=Prefix, Ext=Ext, Step=Step, FilterString=FilterString,\
                            ExcludeStrings=ExcludeStrings, Verbose=Verbose, Sort=Sort)
    if AppendFolder:
        for i in range(len(FilenameList)):
            FilenameList[i] = os.path.join(FolderPath, FilenameList[i])
    return FilenameList

"""
FirstLevelOnly: if True, only returns immediate subdirectories, otherwise returns every directory right down the tree
Returns: list with complete paths of each subdirectory
"""
def FindSubfolders(FolderPath, FirstLevelOnly=True, Prefix='', Step=-1, FilterString='', ExcludeStrings=[], Verbose=0, Sort=None):
    if FirstLevelOnly:
        if (Prefix == ''):
            reslist = [os.path.join(FolderPath, o) for o in os.listdir(FolderPath) if os.path.isdir(os.path.join(FolderPath,o))]
        else:
            reslist = [os.path.join(FolderPath, o) for o in os.listdir(FolderPath) if (os.path.isdir(os.path.join(FolderPath,o)) and o[:len(Prefix)]==Prefix)]
    else:
        reslist = [x[0] for x in os.walk(FolderPath)]
    return FilterStringList(reslist, Prefix='', Ext='', Step=Step, FilterString=FilterString, ExcludeStrings=ExcludeStrings, Verbose=Verbose, Sort=Sort)
    
"""
FilenameList:    list of filenames
index_pos:       index of the desired integer in the list of integer found in each string
"""
def ExtractIndexFromStrings(StringList, index_pos=0, index_notfound=-1):
    res = []
    for cur_name in StringList:
        allInts = AllIntInStr(cur_name)
        if (len(allInts) > 0):
            try:
                val = allInts[index_pos]
                res.append(val)
            except:
                res.append(index_notfound)
        else:
            res.append(index_notfound)
    return res

def filter_kwdict_funcparams(my_dict, my_func):
    return {k: v for k, v in my_dict.items() \
            if k in [p.name for p in inspect.signature(my_func).parameters.values()]}

def LockAcquire(lock):
    if (lock is not None): 
        lock.acquire()

def LockRelease(lock):
    if (lock is not None): 
        lock.release()

def LockPrint(strOut, lock, silent=False):
    if not silent:
        if (lock is not None): 
            lock.acquire()
            print(strOut)
            lock.release()
        else:
            print(strOut)

def LogWrite(strOut, fLog=None, logLevel=None, lock=None, flushAfter=True, silent=True, add_prefix='\n'):
    """
    Log a message
    
    Parameters
    ----------
    strOut :    string, message to be logged
    fLog :      None or handle to text file to write the message with add_prefix prepended
    logLevel :  None or int. If not None, it specifies the logging level for python standard logging console
                Default values are: 
                    logging.NOTSET   = 0
                    logging.DEBUG    = 10
                    logging.INFO     = 20
                    logging.WARN     = 30
                    logging.ERROR    = 40
                    logging.CRITICAL = 50
                if None, no message will be logged
    lock :      None or lock to avoid simultaneous writing to file
    flushAfter: bool. If True, flush the text file after logging
    silent :    bool. If True, print the message to standard print output
    """
    if fLog is not None:
        LockAcquire(lock)
        fLog.write(add_prefix + strOut)
        if flushAfter:
            fLog.flush()
        LockRelease(lock)
    if logLevel is not None:
        logging.log(logLevel, strOut)
    LockPrint(strOut, lock, silent)

def TimeStr(fmt="%H:%M:%S"):
    return NowToStr(fmt=fmt)

def NowToStr(fmt="%m/%d/%Y, %H:%M:%S"):
    return datetime.datetime.now().strftime(fmt)

def MoveListElement(lst, old_idx, new_idx):
    if (new_idx < 0):
        new_idx += len(lst)+1
    lst.insert(new_idx, lst.pop(old_idx))
    return lst

def IsWithinTolerance(t1, t2, tolerance, tolerance_isrelative):
    if tolerance_isrelative:
        return (np.abs(t2 - t1) < tolerance * 0.5 * np.abs(t2 + t1))
    else:
        return (np.abs(t2 - t1) < tolerance)
    
def CheckLoadTxtUsecols(fpath, usecols=None, delimiter=None):
    with open(fpath, 'rb') as f:
        lines = [f.readline()]
    fshape = np.loadtxt(lines, delimiter=delimiter).shape
    if (len(fshape) < 2):
        return False
    elif (fshape[1] <= np.max(usecols)):
        return False

def CountFileLines(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def CountFileColumns(fpath, delimiter=None, singlerow=True, row_index=0):
    if singlerow:
        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                if i==row_index:
                    break
        num_cols = len(line.split(delimiter))
        if i != row_index and row_index > 0:
            logging.warn('DSH.SharedFunctions.CountFileColumns WARNING: row index {0} not found in file {1} ({2} lines). Counting {3} columns in last line.'.format(row_index, fpath, i+1, num_cols))
        return num_cols
    else:
        return np.loadtxt(fpath, delimiter=delimiter, ndmin=2).shape[1]

    
"""
Function parameters:
   - time_per_pair:    Time duration of an image pair, in microseconds
   - delays_per_decade:Number of delays per decade
   - minimum_delay:    Minimum lag, in microseconds
   - cycle_duration:   Duration of an entire cycle, in microseconds.
                       Set it to None to automatically calculate it:
                       it will be num_delays*time_per_pair+start_time
   - start_time:       Delay in microseconds between the beginning
                       of the cycle and the first image
   - verbose:          If True:  prints detailed output
                       If False: only prints critical error notifications
Generates a TimePulses file, formatted as follows:
    - First line: number n_cycle of pulses per cycle
    - 2-nd to (n_cycle+1)-th line: the n_cycle pulse starting time (usec), one per line
    - (n_cycle+2)-th line: cycle duration (must be larger than the n_cycle-th pulse starting time
    - Optional (only relevant if using the interlaced lin-log scheme):
      add 5 lines with recap of the parameters used to generate this file:
      time_per_pair,delays_per_decade,minimum_delay,exposure_time,use_shutter
Returns: None
"""

def SmartTimeSamplingSeries(minimum_delay, pair_duration, cycle_duration, num_points=None):
    '''
    Generates time series of acquisition for 'Smart time sampling' (STS) protocol
    
    Parameters
    ----------
    minimum_delay:    float, Minimum lag
    pair_duration:    float, Constant lag between pairs of datapoints. Equals the largest log-spaced, intra-pair delay
    cycle_duration:   int,   Number of log-spaced, intra-pair delays. 
                      Equals the number of image pairs forming the periodically repeated unit in the acquisition scheme
    num_points:       None or int. Total number of datapoints to generate. If None, it will be set to 2*cycle_duration
                      
    Returns
    -------
    time_series:      1D array, float. Time series of points spaced according to Smart Time Sampling algorithm
    '''
    
    assert pair_duration > minimum_delay, 'Pair duration ({0}) should be larger than minimum delay ({1})'.format(pair_duration, minimum_delay)
    delays = np.geomspace(minimum_delay, pair_duration, num=cycle_duration, endpoint=False)
    if num_points is None or num_points <= 0:
        num_points = 2*cycle_duration
    time_series = np.zeros(num_points, dtype=float)
    for i in range(num_points):
        if i % 2 == 0:
            time_series[i] = i * pair_duration / 2
        else:
            time_series[i] = time_series[i-1] + delays[((i-1) // 2) % len(delays)]
    return time_series

def ValidateAverageInterval(avg_interval, num_datapoints):
    '''
    Validates the averaged interval parameter used in AverageG2M1 and AverageCorrTimetrace functions
    
    Parameters
    ----------
    - avg_interval: None, int, couple interval=[min_idx, max_idx] or list of couples [interval1, ..., intervalN]. 
                 if None or int<=0, result will be averaged on the entire stack
                 if int>0, resolve the average on consecutive chunks of avg_interval images each
                 if interval=[min_idx, max_idx], average will be done in the specified interval
                 if [interval1, ..., intervalN], a list of intervals will be computed as above
    - num_datapoints: int. Number of datapoints
    - sharp_bound: None or bool. 
                 if False, use interval bounds to associate any reference time to the corresponding interval, 
                           but average on all time lags, including those for which ref_time + time_lag >= max_idx
                 if True, for each reference time, restrict average such that both correlated timepoints are in the specified interval
                 if None, decide automatically what to do: set it to False if avg_interval is None or int, otherwise set to True 
    
    Parameters
    ----------
    int_bounds: list of interval bounds [min_index, max_index]. In each interval, max_index > min_index
    '''
    if avg_interval is None:
        int_bounds = [[0, num_datapoints]]
    elif isinstance(avg_interval, collections.abc.Iterable):
        if len(avg_interval) > 0:
            if isinstance(avg_interval[0], collections.abc.Iterable):
                int_bounds = avg_interval
            else:
                int_bounds = [avg_interval]
            for i in range(len(int_bounds)):
                if int_bounds[i][1] <= int_bounds[i][0]:
                    logging.warn('ROIproc.ValidateAverageInterval(): bounds for interval must be strictly ascending. Empty interval generated.')
                    int_bounds[i][1] = int_bounds[i][0]
        else:
            int_bounds = [[0, num_datapoints]]
    elif avg_interval<=0:
        int_bounds = [[0, num_datapoints]]
    else:
        int_bounds = []
        for i in range(int(math.ceil(num_datapoints*1.0/avg_interval))):
            if i*avg_interval>=num_datapoints:
                logging.warn('SharedFunctions.ValidateAverageInterval inconsistency: interval {0}/{1} of size {2} exceeds series length {3}'.format(i, int(math.ceil(num_datapoints*1.0/avg_interval)), avg_interval, num_datapoints))
            else:
                int_bounds.append([i*avg_interval, min((i+1)*avg_interval, num_datapoints)])
    
    for i in range(len(int_bounds)):
        int_bounds[i][0] = min(int_bounds[i][0], num_datapoints)
        int_bounds[i][1] = min(int_bounds[i][1], num_datapoints)
        if int_bounds[i][0] >= int_bounds[i][1]:
            logging.warn('SharedFunctions.ValidateAverageInterval WARNING: inconsistent {0}-th interval {1} with {2} datapoints'.format(i, int_bounds[i], num_datapoints))
    return int_bounds

def FindLags(series, lags_index, subset_intervals=None, tolerance=1e-2, tolerance_isrelative=True, verbose=0):
    '''
    Find lags in physical units (e.g. seconds) given a list of time points (same physical units) and a list of lag indexes
    
    Parameters
    ----------
    series:           list of time points (float), not necessarily equally spaced
    lags_index:       list of lag indexes (int, >=0). WARNING: not yet compatible with negative lag indexes
    subset_intervals: None or list of intervals [min_idx, max_idx]. Eventually divide the analysis in intervals
                      If None, it will be set to the entire section
                
    Returns
    -------
    alllags:          2D list (float if series is float). One element per lagtime. Each element is the list of
                      time delays in physical units associated to that integer lagtime: alllags[i][j] = series[j+lag[i]] - series[j]
                      NOTE: len(alllags[i]) depends on i (no element is added to the list if j+lag[i] >= len(series))
    unique_laglist:   2D list. One element per interval to be analyzed. Element [i][j] is j-th lagtime of i-th analyzed subsection
    '''
    subset_intervals = ValidateAverageInterval(subset_intervals, len(series))
    alllags = []
    for lidx in range(len(lags_index)):
        if (lags_index[lidx]==0):
            alllags.append(np.zeros_like(series, dtype=float))
        elif (lags_index[lidx] < len(series)):
            alllags.append(np.subtract(series[lags_index[lidx]:], series[:-lags_index[lidx]]))
        else:
            alllags.append([])
    unique_laglist = []
    if verbose:
        logging.debug('SharedFunctions.FindLags(): processing lags in time series ({0} time points), specialized to {1} time windows: {2}'.format(len(series), len(subset_intervals), subset_intervals))
    for tavgidx in range(len(subset_intervals)):
        cur_uniquelist = np.unique([alllags[i][j] for i in range(len(lags_index)) 
                                    for j in range(subset_intervals[tavgidx][0], min(subset_intervals[tavgidx][1], len(alllags[i])))])
        cur_coarsenedlist = [cur_uniquelist[0]]
        for lidx in range(1, len(cur_uniquelist)):
            if not IsWithinTolerance(cur_uniquelist[lidx], cur_coarsenedlist[-1],
                                     tolerance=tolerance, tolerance_isrelative=tolerance_isrelative):
                cur_coarsenedlist.append(cur_uniquelist[lidx])
        unique_laglist.append(cur_coarsenedlist)
    return alllags, unique_laglist

def PixelCoordGrid(shape, extent=None, center=[0, 0], angle=0, coords='cartesian', indexing='xy'):
    '''Generates a grid of pixel coordinates
    
    Parameters
    ----------
    shape:  shape of the map [num_rows, num_cols].
    extent: extent of the mapping [x_left, x_right, y_bottom, y_top], 
            in physical units. They can be reversed (e.g. x2<x1)
            If None, it will be set to [0, shape[1], shape[0], 0]
    center: center of the coordinate system, in physical units
    angle:  Eventually rotate the coordinate system by angle, in radians
            for cartesian coordinates:
            - 0 means (xt, xn)=(x, y)
            - pi/2 means (xt, xn)=(y, -x)
              (note: +y points downward if indexing='xy')
            for polar coordinates:
            - 0 means theta=0 along +x
            - pi/2 means theta=0 along +y
              (note: this means downwards if indexing='xy')
    coords: ['cartesian'|'polar'] to return [x,y] or [r,theta] respectively
    indexing: ['xy'|'ij'], numpy.meshgrid indexing method
    
    Returns
    -------
    _grid  : couple of 2D arrays with coordinates for each pixel
             (either [x, y] or [r, theta], with theta in [-pi, pi] range)
    '''
    
    # Note: matplotlib extent is [x_left, x_right, y_bottom, y_top], 
    # meshgrid extent is [x_left, x_right, y_top, y_bottom] if indexing='xy'
    if extent is None:
        x_left, x_right, y_bottom, y_top = 0, shape[1], shape[0], 0
    else:
        x_left, x_right, y_bottom, y_top = extent
    
    # Pixel coordinates in physical units
    _grid = np.meshgrid(np.linspace(x_left-center[0], x_right-center[0], shape[1]),\
                        np.linspace(y_top-center[1], y_bottom-center[1], shape[0]), indexing=indexing)
    
    if coords=='cartesian':    
        if (angle != 0):
            # Coordinatees in the rotated reference frame
            _xt, _xn = np.multiply(_grid[0], np.cos(angle)) - np.multiply(_grid[1], np.sin(angle)),\
                       np.multiply(_grid[1], np.cos(angle)) + np.multiply(_grid[0], np.sin(angle))
            return [_xt, _xn]
        else:
            return _grid
    elif coords=='polar':
        _theta = np.arctan2(_grid[1], _grid[0])-angle
        if (angle != 0):
            _theta = np.mod(_theta+np.pi, 2*np.pi)-np.pi
        _r = np.linalg.norm(_grid, axis=0)
        return [_r, _theta]
    else:
        raise ValueError('Unknown coordinate system ' + str(coords))
        
def IntLogSpace(num_points, ppd='auto', first_point=1, autoppd_maxval=None):
    assert num_points>0,         'Number of points in log-spacing series must be positive'
    assert first_point!=0,       'First point in log-spacing series cannot be zero'
    if ppd=='auto':
        if autoppd_maxval is None:
            raise ValueError('autoppd_maxval parameter must be set to optimize ppd')
        else:
            ppd = IntLogSpace_GetBestPPD(num_points, autoppd_maxval, first_point=first_point)
    else:
        if ppd <= 0:
            raise ValueError('Number of points per decade in log-spacing series must be positive')

    res = np.zeros(num_points, dtype=int)
    res[0] = first_point
    for i in range(1, num_points):
        res[i] = max(res[i-1]+1, int(first_point * 10**(i * 1.0 / ppd)))
    return res

def IntLogSpace_GetBestPPD(num_points, max_value, first_point=1):
    assert num_points>0,   'Number of points in log-spacing series must be positive'
    assert first_point!=0, 'First point in log-spacing series cannot be zero'
    assert max_value*first_point>0,    'First and last values in log-spacing series must have the same sign'
    
    guess_value = max(1, np.floor(num_points * 1.0 / (np.log10(max_value * 1.0 / first_point))))
    while IntLogSpace(num_points, guess_value, first_point)[-1] > max_value:
        guess_value = guess_value + 1
    return guess_value
