import os
import re
import inspect
import numpy as np
import collections

import pkg_resources
pkg_installed = {pkg.key for pkg in pkg_resources.working_set}
if 'json' in pkg_installed:
    import json
    use_json = True
else:
    import ast
    use_json = False
  
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
        if use_json:
            res = json.loads(res)
        else:
            res = ast.literal_eval(res)
    if (type(res) in [list,tuple]):
        for i in range(len(res)):
            if (type(res[i]) in [list,tuple]):
                if (cast_type is not None):
                    for j in range(len(res[i])):
                        res[i][j] = cast_type(res[i][j])
            else:
                if (cast_type is not None):
                    res[i] = cast_type(res[i])
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

def CheckCreateFolder(folderPath):
    if (os.path.isdir(folderPath)):
        return True
    else:
        os.makedirs(folderPath)
        print("Created folder: {0}".format(folderPath))
        return False
    
def IsIterable(var):
    return isinstance(var, collections.abc.Iterable)
    

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

def LoadResFile(fname, delimiter=',', comments='#', readHeader=True, isolateFirst=0):
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

def LogWrite(strOut, fLog, lock=None, flushAfter=True, silent=True, add_prefix='\n'):
    if fLog is not None:
        LockAcquire(lock)
        fLog.write(add_prefix + strOut)
        if flushAfter:
            fLog.flush()
        LockRelease(lock)
    LockPrint(strOut, lock, silent)

def MoveListElement(lst, old_idx, new_idx):
    if (new_idx < 0):
        new_idx += len(lst)+1
    lst.insert(new_idx, lst.pop(old_idx))
    return lst


