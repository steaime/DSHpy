import numpy as np
import scipy as sp
import configparser
from scipy import signal, spatial
import os
import re
import shutil
import sys
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

def GetFolderFromCompletePath(my_string):
    return os.path.dirname(my_string)
        
def GetFilenameFromCompletePath(my_string):
    res = None
    if len(my_string)>0:
        split1 = my_string.split('\\')
        if len(split1)>0:
            split2 = split1[-1].split('/')
            if len(split2)>0:
                res = split2[-1]
    return res

def CheckFolderExists(folderPath):
    if (folderPath is None):
        return False
    else:
        return os.path.isdir(folderPath)


def CheckCreateFolder(folderPath):
    if (os.path.isdir(folderPath)):
        return True
    else:
        print("Created folder: {0}".format(folderPath))
        os.makedirs(folderPath)
        return False

def RemoveFolder(folderPath):
    try:
        shutil.rmtree(folderPath)
        return 0
    except:
        return 1

def CheckFileExists(filePath):
    try:
        return os.path.isfile(filePath)
    except:
        return False

def RenameFile(oldFileName, newFileName, forceOverwrite=False):
    if (CheckFileExists(oldFileName)):
        if (CheckFileExists(newFileName) and not forceOverwrite):
            raise ValueError('RenameFile error: new file name "' + str(newFileName) + '" already present on disk. Set forceOverwrite to overwrite it')
        else:
            os.rename(oldFileName, newFileName)
    else:
        raise IOError('RenameFile error: filename ' + oldFileName + ' not found.')

def RenameDirectory(oldName, newName):
    if (CheckFolderExists(newName)):
        raise ValueError('RenameFolder error: destination name "' + str(newName) + '" already present on disk')
    else:
        os.rename(oldName, newName)
        
def CopyFile(fileName, newName):
    if (CheckFileExists(fileName)):
        shutil.copyfile(fileName, newName)
    else:
        raise IOError('DeleteFile error: filename ' + fileName + ' not found.')

def DeleteFile(fileName):
    if (CheckFileExists(fileName)):
        os.remove(fileName)
    else:
        raise IOError('DeleteFile error: filename ' + fileName + ' not found.')

def GetFileSize(fileName):
    if (CheckFileExists(fileName)):
        return os.path.getsize(fileName)
    else:
        return -1

def PrintAndLog(strMsg, LogFile, addFirst="\n", flushBuffer=True):
    print(strMsg)
    if (LogFile != None):
        LogFile.write(addFirst + strMsg)
        if (flushBuffer):
            LogFile.flush()

def CastFloatListToInt(myList):
    for i in range(len(myList)):
        myList[i] = int(myList[i])

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
            FilenameList[i] = FolderPath + FilenameList[i]
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

def ReadConfig(config_file, config_defaults=[]):
    # Read input file for configuration
    config = configparser.ConfigParser(allow_no_value=True)
    for conf_f in config_defaults:
        print('Reading config file: ' + str(conf_f))
        config.read(conf_f)
    if (config_file is not None):
        config.read(config_file)
    return config

def ExportConfig(config, filename):
    cfgfile = open(filename,'w')
    config.write(cfgfile)
    cfgfile.close()

def ConfigGet(config, sect, key, default=None, cast_type=None, verbose=1):
    if (config.has_option(sect, key)):
        res = config[sect][key]
        if (str(res)[0] in ['[','(', '{']):
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
            return config.getboolean(sect, key)
        elif (cast_type is int):
            return config.getint(sect, key)
        elif (cast_type is float):
            if (res == 'nan'):
                return np.nan
            else:
                return config.getfloat(sect, key)
        else:
            if (cast_type is None):
                return res
            else:
                return cast_type(res)
    else:
        if (verbose>0):
            print('"' + key + '" not found in section "' + sect + '": default value ' + str(default) + ' returned.')
        return default

def ConfigSet(config, sect, key, value):
    config.set(sect, key, value)

# Boundaries: (min_val, max_val) acceptable values.
# if Boundaries != None, values outside boundaries will be discarded
def LoadIntsFromFile(FileName, Boundaries=None):
    listRes = []
    with open(FileName, "r" ) as FileData:
        for line in FileData:
            try:
                words = line.split()
                read_int = int(words[0])
                if (Boundaries == None):
                    listRes.append(read_int)
                else:
                    if (read_int > Boundaries[0] and read_int < Boundaries[1]):
                        listRes.append(read_int)
                    else:
                        print("Warning: skipped element {0} because out of boundaries".format(int(words[0])))
            except:
                pass
    return listRes

# SkipRows: number of initial header rows to be skipped (<=0 not to skip)
# Columns: list of indexes (0-based)
# if Columns != None, only colums in the list will be loaded
# Boundaries: (min_val, max_val) acceptable values.
# if Boundaries != None, values outside boundaries will be discarded
# if MaxNumRows != None, values will be loaded until the maximum row number is reached
def LoadFloatTuplesFromFile(FileName, SkipRow=-1, Columns=None, Boundaries=None, MaxNumRows=None):
    listRes = []
    line_count = 0
    with open(FileName, "r" ) as FileData:
        for line in FileData:
            line_count += 1
            if SkipRow < line_count:
                words = line.split()
                cur_tuple = []
                for word_idx in range(len(words)):
                    bln_read = True
                    if Columns != None:
                        bln_read = (word_idx in Columns)
                    if bln_read:
                        #try:
                            read_float = float(words[word_idx])
                            if (Boundaries == None):
                                cur_tuple.append(read_float)
                            else:
                                if (read_float > Boundaries[0] and read_float < Boundaries[1]):
                                    cur_tuple.append(read_float)
                                else:
                                    print("Warning: skipped element {0} because out of boundaries".format(read_float))
                        #except:
                        #    pass
                if len(cur_tuple) > 1:
                    listRes.append(cur_tuple)
                elif len(cur_tuple) == 1:
                    listRes.append(cur_tuple[0])
                if MaxNumRows != None:
                    if MaxNumRows < len(listRes):
                        break
    return listRes 


def query_yes_no(question, default="no"):
    valid = {"yes": True, "y": True, "Y": True, "ye": True,
             "no": False, "n": False, "N": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def query_string(question, accept_if_substr=''):
    while True:
        sys.stdout.write(question + ' >>> ')
        choice = input().lower()
        if accept_if_substr in choice:
            return choice
        else:
            sys.stdout.write("Answer should contain substring '" + accept_if_substr + "'.\n")
            
def LinearFit(x, y, return_residuals=False, mask=None, nonan=True, catchex=False):
    if (nonan):
        use_mask = np.logical_or(np.isnan(x), np.isnan(y))
        if (mask is None):
            use_mask = np.logical_or(mask, use_mask)            
    if (use_mask is None):
        xma, yma = x, y
    else:
        xma = np.ma.masked_array(x, mask=use_mask).compressed()
        yma = np.ma.masked_array(y, mask=use_mask).compressed()
    if (len(xma) >= 1):
        if (catchex):
            try:
                slope, residuals, _, _ = np.linalg.lstsq(np.asarray(xma)[:,np.newaxis], yma, rcond=None)
            except np.linalg.LinAlgError as err:
                if 'SVD did not converge' in str(err):
                    slope, residuals = [np.nan], [np.nan]
                else:
                    raise
        else:
            slope, residuals, _, _ = np.linalg.lstsq(np.asarray(xma)[:,np.newaxis], yma, rcond=None)
    #if (len(xma) == 1):
    #    slope, residuals = [yma[0]/xma[0]], [np.nan]
    else:
        slope, residuals = [np.nan], [np.nan]
    if (return_residuals):
        return slope[0], residuals[0]
    else:
        return slope[0]

def downsample2d(inputArray, kernelSize, normArr=None):
    if normArr is not None:
        inputArray = np.true_divide(inputArray, normArr)
    if (kernelSize==1):
        return inputArray
    else:
        average_kernel = np.ones((kernelSize,kernelSize))*np.power(1.0*kernelSize, -2)
        blurred_array = sp.signal.convolve2d(inputArray, average_kernel, mode='same', boundary='symm')
        downsampled_array = blurred_array[::kernelSize,::kernelSize]
        return downsampled_array

'''
norm: it depends on norm_type:
- norm_type=='1D' --> 1D Array with scalar normalization factors, one per z slice
- norm_type=='2D' --> 2D Array with xy-resolved normalization factors, common to all z slices
- norm_type=='3D' --> 3D Array with xyz-resolved normalization factors
'''
def downsample3d(inputArray, kernelSizeXY, kernelSizeZ, norm=None, norm_type='1D'):
    # First round the z dimension of inputArray to an integer multiple of kernelSizeZ:
    new_sizez = len(inputArray)//kernelSizeZ
    inputArray = inputArray[:new_sizez*kernelSizeZ]
    # Find the xy downsampled size 
    first_smaller = downsample2d(inputArray[0], kernelSizeXY)
    # initialize result
    smaller = np.zeros((new_sizez, first_smaller.shape[1], first_smaller.shape[0]))
    # number of frames averaged for each z (should be always kernelSizeZ)
    num_frames = np.zeros(smaller.shape[0])
    for i in range(0, len(inputArray), kernelSizeZ):
        smaller[i//kernelSizeZ] = np.zeros((first_smaller.shape[1], first_smaller.shape[0]))
        for j in range(kernelSizeZ):
            num_frames[i//kernelSizeZ]+=1
            if norm is not None:
                if norm_type=='1D':
                    new_frame = np.true_divide(downsample2d(inputArray[i+j], kernelSizeXY), norm[i+j])
                elif norm_type=='2D':
                    new_frame = downsample2d(inputArray[i+j], kernelSizeXY, norm)
                elif norm_type=='3D':
                    new_frame = downsample2d(inputArray[i+j], kernelSizeXY, norm[i+j])
                else:
                    new_frame = downsample2d(inputArray[i+j], kernelSizeXY)
            else:
                new_frame = downsample2d(inputArray[i+j], kernelSizeXY)
            smaller[i//kernelSizeZ] = np.add(smaller[i//kernelSizeZ], new_frame)
    for i in range(smaller.shape[0]):
        smaller[i] = np.true_divide(smaller[i], num_frames[i])
    return smaller
