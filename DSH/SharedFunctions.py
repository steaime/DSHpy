import os
import re
import inspect
import numpy as np
import math
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

def StrParse(my_string, cast_type=None):
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
    

def CheckIterableVariable(var, n_dim, force_length=True, cast_type=None):
    """ Checks if a variable is an iterable with given dimensions (n_dim). 
    If so, returns the variable itself. Otherwise, returns a list of 
    the variable replicated n_dim times
    if cast_type is not None, variable will be cast into cast_type before being replicated
    """
    
    assert n_dim > 0, 'Number of dimensions must be strictly positive'
    
    if isinstance(var, collections.abc.Iterable):
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





def GenerateGrid2D(shape, extent=None, center=[0, 0], angle=0, coords='cartesian', indexing='xy'):
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
             (either [x, y] or [r, theta])
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
        return [np.linalg.norm(_grid, axis=0), _theta]
    else:
        raise ValueError('Unknown coordinate system ' + str(coords))

def PolarMaskBinary(coords, shape, center=None, common_mask=None, binary_res=False):
    """Generate a list of regions of interest, labelled either in the form of binary images, 
    each one with 0s everywhere and 1 inside the region of interest,
    or by mask index, 0-based (in this case only one mask is returned).
    Pixels belonging to no mask will be labeled with -1
    
    Parameters
    ----------
    coords       : list of mask coordinates in the form [r, a, dr, da], as the one generated by PolarMaskCoords()
    shape        :  shape of the binary image (num_rows, num_cols)
    center       : center of polar coordinates. If None, it will be the center of the image
    common_mask  : eventually specify common mask to be multiplied to every mask
    binary_res   : if True, returns list of binary masks, otherwise label pixels by binary masks
                   Warning: no overlap is allowed at this time with this indexing scheme
    
    Returns
    -------
    if binary_res: a 3D array, one page per binary images (mask)
    else: a 2D array with labels, one per binary image. Binary masks can be generated by using np.where() 
    """
    if (center == None):
        center = np.multiply(coords, 0.5)          
    
    px_coord_0, px_coord_1 = GenerateGrid2D(shape, center=center, coords='polar')
    if common_mask is None:
        common_mask = np.ones_like(px_coord_0)
        
    if binary_res:
        res = np.zeros((len(coords), shape[0], shape[1]), dtype=np.dtype('b'))
    else:
        res = np.zeros(shape, dtype=int)
    for m_idx in range(len(coords)):
        cur_mask = np.multiply(common_mask, np.multiply(\
                        np.multiply(0.5 * (np.sign(np.add    (-(coords[m_idx][0]-0.5*coords[m_idx][2])+np.finfo(np.float32).eps,px_coord_0)) + 1),\
                                    0.5 * (np.sign(np.subtract((coords[m_idx][0]+0.5*coords[m_idx][2])-np.finfo(np.float32).eps,px_coord_0)) + 1)),\
                        np.multiply(0.5 * (np.sign(np.add    (-(coords[m_idx][1]-0.5*coords[m_idx][3])+np.finfo(np.float32).eps,px_coord_1)) + 1),\
                                    0.5 * (np.sign(np.subtract((coords[m_idx][1]+0.5*coords[m_idx][3])-np.finfo(np.float32).eps,px_coord_1)) + 1))))
        if binary_res:
            res[m_idx] += cur_mask
        else:
            res += cur_mask*(m_idx+1)*np.where(res>0, 0, 1)
    
    if binary_res:
        return res
    else:
        return res-1

def PolarMaskCoords(r_list, a_list):
    """Generate a list of polar masks coordinates, each mask of the form [r, a, dr, da]
    
    Parameters
    ----------
    r_list    : list of radii (in pixel units)
    a_list    : list of angles. zero angle corresponds with the +x axis direction
    
    Returns
    -------
    mSpecs: List of [r, a, dr, da]
    """
    mSpecs = []
    for r_idx in range(len(r_list) - 1):
        for a_idx in range(len(a_list) - 1):
            mSpecs.append([0.5*(r_list[r_idx] + r_list[r_idx+1]),\
                           0.5*(a_list[a_idx] + a_list[a_idx+1]),\
                           r_list[r_idx+1] - r_list[r_idx],\
                           a_list[a_idx+1] - a_list[a_idx]])
    return mSpecs

def radialAverage(image, center=None, r_range=None, weights=None, returnangles=False):
    """
    Calculate the angular profile averaged along the radial direction.

    Parameters
    ----------
    image        - The 2D image
    center       - The [x,y] pixel coordinates used as the center. The default is 
                   None, which then uses the center of the image (including 
                   fractional pixels).
    r_range      - [r_min, r_max], the range of radii to consider in the average.
                   Set r_max to -1 to 
    returnangles - if specified, return (radii_array,radial_profile)
    return_na    - if specified, return number of pixels per angle *and* radius
    binsize - size of the averaging bin, in radians.
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.
   interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.
    """
    if r_range is None:
        r_min, r_max = 0, np.linalg.norm(image.shape)
    else:
        r_min, r_max = r_range
    
    
def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None):
    """
    Calculate the azimuthally averaged radial profile.

    Parameters
    ----------
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
   interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.
    """        
    
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flat,bins)

    if weights is None:
        weights = np.ones(image.shape)
        # how many per bin (i.e., histogram)?
        # there are never any in bin 0, because the lowest index returned by digitize is 1
        #nr = np.bincount(whichbin)[1:]
        nr = np.bincount(whichbin)
    else:
        nr = np.array([weights.flat[whichbin==b].sum() for b in range(1,int(nbins+1))])
        if stddev:
            raise ValueError("Weighted standard deviation is not defined.")

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        radial_prof = np.array([image.flat[whichbin==b].std() for b in range(1,nbins+1)])
    else:
        radial_prof = np.array([(image*weights).flat[whichbin==b].sum() /\
                                weights.flat[whichbin==b].sum() for b in range(1,int(nbins+1))])

    #import pdb; pdb.set_trace()

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],\
                                radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof


def convolve2d(slab, kernel, max_missing=0.9999, sciconv_mode='reflect', verbose=True):
    '''2D convolution with missings ignored

    Parameters
    ----------
    slab:          2d array. Input array to convolve. 
                   Can have numpy.nan or masked values.
    kernel:        2d array, convolution kernel, must have sizes as odd numbers.
    max_missing:   float in (0,1), max percentage of missing in each convolution
                   window is tolerated before a missing is placed in the result.
    sciconv_mode:  boundary conditions used by scipy convolution

    Return <result>: 2d array, convolution result. Missings are represented as
                     numpy.nans if they are in <slab>, or masked if they are masked
                     in <slab>.

    '''

    from scipy.ndimage import convolve as sciconvolve

    assert np.ndim(slab)==2, "<slab> needs to be 2D."
    assert np.ndim(kernel)==2, "<kernel> needs to be 2D."
    assert kernel.shape[0]%2==1 and kernel.shape[1]%2==1, "<kernel> shape needs to be an odd number."
    assert max_missing > 0 and max_missing < 1, "<max_missing> needs to be a float in (0,1)."

    #--------------Get mask for missings--------------
    if not hasattr(slab,'mask') and np.any(np.isnan(slab))==False:
        has_missing=False
        slab2=slab.copy()

    elif not hasattr(slab,'mask') and np.any(np.isnan(slab)):
        has_missing=True
        slabmask=np.where(np.isnan(slab),1,0)
        slab2=slab.copy()
        missing_as='nan'

    elif (slab.mask.size==1 and slab.mask==False) or np.any(slab.mask)==False:
        has_missing=False
        slab2=slab.copy()

    elif not (slab.mask.size==1 and slab.mask==False) and np.any(slab.mask):
        has_missing=True
        slabmask=np.where(slab.mask,1,0)
        slab2=np.where(slabmask==1,np.nan,slab)
        missing_as='mask'

    else:
        has_missing=False
        slab2=slab.copy()

    #--------------------No missing--------------------
    if not has_missing:
        result=sciconvolve(slab2,kernel,mode=sciconv_mode,cval=0.)
    else:
        H,W=slab.shape
        hh=int((kernel.shape[0]-1)/2)  # half height
        hw=int((kernel.shape[1]-1)/2)  # half width
        min_valid=(1-max_missing)*kernel.shape[0]*kernel.shape[1]

        # dont forget to flip the kernel
        kernel_flip=kernel[::-1,::-1]

        result=sciconvolve(slab2,kernel,mode=sciconv_mode,cval=0.)
        slab2=np.where(slabmask==1,0,slab2)

        #------------------Get nan holes------------------
        miss_idx=zip(*np.where(slabmask==1))

        if missing_as=='mask':
            mask=np.zeros([H,W])

        for yii,xii in miss_idx:

            #-------Recompute at each new nan in result-------
            hole_ys=range(max(0,yii-hh),min(H,yii+hh+1))
            hole_xs=range(max(0,xii-hw),min(W,xii+hw+1))

            for hi in hole_ys:
                for hj in hole_xs:
                    hi1=max(0,hi-hh)
                    hi2=min(H,hi+hh+1)
                    hj1=max(0,hj-hw)
                    hj2=min(W,hj+hw+1)

                    slab_window=slab2[hi1:hi2,hj1:hj2]
                    mask_window=slabmask[hi1:hi2,hj1:hj2]
                    kernel_ij=kernel_flip[max(0,hh-hi):min(hh*2+1,hh+H-hi), 
                                     max(0,hw-hj):min(hw*2+1,hw+W-hj)]
                    kernel_ij=np.where(mask_window==1,0,kernel_ij)

                    #----Fill with missing if not enough valid data----
                    ksum=np.sum(kernel_ij)
                    if ksum<min_valid:
                        if missing_as=='nan':
                            result[hi,hj]=np.nan
                        elif missing_as=='mask':
                            result[hi,hj]=0.
                            mask[hi,hj]=True
                    else:
                        result[hi,hj]=np.sum(slab_window*kernel_ij)

        if missing_as=='mask':
            result=np.ma.array(result)
            result.mask=mask

    return result

def downsample2d(inputArray, kernelSize, normArr=None):
    '''Downsample 2d array by spatial convolution (binning on nearby pixels)

    Parameters
    ----------
    inputArray : input 2d array to be downsampled
    kernelSize : odd integer. It is 2N+1, where N is the number of
                 nearest neighbors to be incorporated in each pixel
    normArr :    2d array with same shape as inputArray (optional)
                 If not none, inputArray is first normalized by normArr
                 (useful for 3d subsampling)

    Returns
    -------
    downsampled_array: 2d array with each dimension reduced by a factor kernelSize

    '''
    
    if normArr is not None:
        inputArray = np.true_divide(inputArray, normArr)
    if (kernelSize==1):
        return inputArray
    else:
        average_kernel = np.ones((kernelSize,kernelSize))*np.power(1.0*kernelSize, -2)
        blurred_array = convolve2d(inputArray, average_kernel)
        downsampled_array = blurred_array[::kernelSize,::kernelSize]
        return downsampled_array