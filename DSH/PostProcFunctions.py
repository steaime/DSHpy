import sys
import logging
import bisect
import collections
import numpy as np
from scipy import ndimage as nd
import importlib.util

from DSH import SharedFunctions as sf
from DSH import ROIproc as rp

if importlib.util.find_spec('astropy') is not None:
    from astropy import convolution as astroconv
else:
    logging.warning('astropy package not found. Advanced image correlation functions will not be available')
    from scipy import signal

if importlib.util.find_spec('skimage') is not None:
    import skimage.transform as sktr
    
    def DownsampleImage(image, factors):
        if factors==1:
            return image
        else:
            image = np.asarray(image)
            if type(factors) is int:
                factors = tuple([factors] * image.ndim)
            return sktr.downscale_local_mean(image, factors)
else:
    logging.warning('skimage package not found. DownsampleImage function will not be available')    

def hue_to_rgb(hue_map):
    """ Converts hue map to RGB
    
    Parameters
    ----------
    hue_map : 2d array of size [N,M] of floats in [0,1] range.
              They represent the hue coordinates of color
    
    Returns
    -------
    im_rgb : 3d array of same shape as im_hsl, except that the channels
             (again floats in [0,1] range) now represent RGB coordinates
             of a color with saturation=1 and luminescence=0.5
    """
    res_rgb = np.empty((hue_map.shape[0], hue_map.shape[1], 3))
    res_rgb[:,:,0] = np.abs(hue_map * 6.0 - 3.0) - 1.0
    res_rgb[:,:,1] = 2.0 - np.abs(hue_map * 6.0 - 2.0)
    res_rgb[:,:,2] = 2.0 - np.abs(hue_map * 6.0 - 4.0)
    return np.clip(res_rgb, 0.0, 1.0)
    

def hsl_to_rgb(im_hsl):
    """ Converts HSL image to RGB
    
    Parameters
    ----------
    im_hsl : 3d array of size [N,M,3], with three channels 
             (floats in [0,1] range) representing color coordinates in HSL space
    
    Returns
    -------
    im_rgb : 3d array of same shape as im_hsl, except that the channels
             (again floats in [0,1] range) now represent RGB coordinates
    """
    im_rgb = hue_to_rgb(im_hsl[:,:,0])
    c = (1.0 - np.abs(2.0 * im_hsl[:,:,2] - 1.0)) * im_hsl[:,:,1]
    for i in range(3):
        im_rgb[:,:,i] = (im_rgb[:,:,i] - 0.5) * c + im_hsl[:,:,2]
    return im_rgb


def TrimOutliers(data, bounds, x_arr=None, badval=np.nan):
    """ Trims outliers in a 2D array by setting them to badval
    
    Parameters 
    ----------
    data : 2D array
    lin_bounds : [lower_bound, upper_bound]
                 either bound can be one of the following:
                     None:  no bound is applied
                     float: constant value is applied
                     array of coefficients: lower bound is set to a polynomial function of x_arr.
                                            This requires x_arr to be specified
    badval : value to replace outliers
             if badval=='clip', data will be clipped instead
    """
    
    if bounds is None:
        return data
    else:
        data = np.asarray(data)
        bound_vals = [None] * 2
        for i in range(2):
            if (bounds[i] is None):
                bound_vals[i] = np.nanmin(data) if i==0 else np.nanmax(data)
            elif type(bounds[i]) is float:
                bound_vals[i] = bounds[i]
            else:
                poly = np.polyval(bounds[i], x_arr)
                if len(data.shape)>1:
                    bound_vals[i] = np.tile(poly, (data.shape[1], 1)).T
                else:
                    bound_vals[i] = poly
        if badval=='clip':
            return np.clip(data, bound_vals[0], bound_vals[1])
        else:
            return np.where(np.logical_and(data>=bound_vals[0], data<=bound_vals[1]), data, badval)

def ImageConvolve(source_image, kernel, interp_nans=True, conv_norm='auto', norm_kernel=False):
    """Computes 2D correlations taking special care of NaN values

    Parameters
    ----------
    source_image : 1,2, or 3D array
    kernel       : kernel, same number of axes as source_image. Must have odd dimensions along any axis.
    interp_nans  : bool, True to interpolate NaN values with nearest neighbors,
                         False to ignore NaN values in the convolution
    conv_norm    : {None, 'auto', ndarray}. Normalization factor to be applied to 
                   properly account for NaNs (unless interpolated) and edge effects.
                   If ndarray, the shape should match source_image.shape.
                   If 'auto', it is computer by convolving the kernel to ones (eventually masked at NaNs)
                   If None, no normalization will be computed. Edge effects are expected in this case
    norm_kernel  : If True, kernel will be normalized such that np.sum(kernel)=1 prior to convolution

    """
    if 'astropy' in sys.modules:
        nantr = 'interpolate' if interp_nans else 'fill'  
        raw_conv = astroconv.convolve(source_image, kernel, boundary='fill', fill_value=0,
                                      normalize_kernel=norm_kernel, nan_treatment=nantr)
        if conv_norm is None:
            return raw_conv
        else:
            if conv_norm=='auto':
                if interp_nans:
                    norm_arr = np.ones_like(source_image)
                else:
                    norm_arr = np.where(np.isnan(source_image), 0, 1)
                conv_norm = astroconv.convolve(norm_arr, kernel, boundary='fill', fill_value=0, normalize_kernel=norm_kernel)
            return np.divide(raw_conv, conv_norm)
    else:
        logging.warning('ImageConvolve() function best works with astropy package (not found). Using scipy instead: possible nan-related artifacts...')
        if norm_kernel:
            kernel = np.true_divide(kernel, np.sum(kernel))
        raw_conv = signal.convolve2d(source_image, kernel, mode='same', boundary='fill', fillvalue=0)
        if conv_norm is None:
            return raw_conv
        else:
            if conv_norm=='auto':
                conv_norm = signal.convolve2d(np.ones_like(source_image), kernel, mode='same', boundary='fill', fillvalue=0)
            return np.divide(raw_conv, conv_norm)

def BinaryDilation(data, invalid=None, open_range=-1, iterations=1, fill_value=np.nan):
    """
    Expand the domain of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Parameters
    ----------
    data:    numpy array of any dimension
    invalid: a binary array of same shape as 'data'. 
             data value are replaced where invalid is True
             If None (default), use: invalid  = np.isnan(data)
    open_range : range of the opening step. Not used if <=0
    iterations : The erosion step of the opening, then the 
             dilation step are each repeated iterations times
    fill_value : value to be used to fill the invalid domains

    Returns
    -------
    Return an array with open domains filled with specified value. 
    """
    if invalid is None: invalid = np.isnan(data)
    if open_range>0:
        invalid_open = nd.binary_dilation(invalid, iterations=iterations,\
                                          structure=np.ones((2*open_range+1,2*open_range+1)))
    else:
        invalid_open = nd.binary_dilation(invalid, iterations=iterations)
    return np.where(invalid_open, fill_value, data)

def SectImage(im, seg, npoints, fillval=np.nan):
    '''Slice 2D array along a segment

    Parametes 
    ---------
    im:      the 2D array
    seg:     2x2 array defining segment: [[x0,y0],[x1,y1]]
    npoints: number of points to sample along the segment
    fillval: value to use for points that fall outside im
    '''
    x, y = np.linspace(seg[0][1], seg[1][1], npoints), np.linspace(seg[0][0], seg[1][0], npoints)
    return nd.map_coordinates(im, np.vstack((x, y)), order=0, mode='constant', cval=fillval)
 

def RectReslice(im, seg, vrange=None, shape=None, fillval=np.nan):
    ''' Reslice 2D array by slicing it with a segment that is displaced perpendicular to itself

    Parameters
    ----------
    im:     the 2D array
    seg:    2x2 array defining segment: [[x0,y0],[x1,y1]]
    vrange: [vmin, vmax]: range of distances traveled by segment (with sign)
            if None, it will be set to the maximum range compatible with input shape
    shape:  [nrows, ncols]: shape of output array.
            ncols counts pixels along segment, nrows counts pixels across segments
            if None, it will be set to optimize the number of pixels actually included in the section
    fillval: value to be filled outside array boundaries

    Returns
    -------
    roi:    resliced 2D with shape equal to shape
    coords: [us, vs]: 2 lists of coordinates (in pixels) of rows and columns in roi
            us will have length ncols, measuring length along segment
            vs will have length nrows, measuring length in perpendicular direction
    '''
    seg_vec = np.subtract(seg[1], seg[0])
    seg_length = np.sqrt(np.sum(np.square(seg_vec)))
    parall_versor = np.true_divide(seg_vec, seg_length)
    normal_versor = np.asarray([parall_versor[1], -parall_versor[0]])
    if vrange is None:
        vrange = [-max(min(seg[0][0]/abs(normal_versor[0]), seg[0][1]/abs(normal_versor[1])),\
                       min(seg[1][0]/abs(normal_versor[0]), seg[1][1]/abs(normal_versor[1])))-200,\
                  max(min((im.shape[1]-seg[0][0])/abs(normal_versor[0]),\
                          (im.shape[0]-seg[0][1])/abs(normal_versor[1])),\
                      min((im.shape[1]-seg[0][0])/abs(normal_versor[0]),\
                          (im.shape[0]-seg[0][1])/abs(normal_versor[1])))+200]
    if shape is None:
        shape = [int(vrange[1]-vrange[0]), int(seg_length)]
    us = np.linspace(0, seg_length, shape[1])
    vs = np.linspace(vrange[0], vrange[1], shape[0])
    roi = np.empty(shape)
    for i in range(len(vs)):
        probe = np.empty((2,2))
        probe[0] = seg[0] + vs[i]*normal_versor
        probe[1] = probe[0] + seg_length*parall_versor
        roi[i] = SectImage(im, probe, len(us), fillval=fillval)
    return roi, [us, vs]

def ProbePolarLocs(loc, matrix, center, return_if_outside=False):
    """Probe matrix cell whose location is closest to given location

    Parameters
    ----------
    loc    : [r, theta] coordinates (each one being a float, or a N-long array)
    matrix : 2D array
    center : [x0,y0], center of coordinate system
    return_if_outside: if False, return None if loc is outside the boundaries of coords

    Returns
    -------
    val : matrix element (float, or N-long array)
    """
    loc_x = np.around(center[0]+loc[0]*np.cos(loc[1])).astype(int)
    loc_y = np.around(center[1]+loc[0]*np.sin(loc[1])).astype(int)
    pos_idx = tuple([np.clip(loc_y, 0, matrix.shape[0]-1), np.clip(loc_x, 0, matrix.shape[1]-1)])
    if return_if_outside:
        return matrix[pos_idx]
    else:
        pos_ok = np.logical_and.reduce((loc_x>=0, loc_x<matrix.shape[1], loc_y>=0, loc_y<matrix.shape[0]))
        return np.where(pos_ok, matrix[pos_idx], np.nan)
    
def ProbeLocation2D(loc, matrix, coords=None, metric='cartesian', interpolate='nearest', return_if_outside=False):
    """Probe matrix cell whose location is closest to given location

    Parameters
    ----------
    loc    : (loc[0], loc[1]), couple of coordinates (float)
    matrix : 2D array
    coords : [coord_0[i,j], coord_1[i,j]], couple of coordinate grids
             such as the ones that are generated by GenerateGrid2D
             If None, they will be generated using matrix.shape
    metric : {'cartesian', 'polar'}: metric to be used to calculate distances
    interpolate : {'nearest'}, for future developments
    return_if_outside: if False, return None if loc is outside the boundaries of coords

    Returns
    -------
    val : matrix element
    """
    if np.isnan(loc[0]) or np.isnan(loc[1]):
        return None
    else:
        if coords is None:
            coords = rp.GenerateGrid2D(matrix.shape, coords=metric)
        if not return_if_outside:
            if (loc[0]<np.min(coords[0]) or loc[0]>np.max(coords[0])) and (loc[1]<np.min(coords[1]) or loc[1]>np.max(coords[1])):
                return None
        if (metric=='cartesian'):
            dist = np.hypot(coords[0]-loc[0], coords[1]-loc[1])
        elif (metric=='polar'):
            dist = np.hypot(coords[0]*np.cos(coords[1])-loc[0]*np.cos(loc[1]),\
                            coords[0]*np.sin(coords[1])-loc[0]*np.sin(loc[1]))
        else:
            raise ValueError('Metric ' + str(metric) + ' not supported')
        min_pos = np.unravel_index(dist.argmin(), dist.shape)
        return matrix[min_pos]
    
def MaxRadius(image_shape, center):
    """ Calculate maximum radius given image shape and center

    Parameters
    ----------
    image_shape : [height, width]
    center : [center_x, center_y]

    """
    return np.max([np.linalg.norm(np.subtract(center, curcorner)) for curcorner in [[0,0], [0,image_shape[0]], [image_shape[1], 0], [image_shape[1],image_shape[0]]]])

def radialAverage(image, nbins, center, r_range=None, stddev=False, weights=None, returnangles=False, 
                  return_norm=False, masknans=True, interpnan=False, left=None, right=None):
    """
    Calculate the angular profile averaged along the radial direction.

    Parameters
    ----------
    image        - The 2D image
    nbins        - Number of angular slices
    center       - The [x,y] pixel coordinates used as the center.
    r_range      - [r_min, r_max], the range of radii to consider in the average.
                   Set r_max to None not to set any upper bound
    stddev       - if specified, return the azimuthal standard deviation instead of the average
    weights      - can do a weighted average instead of a simple average if this keyword parameter
                   is set.  weights.shape must = image.shape.
    returnangles - if specified, return (radii_array,radial_profile)
    return_norm  - if specified, return normalization factor *and* radius
    masknans     - assume the presence of NaNs in the array: mask them and don't count them in
                   the normalization by setting their weight to zero. Set it to False if you are
                   sure that there are no NaNs to improve calculation speed
    interpnan    - Interpolate over NAN values, i.e. bins where there is no data?
                   left,right - passed to interpnan; they set the extrapolated values
    """

    if r_range is None:
        r_min, r_max = 0, None
    else:
        r_min, r_max = r_range
    if r_max is None:
        r_max = MaxRadius(image.shape, center)
        
    bins = np.linspace(-np.pi, np.pi, nbins+1, endpoint=True)
    r_map, a_map = GenerateGrid2D(image.shape, center=center, coords='polar')
    whichbin = np.digitize(a_map, bins)*np.logical_and(r_map>=r_min, r_map<r_max)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    if stddev:
        ang_prof, norm = rp.ROIAverage(image, whichbin-1, weights=weights, masknans=masknans, evalFunc=SquareDistFromMean)
        ang_prof = np.sqrt(ang_prof)
    else:
        # NOTE: np.digitize produces output in the [0, nbins+1] range, 0 denoting unassigned pixels. 
        #       ROIAverage expects a mask in the [-1, nbins] range, -1 denoting unassigned pixels.
        ang_prof, norm = rp.ROIAverage(image, whichbin-1, weights=weights, masknans=masknans)

    if interpnan:
        ang_prof = np.interp(bin_centers,bin_centers[ang_prof==ang_prof], ang_prof[ang_prof==ang_prof],left=left,right=right)

    if returnangles:
        return bin_centers, ang_prof
    elif return_norm:
        return norm, bin_centers, ang_prof
    else:
        return ang_prof
    
def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=0.5, weights=None, steps=False, masknans=True, interpnan=False, left=None, right=None):
    """
    Calculate the azimuthally averaged radial profile.

    Parameters
    ----------
    image       - The 2D image
    center      - The [x,y] pixel coordinates used as the center. The default is 
                  None, which then uses the center of the image (including fractional pixels).
    stddev      - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize     - size of the averaging bin.  Can lead to strange results if
                  non-binsize factors are used to specify the center and the binsize is too large
    weights     - can do a weighted average instead of a simple average if this keyword parameter
                  is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
                  set weights and stddev.
    steps       - if specified, will return a double-length bin array and radial
                  profile so you can plot a step-form radial profile
    masknans    - assume the presence of NaNs in the array: mask them and don't count them in
                  the normalization by setting their weight to zero. Set it to False if you are
                  sure that there are no NaNs to improve calculation speed
    interpnan   - Interpolate over NAN values, i.e. bins where there is no data?
                  left,right - passed to interpnan; they set the extrapolated values
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
    whichbin = np.digitize(r,bins)
    
    if stddev:
        radial_prof, norm = rp.ROIAverage(image, whichbin-1, weights=weights, masknans=masknans, evalFunc=SquareDistFromMean)
        radial_prof = np.sqrt(radial_prof)
    else:
        radial_prof, norm = rp.ROIAverage(image, whichbin-1, weights=weights, masknans=masknans)
    
    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof], radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return norm,bin_centers,radial_prof
    else:
        return radial_prof

def SquareDistFromMean(data):
    return np.square(np.subtract(data, np.mean(data)))


def FindAzimuthalExtrema(arr, center=[0,0], search_start=[0], update_search=True, r_avg_w=2, r_avg_cut=0, search_range=np.pi/4,
                         extrema_ismin=True, fit_range=None,  accept_range=None, r_step=1, r_start=None, angbins=360, 
                         mask=None, return_quads=True, extrap_first=False, save_fname=None):
    """Finds the min and max of a 2D array along the azimuthal direction
    
    Parameters
    ----------
    arr          : 2D array with scalar quantity to be analyzed
    center       : [x0, y0], origin position, in pixels
    search_start : [theta_0, theta_1, ..., theta_n], prior guess of extrema position (in radians)
                   Values will be used for the smallest radial annulus (r=r_step) and then eventually updated.
                   Length of this list will set n_extrema
    update_search: if >0, use input prior for r<update_search (in pixels), and 
                          update search_start with last position found for r>=update_search.
                   if <=0, always use given prior for search position.
                   if accept_range is not None, update_search is also used to 'activate'
                   the finer accept range 
    r_avg_w      : calculate azimuthal profile by averaging over a small radial window.
                   r_avg_w is the std of the Gaussian window used, in pixels
    r_avg_cut    : if 0, use all available radii (with Gaussian weights)
                   if >0, trim weights to +/- r_avg_cut*r_avg_w
    search_range : the extremum will be searched by fitting a parabola to data within 
                   search_range radians from the previous position
                   Can be a list, one per extremum
    extrema_ismin: list of boolean variables, one per extrema. Only used if fit_range is not None.
                   if extrema_ismin[i] is True, the i-th extremum will be considered a local minimum
    fit_range    : the extremum will be searched by fitting a parabola to data within 
                   search_range radians from the local min or max (it needs to know what type of extrema from extrema_type)
                   Can be a list, one per extremum
    accept_range : the extremum will be accepted only if it lays within accept_range of prior guess
                   if None (default), accept_range will be set to search_range
                   Can be a list, one per extremum
    r_step       : sample step, in pixels
    r_start      : starting radius to be analyzed. If None, r_start=r_step
    angbins      : number of angular bins in evaluating the radial profile
    mask         : 2d array of weights to be assigned to each arr elements. Should be float in [0,1] range. 
                   Default (None) sets all weights to 1
    extrap_first : if True, eventual np.nan values for small radii will be set to first non-nan value
    save_fname   : if not None, save results to file
    
    Returns
    -------
    ext_pos      : [n_radii x n_extrema] 2D array with extrema positions [radians]
                   element will be np.nan if search failed
    ext_val      : [n_radii x n_extrema] 2D array with array values at the extrema
                   element will be np.nan if search failed
    res_r        : array of length n_radii with radii of annuli analyzed
    quad_id      : map of quadrant id (int). Value would increase by 1 as an extremum is encountered
                   running along an annulus counterclockwise. Returned only if return_quads==True
    """
    
    _r, _theta = GenerateGrid2D(arr.shape, extent=None, center=center, angle=0, coords='polar')
    if r_start is None:
        r_start = r_step
    n_radii = int((np.max(_r) - r_start)//r_step)+2
    n_extrema = len(search_start)
    if type(extrema_ismin) is bool:
        extrema_ismin = [extrema_ismin] * n_extrema
    
    res_r = np.linspace(r_start, r_start+r_step*(n_radii-1), n_radii, endpoint=True)
    ext_pos = np.ones((n_radii, n_extrema), dtype=float) * np.nan
    ext_val = np.ones_like(ext_pos) * np.nan
    if return_quads:
        quad_review_first = 0
        quad_id = -np.ones_like(arr, dtype=int)
    
    ext_priorpos = search_start.copy()
    last_valid_ext = np.ones_like(search_start) * np.nan
    
    if not isinstance(search_range, collections.abc.Iterable):
        search_range = [search_range] * len(ext_priorpos)
    if not isinstance(fit_range, collections.abc.Iterable):
        fit_range = [fit_range] * len(ext_priorpos)
    if not isinstance(accept_range, collections.abc.Iterable):
        accept_range = [accept_range] * len(ext_priorpos)
    for i in range(len(ext_priorpos)):
        if accept_range[i] is None:
            if fit_range[i] is None:
                accept_range[i] = search_range[i]
            else:
                accept_range[i] = fit_range[i]
    
    if mask is None:
        mask = np.ones_like(arr)
    
    for ridx in range(n_radii):
        cur_w = np.multiply(np.exp(np.divide(np.square(_r-res_r[ridx]),-2*r_avg_w**2)), mask)
        if r_avg_cut<=0:
            cur_rrange = None
        else:
            cur_rrange = [res_r[ridx]-r_avg_w*r_avg_cut, res_r[ridx]+r_avg_w*r_avg_cut]
        _ang, _angprof = radialAverage(arr, nbins=360, center=center, weights=cur_w, r_range=cur_rrange, returnangles=True)
        for i in range(n_extrema):
            cur_minidx = bisect.bisect_left(_ang, ext_priorpos[i]-search_range[i])
            cur_maxidx = bisect.bisect_right(_ang, ext_priorpos[i]+search_range[i])
            search_x, search_y = _ang[cur_minidx:cur_maxidx], _angprof[cur_minidx:cur_maxidx]
            if (ext_priorpos[i]-search_range[i] < _ang[0]):
                cur_addidx = bisect.bisect_left(_ang, ext_priorpos[i]-search_range[i]+2*np.pi)
                search_x = np.concatenate((_ang[cur_addidx:]-2*np.pi, search_x))
                search_y = np.concatenate((_angprof[cur_addidx:], search_y))
            if (ext_priorpos[i]+search_range[i] > _ang[-1]):
                cur_addidx = bisect.bisect_left(_ang, ext_priorpos[i]+search_range[i]-2*np.pi)
                search_x = np.concatenate((search_x, _ang[:cur_addidx]+2*np.pi))
                search_y = np.concatenate((search_y, _angprof[:cur_addidx]))
            filter_idx = np.isfinite(search_y)
            if np.count_nonzero(filter_idx)>3:
                search_x, search_y = search_x[filter_idx], search_y[filter_idx]
                if fit_range[i] is not None:
                    if extrema_ismin[i]:
                        fine_search_idx = np.argmin(search_y)
                    else:
                        fine_search_idx = np.argmax(search_y)
                    cur_minidx = bisect.bisect_left(search_x, search_x[fine_search_idx]-fit_range[i])
                    cur_maxidx = bisect.bisect_right(search_x, search_x[fine_search_idx]+fit_range[i])
                    search_x, search_y = search_x[cur_minidx:cur_maxidx], search_y[cur_minidx:cur_maxidx]                   
                z = np.polyfit(search_x, search_y, 2)
                cur_pos = -z[1] / (2*z[0])
                cur_pos_xy = np.add(center,[res_r[ridx]*np.cos(cur_pos), res_r[ridx]*np.sin(cur_pos)])
                if (update_search>0 and res_r[ridx]<update_search) or np.isnan(last_valid_ext[i]):
                    if fit_range[i] is None:
                        cur_accept = (np.abs(cur_pos-ext_priorpos[i])<search_range[i])
                    else:
                        cur_accept = (np.abs(cur_pos-ext_priorpos[i])<fit_range[i])
                else:
                    cur_accept = (np.abs(cur_pos-ext_priorpos[i])<accept_range[i])
                if (cur_accept and cur_pos_xy[0]>0 and cur_pos_xy[0]<arr.shape[1] and cur_pos_xy[1]>0 and cur_pos_xy[1]<arr.shape[0]):
                    ext_pos[ridx,i] = cur_pos
                    ext_val[ridx,i] = (4*z[0]*z[2]-z[1]**2)/(4*z[0])
                    if update_search>0 and res_r[ridx]>=update_search:
                        ext_priorpos[i] = cur_pos
                    last_valid_ext[i] = cur_pos

        if return_quads or extrap_first:
            if np.count_nonzero(np.isnan(last_valid_ext))>0:
                logging.debug('FindAzimuthalExtrema() -- {0}th radius has incomplete positions: {1}'.format(ridx, last_valid_ext))
                quad_review_first += 1
            elif return_quads:
                if ridx==0:
                    ann_pos = np.where(_r<=res_r[ridx])              
                else:
                    ann_pos = np.where(np.logical_and(_r>=res_r[ridx]-r_step, _r<=res_r[ridx]))
                reported_valid_ext = list(last_valid_ext)
                add_val_id = 0
                while reported_valid_ext[0]<-np.pi:
                    reported_valid_ext.append(reported_valid_ext.pop(0)+2*np.pi)
                    add_val_id += 1
                while reported_valid_ext[-1]>np.pi:
                    reported_valid_ext.insert(0, reported_valid_ext.pop(len(reported_valid_ext)-1)-2*np.pi)
                    add_val_id -= 1
                quad_id[ann_pos] = np.digitize(_theta[ann_pos], reported_valid_ext)
                quad_id[quad_id==len(search_start)]=0
                if add_val_id!=0:
                    logging.debug('FindAzimuthalExtrema() -- {0}th radius needed to report angles from {1} to {2}: index will be rotated by {3}'.format(ridx, last_valid_ext, reported_valid_ext, add_val_id))
                    quad_id[ann_pos] = np.mod(quad_id[ann_pos]+add_val_id, len(search_start))

    if return_quads or extrap_first:
        if quad_review_first>0:
            logging.debug('FindAzimuthalExtrema() -- Reviewing first {0} radii'.format(quad_review_first))
            for i in range(n_extrema):
                for ridx in range(n_radii):
                    if not np.isnan(ext_pos[ridx,i]):
                        last_valid_ext[i] = ext_pos[ridx,i]
                        logging.debug('FindAzimuthalExtrema() -- First radius with valid {0}-th position: {1} (pos = {2:.3f})'.format(i, ridx, ext_pos[ridx,i]))
                        break
            logging.debug('FindAzimuthalExtrema() -- Complete set of valid positions: {0}'.format(last_valid_ext))
            for ridx in range(quad_review_first-1, -1, -1):
                for i in range(n_extrema):
                    if not np.isnan(ext_pos[ridx,i]):
                        last_valid_ext[i] = ext_pos[ridx,i]
                        logging.debug('FindAzimuthalExtrema() -- Updated {0}th position at radius {1} to {2:.3f}'.format(i, ridx, ext_pos[ridx,i]))
                    elif extrap_first:
                        ext_pos[ridx,i] = last_valid_ext[i]
                if return_quads:
                    if ridx==0:
                        ann_pos = np.where(_r<=res_r[ridx])              
                    else:
                        ann_pos = np.where(np.logical_and(_r>=res_r[ridx]-r_step, _r<=res_r[ridx]))
                    reported_valid_ext = list(last_valid_ext)
                    add_val_id = 0
                    while reported_valid_ext[0]<-np.pi:
                        reported_valid_ext.append(reported_valid_ext.pop(0)+2*np.pi)
                        add_val_id += 1
                    while reported_valid_ext[-1]>np.pi:
                        reported_valid_ext.insert(0, reported_valid_ext.pop(len(reported_valid_ext)-1)-2*np.pi)
                        add_val_id -= 1
                    cur_quad = np.digitize(_theta[ann_pos], reported_valid_ext)
                    cur_quad[cur_quad==len(search_start)]=0
                    if add_val_id!=0:
                        logging.debug('FindAzimuthalExtrema() -- {0}th radius needed to report angles from {1} to {2}: index will be rotated by {3}'.format(ridx, last_valid_ext, reported_valid_ext, add_val_id))
                        cur_quad = np.mod(cur_quad+add_val_id, len(search_start))
                    quad_id[ann_pos] = cur_quad

    if save_fname is not None:
        res_arr = np.concatenate((np.reshape(res_r, (len(res_r), 1)), ext_pos, ext_val), axis=1)
        np.save(save_fname, res_arr)

    if return_quads:
        return ext_pos, ext_val, res_r, quad_id
    else:
        return ext_pos, ext_val, res_r

# downsample2d: see skimage.transform.downscale_local_mean()