import sys
import logging
import bisect
import collections
import numpy as np
from scipy import ndimage as nd
import importlib.util

if importlib.util.find_spec('astropy') is not None:
    from astropy import convolution as astroconv
else:
    logging.warning('astropy package not found. Advanced image correlation functions will not be available')
    from scipy import signal

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
        return [np.linalg.norm(_grid, axis=0), _theta]
    else:
        raise ValueError('Unknown coordinate system ' + str(coords))

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
            coords = GenerateGrid2D(matrix.shape, coords=metric)
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
    
    if accept_range is None:
        if fit_range is None:
            accept_range = search_range
        else:
            accept_range = fit_range
    for _rangespec in [search_range, fit_range, accept_range]:
        if not isinstance(_rangespec, collections.abc.Iterable):
            _rangespec = [_rangespec] * len(ext_priorpos)
    
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

def GeneratePolarMasks(coords, shape, center=None, common_mask=None, binary_res=False):
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

def PolarMaskCoords(r_list, a_list, flatten_res=True):
    """Generate a list of polar masks coordinates, each mask of the form [r, a, dr, da]
    
    Parameters
    ----------
    r_list    : list of radii (in pixel units)
    a_list    : list of angles. zero angle corresponds with the +x axis direction
    flatten_res: if True, flatten the (r, a) dimensions into a single list.
                otherwise, return a 3D array with separate r and a axes
    
    Returns
    -------
    mSpecs: 2D or 3D array, depending on flatten_res. Last dimension is [r, a, dr, da]
    """
    mSpecs = np.empty((len(r_list) - 1, len(a_list) - 1, 4), dtype=float)
    for r_idx in range(len(r_list) - 1):
        for a_idx in range(len(a_list) - 1):
            mSpecs[r_idx, a_idx] = [0.5*(r_list[r_idx] + r_list[r_idx+1]),\
                                    0.5*(a_list[a_idx] + a_list[a_idx+1]),\
                                    r_list[r_idx+1] - r_list[r_idx],\
                                    a_list[a_idx+1] - a_list[a_idx]]
    if flatten_res:
        return mSpecs.reshape(-1, mSpecs.shape[-1])
    else:
        return mSpecs

def radialAverage(image, nbins, center=None, r_range=None, stddev=False, weights=None, returnangles=False, 
                  return_norm=False, masknans=True, interpnan=False, left=None, right=None):
    """
    Calculate the angular profile averaged along the radial direction.

    Parameters
    ----------
    image        - The 2D image
    nbins        - Number of angular slices
    center       - The [x,y] pixel coordinates used as the center. The default is 
                   None, which then uses the center of the image (including 
                   fractional pixels).
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

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.
    """
    if r_range is None:
        r_min, r_max = 0, None
    else:
        r_min, r_max = r_range
    if r_max is None:
        r_max = np.linalg.norm(image.shape)
        
    bins = np.linspace(-np.pi, np.pi, nbins+1, endpoint=True)
    r_map, a_map = GenerateGrid2D(image.shape, center=center, coords='polar')
    whichbin = np.digitize(a_map, bins)*np.logical_and(r_map>=r_min, r_map<r_max)
    bin_centers = (bins[1:]+bins[:-1])/2.0
    
    #coords = PolarMaskCoords(r_list=[r_min, r_max], a_list=a_list, flatten_res=True)
    #whichbin = GeneratePolarMasks(coords, image.shape, center=center, binary_res=False)

    if weights is None:
        weights = np.ones_like(image)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")
    if masknans and not stddev:
        weights = np.multiply(weights, ~np.isnan(image))
    norm = np.array([weights[whichbin==b].sum() for b in range(1,nbins+1)])
    
    if stddev:
        ang_prof = np.array([np.nanstd((image*weights)[whichbin==b]) for b in range(1,nbins+1)])
    else:
        ang_prof = np.array([np.nansum((image*weights)[whichbin==b]) *1.0/norm[b-1] for b in range(1,nbins+1)])

    if interpnan:
        ang_prof = np.interp(bin_centers,bin_centers[ang_prof==ang_prof],\
                             ang_prof[ang_prof==ang_prof],left=left,right=right)

    
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
    masknans - assume the presence of NaNs in the array: mask them and don't count them in
        the normalization by setting their weight to zero. Set it to False if you are
        sure that there are no NaNs to improve calculation speed
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
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")
    if masknans and not stddev:
        weights = np.multiply(weights, ~np.isnan(image))
    # normalization factor for each bin
    nr = np.array([weights.flat[whichbin==b].sum() for b in range(1,nbins+1)])

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        radial_prof = np.array([np.nanstd(image.flat[whichbin==b]) for b in range(1,nbins+1)])
    else:
        radial_prof = np.array([np.nansum((image*weights).flat[whichbin==b]) *1.0/nr[b-1] for b in range(1,nbins+1)])

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

# downsample2d: see skimage.transform.downscale_local_mean()
