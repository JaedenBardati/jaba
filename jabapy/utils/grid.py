from operator import pos
import math, warnings

import numpy as np
import numba

from contextlib import contextmanager
@contextmanager
def numba_threads(n_threads):
    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(n_threads)
        yield
    finally:
        numba.set_num_threads(old_threads)


def quick_image_downsize(image, factor=2):
    """Factor should be an integer."""
    if factor == 1:
        return image

    h, w = image.shape
    h_new = h // factor
    w_new = w // factor

    image_cropped = image[:h_new*factor, :w_new*factor]
    return image_cropped.reshape(h_new, factor, w_new, factor).mean(axis=(1, 3))


##########################################
################ Kernels #################
##########################################

def cubic_kernel(r, invh, d=3, w=True, wprime=False, wprime2=False, q_instead=False, dw_dq_instead=False, mask=None):
    """
    Cubic kernel normalized over a ball of dimension d, and its 1st and 2nd derivatives with respect to r.
    Pass numpy arrays to get vectorized results (optimized for large arrays).
    Here, W(r, h) = K_d * h^-d * w(q) is the kernel normalized over a ball of dimension d (units of 1/volume), and w(q) is the cubic kernel normalized over q = r/h (unitless).
    Parameters:
        - r              : the distance from the particle; must be strictly non-negative
        - invh           : the inverse of the compact support radius (2h for typical SPH smoothing length, h for MFM); must be strictly non-negative
        - d              : the number of dimensions; must be strictly integers 1, 2, or 3; defaults to 3d
        - w              : if True, computes the kernel values in new array; if set to an array, sets kernel values in-place over that array (must be same shape as r), this can be used to speed up multiple calls with the same set of points/neighbours
        - wprime         : if True, computes the 1st derivative values in new array; if set to an array, sets 1st derivative values in-place over that array (must be same shape as r)
        - wprime2        : if True, computes the 2nd derivative values in new array; if set to an array, sets 2nd derivative values in-place over that array (must be same shape as r)
        - q_instead      : if True, interprets first input as q = r/h instead of r
        - dw_dq_instead  : if True, returns the derivatives of w(q) with respect to q = r/h instead of W(r, h) with respect to r
        - mask           : optional boolean mask to apply to the kernel and derivative values (must be same shape as r)
    """
    # initialize relevant arrays
    q = r if q_instead else r * invh
    ret_w, ret_wprime, ret_wprime2 = w is True, wprime is True, wprime2 is True
    w = np.zeros_like(q, dtype=np.float64) if w is True else w if w is not False else None
    wprime = np.zeros_like(q, dtype=np.float64) if wprime is True else wprime if wprime is not False else None
    wprime2 = np.zeros_like(q, dtype=np.float64) if wprime2 is True else wprime2 if wprime2 is not False else None

    # get the window function w(q) normalized over q = 0 to q = 1.0, for q < 0.5
    cut = q < 0.5
    # if mask is not None:
    #     cut &= mask
    _qt = q[cut]
    _qt2 = _qt*_qt
    if w is not None:
        w[cut] = (8.0/3.0) - 16.0*(1.0 - _qt)*_qt2
    if wprime is not None:
        wprime[cut] = 48.0*_qt2 - 32.0*_qt
    if wprime2 is not None:
        wprime2[cut] = 96.0*_qt - 32.0
    
    # get the window function w(q) normalized over q = 0 to q = 1.0, for 0.5 < q < 1.0
    if mask is not None:
        np.copyto(cut, mask) 
    else:
        np.invert(cut, out=cut)
    cut &= q < 1.0
    _qt = 1.0 - q[cut]
    _qt2 = _qt*_qt
    if w is not None:
        w[cut] = (16.0/3.0)*_qt2*_qt
    if wprime is not None:
        wprime[cut] = -16.0*_qt2
    if wprime2 is not None:
        wprime2[cut] = 12.0*_qt

    # if desired, convert to the window function W(r,h) normalized from over radius h ball in d dimensions
    if not dw_dq_instead:
        _Kd = (3.0 / np.pi)*invh*invh*invh if d == 3 else (15.0 / (7.0 * np.pi))*invh*invh if d == 2 else 0.5*invh if d == 1 else None
        if w is not None:
            w *= _Kd
        if wprime is not None or wprime2 is not None:
            _Kd *= invh
            if wprime is not None:
                wprime *= _Kd
            if wprime2 is not None:
                wprime2 *= _Kd*invh

    # return output
    if ret_w:
        if ret_wprime:
            return (w, wprime, wprime2) if ret_wprime2 else (w, wprime)
        elif ret_wprime2:
            return (w, wprime2)
        return w
    elif ret_wprime:
        return (wprime, wprime2) if ret_wprime2 else wprime
    elif ret_wprime2:
        return wprime2
    return None



@numba.njit()
def _cubic_kernel(q): # TODO update as above
    q = 2.0 * q # cludge for gizmo implementaion instead
    if q < 1.0:
        return 1.0 - 1.5*q*q + 0.75*q*q*q
    elif q < 2.0:
        t = 2.0 - q
        return 0.25 * t*t*t
    else:
        return 0.0


##########################################
######## Interpolation onto grids ########
##########################################

### Direct binning method for any geometry of grid ###
@numba.njit(parallel=False, cache=True)
def _bin_particles_direct_serial(pos, qty, mins, maxs, dims, strides, grid):
    """
    Helper function. Requires precomputed strides and zero-copy grid (which it updates in place).
    Serial version. Much faster than np.histogram.
    """ 
    N = pos.shape[0]
    d = dims.shape[0]
    M = qty.shape[1]
    inv_dx = dims/(maxs - mins)
    for n in range(N):
        linear_index = 0
        valid = True
        for k in range(d):
            idx = int((pos[n, k] - mins[k]) * inv_dx[k])
            if idx < 0 or idx >= dims[k]:
                valid = False
                break
            linear_index += idx * strides[k]
        if valid:
            for m in range(M):
                grid[m, linear_index] += qty[n, m]

@numba.njit(parallel=True, cache=True)
def _bin_particles_direct_parallel(pos, qty, mins, maxs, dims, strides, grid, nthreads):
    """
    Helper function. Requires precomputed strides and zero-copy grid (which it updates in place).
    Parallel version, parallelizing across x-axis and M qty axis. 
    Tends to be only occasionally faster than serial version. Requires nthreads < M.
    """
    N = pos.shape[0]
    d = dims.shape[0]
    M = qty.shape[1]
    inv_dx = dims/(maxs - mins)

    for thread_idx in numba.prange(nthreads):
        # determine which x-bin and m-bin this thread is responsible for
        m = thread_idx % M
        x_idx = thread_idx // M
        total_xbins = (nthreads - m + M - 1) // M
        bin0_start = (dims[0] * x_idx) // total_xbins
        bin0_end = (dims[0] * (x_idx + 1)) // total_xbins

        # for all particles
        for n in range(N):
            # skip if not in this thread's x-bin range
            idx0 = int((pos[n, 0] - mins[0]) * inv_dx[0])
            if idx0 < bin0_start or idx0 >= bin0_end:
                continue
            
            # skip if out of bounds in any other dimension
            linear_index = idx0 * strides[0]
            valid = True
            for k in range(1, d):
                idx = int((pos[n, k] - mins[k]) * inv_dx[k])
                if idx < 0 or idx >= dims[k]:
                    valid = False
                    break
                linear_index += idx * strides[k]
            
            # if valid, add to the grid at this linear bin index
            if valid:
                grid[m, linear_index] += qty[n, m]


def bin_particles_direct(pos, *qty, mins=None, maxs=None, dims=None, ret_bins=False, nthreads=None, out=None, _npart_multithread_threshold=1e6):
    """
    General N-dimensional direct binning algorithm.
    An extremely quick O(N) way to interpolate particles onto a Cartesian grid.
    Suitable in the limit of small smoothing length / high density relative to grid size, but
    otherwise will be noisy and not very accurate in 2d or 3d. Usually good enough for small enough 1d histograms though.
    
    Parameters
    ----------
    pos      : (N, D)   array of particle positions
    qty      : (N, *M)  quantity (*optionally M quantities)
    mins     : (d,)     lower bounds (DEFAULT: mins of each dimension)
    maxs     : (d,)     upper bounds (DEFAULT: maxes of each dimension)
    dims     : (d,)     number of bins per dimension (DEFAULT: 512x512 image)
    nthreads : int      number of threads to use (DEFAULT: all available threads)
    ret_bins : bool     also return the bin edges if True (DEFAULT: False)
    out      : ndarray  optional writable float64 output grid to accumulate into, without clearing

    out must have the returned grid shape: tuple(dims) for one quantity, otherwise
    (M, *dims). Strided views are supported if their spatial axes can be flattened
    without copying. This allows multiple particle chunks to share one output grid.
    
    If D > d, the extra dimensions of pos are ignored. If D < d, an error is raised.
    """
    # handle inputs
    pos = np.ascontiguousarray(pos)
    assert len(pos.shape) == 2, 'Invalid pos format.'

    if isinstance(qty, tuple): # if qty is a tuple of arrays, stack them together
        qty = qty[0] if len(qty) == 1 else np.column_stack(qty)  # if you get an error here, double check you are passing keyword arguments (e.g. mins, maxs, dims) as keyword arguments, not positional arguments
    qty = np.ascontiguousarray(qty)
    assert len(qty.shape) == 1 or len(qty.shape) == 2, 'Invalid qty format.'
    assert pos.shape[0] == qty.shape[0], 'Not matching number of particles in pos and qty.'

    dims = np.array([512, 512], dtype=np.int64, order='C') if dims is None else np.ascontiguousarray(dims, dtype=np.int64) # default to 512x512 image
    assert len(dims.shape) == 1, 'Invalid dims format.'
    assert dims.shape[0] <= pos.shape[1], 'Grid dimensions (d) cannot exceed number of particle position dimensions (D).'

    # calculate internal variables
    d = dims.shape[0]
    M = 1 if len(qty.shape) == 1 else qty.shape[1] # number of quantities to bin
    _pos = pos[:, :d] if pos.shape[1] > d else pos # if D > d, truncate pos to the number of dimensions of the grid
    _qty = qty.reshape(-1, 1) if M == 1 else qty

    # handle other inputs
    mins = np.min(_pos, axis=0) if mins is None else np.ascontiguousarray(mins) # default to the min of the positions
    assert len(mins.shape) == 1, 'Invalid mins format.'
    assert mins.shape[0] == dims.shape[0], 'Non-matching mins and dims shape.'

    maxs = np.max(_pos, axis=0) if maxs is None else np.ascontiguousarray(maxs) # default to the max of the positions
    assert len(maxs.shape) == 1, 'Invalid maxs format.'
    assert maxs.shape[0] == dims.shape[0], 'Non-matching maxs and dims shape.'

    if nthreads is None:
        if pos.shape[0] > _npart_multithread_threshold: # if there are a lot of particles, use parallel version
            nthreads = -1
        else:
            nthreads = 1
    if nthreads == -1:
        nthreads = numba.get_num_threads()
    if nthreads != 1 and nthreads < M:
        warnings.warn(f"Number of threads ({nthreads}) is less than the number of quantities ({M}). Switching to serial version since this is not currently supported.")
        nthreads = 1

    # calculate strides for flattening
    strides = np.empty(d, dtype=np.int64, order='C')
    strides[d-1] = 1
    for i in range(d-2, -1, -1):
        strides[i] = strides[i+1] * dims[i+1]
    total_size = strides[0] * dims[0]

    # calculate the grid
    shape = tuple(dims) if M == 1 else (M,) + tuple(dims)
    if out is None:
        grid = np.zeros((M, total_size), dtype=np.float64, order='C')
    else:
        assert isinstance(out, np.ndarray) and out.shape == shape, 'out must match the returned grid shape.'
        assert out.dtype == np.float64 and out.flags.writeable, 'out must be a writable float64 array.'
        grid = out.view()
        grid.shape = (M, total_size)
    if nthreads == 1:
        _bin_particles_direct_serial(_pos, _qty, mins, maxs, dims, strides, grid)
    else:
        with numba_threads(nthreads):
            _bin_particles_direct_parallel(_pos, _qty, mins, maxs, dims, strides, grid, nthreads)
    grid = grid.reshape(shape) if out is None else out
    
    if not ret_bins:
        return grid
    
    # also return with position bins edges if requested
    bins = [np.arange(d + 1) * ((mx - mn) / d) + mn for mn, mx, d in zip(mins, maxs, dims)]
    return bins, grid



### Instead of sum above, get min/max ###
@numba.njit(parallel=False, cache=True)
def _bin_particles_maxmin_serial(pos, qty, mins, maxs, dims, strides, max_grid, min_grid):
    """
    Helper function. Requires precomputed strides and zero-copy grid (which it updates in place).
    Serial version.
    # TODO: make parallel version 
    # TODO: generalize this together with sum (above) and median, q1, q3, etc.
    """ 
    N = pos.shape[0]
    d = dims.shape[0]
    M = qty.shape[1]
    inv_dx = dims/(maxs - mins)
    for n in range(N):
        linear_index = 0
        valid = True
        for k in range(d):
            idx = int((pos[n, k] - mins[k]) * inv_dx[k])
            if idx < 0 or idx >= dims[k]:
                valid = False
                break
            linear_index += idx * strides[k]
        if valid:
            for m in range(M):
                max_grid[m, linear_index] = max(max_grid[m, linear_index], qty[n, m])
                min_grid[m, linear_index] = min(min_grid[m, linear_index], qty[n, m])

def bin_particles_maxmin(pos, *qty, mins=None, maxs=None, dims=None, ret_bins=False, nthreads=None, _npart_multithread_threshold=np.inf):
    """
    General N-dimensional direct binning algorithm, returning max/min. Copied mostly from bin_particles_direct above.
    """
    # handle inputs
    pos = np.ascontiguousarray(pos)
    assert len(pos.shape) == 2, 'Invalid pos format.'

    if isinstance(qty, tuple): # if qty is a tuple of arrays, stack them together
        qty = qty[0] if len(qty) == 1 else np.column_stack(qty)
    qty = np.ascontiguousarray(qty)
    assert len(qty.shape) == 1 or len(qty.shape) == 2, 'Invalid qty format.'
    assert pos.shape[0] == qty.shape[0], 'Not matching number of particles in pos and qty.'

    dims = np.array([512, 512], dtype=np.int64, order='C') if dims is None else np.ascontiguousarray(dims, dtype=np.int64) # default to 512x512 image
    assert len(dims.shape) == 1, 'Invalid dims format.'
    assert dims.shape[0] <= pos.shape[1], 'Grid dimensions (d) cannot exceed number of particle position dimensions (D).'

    # calculate internal variables
    d = dims.shape[0]
    M = 1 if len(qty.shape) == 1 else qty.shape[1] # number of quantities to bin
    _pos = pos[:, :d] if pos.shape[1] > d else pos # if D > d, truncate pos to the number of dimensions of the grid
    _qty = qty.reshape(-1, 1) if M == 1 else qty

    # handle other inputs
    mins = np.min(_pos, axis=0) if mins is None else np.ascontiguousarray(mins) # default to the min of the positions
    assert len(mins.shape) == 1, 'Invalid mins format.'
    assert mins.shape[0] == dims.shape[0], 'Non-matching mins and dims shape.'

    maxs = np.max(_pos, axis=0) if maxs is None else np.ascontiguousarray(maxs) # default to the max of the positions
    assert len(maxs.shape) == 1, 'Invalid maxs format.'
    assert maxs.shape[0] == dims.shape[0], 'Non-matching maxs and dims shape.'
    
    if nthreads is None:
        if pos.shape[0] > _npart_multithread_threshold: # if there are a lot of particles, use parallel version
            nthreads = -1
        else:
            nthreads = 1
    if nthreads == -1:
        nthreads = numba.get_num_threads()
    if nthreads != 1 and nthreads < M:
        warnings.warn(f"Number of threads ({nthreads}) is less than the number of quantities ({M}). Switching to serial version since this is not currently supported.")
        nthreads = 1

    # calculate strides for flattening
    strides = np.empty(d, dtype=np.int64, order='C')
    strides[d-1] = 1
    for i in range(d-2, -1, -1):
        strides[i] = strides[i+1] * dims[i+1]
    total_size = strides[0] * dims[0]

    # calculate the grid
    max_grid = np.zeros((M, total_size), dtype=np.float64, order='C')
    min_grid = np.zeros((M, total_size), dtype=np.float64, order='C')
    if nthreads == 1:
        _bin_particles_maxmin_serial(_pos, _qty, mins, maxs, dims, strides, max_grid, min_grid)
    else:
        with numba_threads(nthreads):
            raise NotImplementedError("Parallel version of bin_particles_maxmin not yet implemented.")
            #_bin_particles_maxmin_parallel(_pos, _qty, mins, maxs, dims, strides, min_grid, max_grid, nthreads)
    max_grid = max_grid.reshape(tuple(dims)) if M == 1 else max_grid.reshape((M,) + tuple(dims))
    min_grid = min_grid.reshape(tuple(dims)) if M == 1 else min_grid.reshape((M,) + tuple(dims))

    if not ret_bins:
        return max_grid, min_grid

    # also return with position bins edges if requested
    bins = [np.arange(d + 1) * ((mx - mn) / d) + mn for mn, mx, d in zip(mins, maxs, dims)]
    return bins, max_grid, min_grid


def bin_particles_percentiles(pos, qty, weight=None, mins=None, maxs=None, dims=None, ret_bins=False, p_volume=75.0):
    """
    Compute per-bin percentiles (q1, median, q3) using a numba-accelerated
    quickselect-like approach. This function mirrors the API/shape of the
    other binning helpers and returns either (q1, median, q3) or
    (bins, q1, median, q3) when `ret_bins=True`.

    Implementation notes:
      - First pass computes per-bin counts to allocate a flat storage array.
      - Second pass fills the flat array with per-bin values.
      - Then per-bin selection is done with an in-place nth-element (quickselect)
        implemented in numba for speed.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    qty = np.ascontiguousarray(qty, dtype=np.float64)
    if weight is None:
        weight = np.ones(qty.shape[0], dtype=np.float64)
    else:
        weight = np.ascontiguousarray(weight, dtype=np.float64)

    if np.any(weight < 0):
        raise ValueError("Weights must be non-negative.")

    if dims is None:
        dims = np.array([512, 512], dtype=np.int64)
    else:
        dims = np.asarray(dims, dtype=np.int64)
    d = dims.shape[0]

    if mins is None:
        mins = np.min(pos[:, :d], axis=0)
    else:
        mins = np.asarray(mins, dtype=np.float64)
    if maxs is None:
        maxs = np.max(pos[:, :d], axis=0)
    else:
        maxs = np.asarray(maxs, dtype=np.float64)

    # compute strides and total size
    strides = np.empty(d, dtype=np.int64)
    strides[d-1] = 1
    for i in range(d-2, -1, -1):
        strides[i] = strides[i+1] * dims[i+1]
    total_size = int(strides[0] * dims[0])

    # call into numba core
    q1_flat, median_flat, q3_flat = _percentiles_core(pos, qty, weight, mins, maxs, dims, strides, p_volume)

    q1 = q1_flat.reshape(tuple(dims))
    median = median_flat.reshape(tuple(dims))
    q3 = q3_flat.reshape(tuple(dims))

    if not ret_bins:
        return q1, median, q3

    bins = [np.arange(int(d) + 1) * ((mx - mn) / int(d)) + mn for mn, mx, d in zip(mins, maxs, dims)]
    return bins, q1, median, q3


@numba.njit
def _nth_element(a, w, left, right, n):
    """In-place quickselect (nth_element) that returns the n-th smallest element index relative to `a`."""
    while True:
        if left == right:
            return a[left]
        pivot = a[(left + right) // 2]
        i = left
        j = right
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                tmp = a[i]
                a[i] = a[j]
                a[j] = tmp
                tmp = w[i]
                w[i] = w[j]
                w[j] = tmp
                i += 1
                j -= 1
    
        left_weight = 0.0
        for k in range(left, j + 1):
            left_weight += w[k]

        middle_weight = 0.0
        for k in range(j + 1, i):
            middle_weight += w[k]

        if n <= left_weight:
            right = j
        elif n <= left_weight + middle_weight:
            return pivot
        else:
            n -= left_weight + middle_weight
            left = i
    
            


@numba.njit(parallel=True)
def _percentiles_core(pos, qty, weight, mins, maxs, dims, strides, p_volume):
    N = pos.shape[0]
    d = dims.shape[0]
    inv_dx = dims.astype(np.float64) / (maxs - mins)

    total_size = int(strides[0] * dims[0])
    counts = np.zeros(total_size, dtype=np.int64)

    # pass 1: counts
    for n in range(N):
        lin = 0
        valid = True
        for k in range(d):
            idx = int((pos[n, k] - mins[k]) * inv_dx[k])
            if idx < 0 or idx >= dims[k]:
                valid = False
                break
            lin += idx * strides[k]
        if valid:
            counts[lin] += 1

    # offsets
    offsets = np.empty(total_size + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(total_size):
        offsets[i+1] = offsets[i] + counts[i]
    total_vals = offsets[total_size]

    # pass 2: fill flat array
    flat_vals = np.empty(total_vals, dtype=np.float64)
    flat_weights = np.empty(total_vals, dtype=np.float64)
    cur = np.zeros(total_size, dtype=np.int64)
    for n in range(N):
        lin = 0
        valid = True
        for k in range(d):
            idx = int((pos[n, k] - mins[k]) * inv_dx[k])
            if idx < 0 or idx >= dims[k]:
                valid = False
                break
            lin += idx * strides[k]
        if valid:
            idx_flat = offsets[lin] + cur[lin]
            flat_vals[idx_flat] = qty[n]
            flat_weights[idx_flat] = weight[n]
            cur[lin] += 1

    # prepare outputs
    q1 = np.empty(total_size, dtype=np.float64)
    med = np.empty(total_size, dtype=np.float64)
    q3 = np.empty(total_size, dtype=np.float64)

    p = float(p_volume)
    frac_low = (1.0 - p/100.0) / 2.0
    frac_med = 0.5
    frac_high = 1.0 - frac_low

    # per-bin selection (parallel across bins)
    for i in numba.prange(total_size):
        m = counts[i]
        if m == 0:
            q1[i] = np.nan
            med[i] = np.nan
            q3[i] = np.nan
            continue
        start = offsets[i]
        end = offsets[i+1] - 1

        total_weight = 0.0
        for k in range(start, end + 1):
            total_weight += flat_weights[k]

        target_low = frac_low * total_weight
        target_med = frac_med * total_weight
        target_high = frac_high * total_weight

        q1[i] = _nth_element(flat_vals, flat_weights, start, end, target_low)
        med[i] = _nth_element(flat_vals, flat_weights, start, end, target_med)
        q3[i] = _nth_element(flat_vals, flat_weights, start, end, target_high)

    return q1, med, q3




#### Scatter-splat method for SPH with Cartesian grids ####
# -> numba version of cython implementation in pynbody (tends to be slower than cython version)
# -> TODO: implement this in cython myself to avoid importing all of pynbody
# @numba.njit(parallel=True)  # OLD VERSION TO COMPARE
# def _bin_particles_scatter(pos, qty, h, mins, maxs, dims, strides, total_size):
#     N, D = pos.shape
#     d = dims.shape[0]

#     dx = (maxs - mins) / dims
#     inv_dx = 1/dx
    
#     nthreads = numba.get_num_threads()
#     thread_grids = np.zeros((nthreads, total_size))

#     for n in numba.prange(N):
#         tid = numba.get_thread_id()
#         grid = thread_grids[tid]

#         x = pos[n, 0]
#         y = pos[n, 1]
#         x_idx = int((x - mins[0]) * inv_dx[0])
#         y_idx = int((y - mins[1]) * inv_dx[1])

#         if x_idx < 0 or x_idx >= dims[0] or y_idx < 0 or y_idx >= dims[1]:
#             continue

#         h_i = h[n]

#         # support radius = 2h (SPH kernel)
#         r_support = 2.0 * h_i

#         # compute bounding box
#         x_min = mins[0]
#         y_min = mins[1]

#         x_start = int((pos[n,0] - r_support - x_min) * inv_dx[0])
#         x_stop  = int((pos[n,0] + r_support - x_min) * inv_dx[0])
#         y_start = int((pos[n,1] - r_support - y_min) * inv_dx[1])
#         y_stop  = int((pos[n,1] + r_support - y_min) * inv_dx[1])

#         # clamp
#         if x_start < 0: x_start = 0
#         if y_start < 0: y_start = 0
#         if x_stop > dims[0]: x_stop = dims[0]
#         if y_stop > dims[1]: y_stop = dims[1]

#         norm = 10.0 / (7.0 * np.pi * h_i * h_i)  # 2D cubic spline normalization

#         if (2.0 * h_i * inv_dx[0] < 0.5) and (2.0 * h_i * inv_dx[1] < 0.5):
#             i = x_idx
#             j = y_idx

#             x_center = mins[0] + (i + 0.5) * dx[0]
#             y_center = mins[1] + (j + 0.5) * dx[1]

#             dxp = x - x_center
#             dyp = y - y_center
#             r = np.sqrt(dxp * dxp + dyp * dyp)
#             q = r / h_i

#             idx_lin = i * strides[0] + j * strides[1]
#             if q < 2.0:
#                 grid[idx_lin] += qty[n] * _cubic_kernel(q) * norm
#             continue

#         for i in range(x_start, x_stop):
#             x_center = mins[0] + (i + 0.5)*dx[0]

#             for j in range(y_start, y_stop):
#                 y_center = mins[1] + (j + 0.5)*dx[1]

#                 # distance
#                 dxp = x - x_center
#                 dyp = y - y_center
#                 r = np.sqrt(dxp*dxp + dyp*dyp)

#                 q = r / h_i

#                 if q < 2.0:
#                     w = _cubic_kernel(q) * norm

#                     idx_lin = i*strides[0] + j*strides[1]
#                     grid[idx_lin] += qty[n] * w

#     grid = np.zeros(total_size)
#     for t in range(nthreads):
#         for i in range(total_size):
#             grid[i] += thread_grids[t, i]

#     return grid


# def bin_particles_scatter(pos, qty, h, mins=None, maxs=None, dims=None):
#     pos = np.asarray(pos)
#     qty = np.asarray(qty)
#     h = np.asarray(h)

#     if dims is None:
#         dims = np.array([512, 512], dtype=np.int64)
#     else:
#         dims = np.asarray(dims, dtype=np.int64)

#     d = dims.shape[0]

#     if mins is None:
#         mins = np.min(pos[:, :d], axis=0)
#     else:
#         mins = np.asarray(mins)

#     if maxs is None:
#         maxs = np.max(pos[:, :d], axis=0)
#     else:
#         maxs = np.asarray(maxs)

#     # strides for flattening
#     strides = np.empty(d, dtype=np.int64)
#     strides[d-1] = 1
#     for i in range(d-2, -1, -1):
#         strides[i] = strides[i+1] * dims[i+1]
#     total_size = strides[0] * dims[0]

#     return _bin_particles_scatter(pos, qty, h, mins, maxs, dims, strides, total_size).reshape(tuple(dims))

# second version
# @numba.njit(parallel=True, cache=True, fastmath=True)
# def _bin_particles_scatter_parallel(pos, qty, h, mins, maxs, dims, strides, total_size):
#     N, D = pos.shape
    
#     dx = (maxs - mins) / dims
#     inv_dx = 1.0 / dx
    
#     # Internal allocation fused cleanly across threads (Loop #3 in diagnostics)
#     grid = np.zeros(total_size, dtype=np.float64)
    
#     for n in numba.prange(N):
#         x = pos[n, 0]
#         y = pos[n, 1]
        
#         x_idx = int((x - mins[0]) * inv_dx[0])
#         y_idx = int((y - mins[1]) * inv_dx[1])
        
#         # Out of bounds check
#         if x_idx < 0 or x_idx >= dims[0] or y_idx < 0 or y_idx >= dims[1]:
#             continue
            
#         h_i = h[n]
#         r_support = 2.0 * h_i
        
#         # Exact bounding box math matching your reference code
#         x_start = int((x - r_support - mins[0]) * inv_dx[0])
#         x_stop = int((x + r_support - mins[0]) * inv_dx[0])
#         y_start = int((y - r_support - mins[1]) * inv_dx[1])
#         y_stop = int((y + r_support - mins[1]) * inv_dx[1])
        
#         # Boundary clamping 
#         if x_start < 0: x_start = 0
#         if y_start < 0: y_start = 0
#         if x_stop > dims[0]: x_stop = dims[0]
#         if y_stop > dims[1]: y_stop = dims[1]
        
#         # Dynamic Normalization Selection based on your dataset dimensionality
#         if D == 1:
#             norm = 2.0 / (3.0 * h_i)
#         elif D == 2:
#             norm = 10.0 / (7.0 * np.pi * h_i * h_i)
#         elif D == 3:
#             norm = 1.0 / (np.pi * h_i * h_i * h_i)
#         else:
#             norm = 1.0
        
#         # --- SUB-PIXEL ACCURACY BRANCH ---
#         if (2.0 * h_i * inv_dx[0] < 0.5) and (2.0 * h_i * inv_dx[1] < 0.5):
#             x_center = mins[0] + (x_idx + 0.5) * dx[0]
#             y_center = mins[1] + (y_idx + 0.5) * dx[1]
            
#             dxp = x - x_center
#             dyp = y - y_center
#             r = np.sqrt(dxp * dxp + dyp * dyp)
#             q = r / h_i
            
#             if q < 2.0:
#                 idx_lin = x_idx * strides[0] + y_idx * strides[1]
#                 grid[idx_lin] += qty[n] * _cubic_kernel(q) * norm
#             continue
            
#         # --- STANDARD SCATTER BOUNDING BOX LOOP ---
#         for i in range(x_start, x_stop):
#             x_center = mins[0] + (i + 0.5) * dx[0]
#             for j in range(y_start, y_stop):
#                 y_center = mins[1] + (j + 0.5) * dx[1]
                
#                 dxp = x - x_center
#                 dyp = y - y_center
#                 r = np.sqrt(dxp * dxp + dyp * dyp)
#                 q = r / h_i
                
#                 if q < 2.0:
#                     w = _cubic_kernel(q) * norm
#                     idx_lin = i * strides[0] + j * strides[1]
#                     grid[idx_lin] += qty[n] * w
                    
#     return grid

KERNEL_TABLE_SIZE = 4096

kernel_table = np.empty(KERNEL_TABLE_SIZE, dtype=np.float64)

# q² ranges from 0 to 4
kernel_inv_dq2 = (KERNEL_TABLE_SIZE - 1) / 4.0

for i in range(KERNEL_TABLE_SIZE):
    q2 = 4.0 * i / (KERNEL_TABLE_SIZE - 1)
    kernel_table[i] = _cubic_kernel(np.sqrt(q2))


@numba.njit(parallel=True, cache=True, fastmath=True)
def _bin_particles_scatter_parallel(
    pos,
    qty,
    h,
    mins,
    maxs,
    dims,
    strides,
    kernel_table,
    kernel_inv_dq2,
    x_centers,
    y_centers,
    total_size,
):
    N = pos.shape[0]
    xmin, ymin = mins
    xmax, ymax = maxs

    dx0 = (xmax - xmin) / dims[0]
    dx1 = (ymax - ymin) / dims[1]
    inv_dx0 = 1.0 / dx0
    inv_dx1 = 1.0 / dx1

    stride0 = strides[0]
    stride1 = strides[1]

    nx = dims[0]
    ny = dims[1]

    norm_const = 10.0 / (7.0 * np.pi)

    grid = np.zeros(total_size, dtype=np.float64)

    for n in numba.prange(N):

        x = pos[n, 0]
        y = pos[n, 1]

        q_i = qty[n]

        h_i = h[n]
        inv_h = 1.0 / h_i
        inv_h2 = inv_h * inv_h

        norm = norm_const * inv_h2

        support = 2.0 * h_i
        support2 = support * support

        x_pix = (x - xmin) * inv_dx0
        y_pix = (y - ymin) * inv_dx1

        x_idx = int(x_pix)
        y_idx = int(y_pix)

        if x_idx < 0 or x_idx >= nx or y_idx < 0 or y_idx >= ny:
            continue

        r_pix_x = support * inv_dx0
        r_pix_y = support * inv_dx1

        x_start = int(x_pix - r_pix_x)
        x_stop = int(x_pix + r_pix_x) + 1
        y_start = int(y_pix - r_pix_y)
        y_stop = int(y_pix + r_pix_y) + 1

        if x_start < 0:
            x_start = 0
        if y_start < 0:
            y_start = 0
        if x_stop > nx:
            x_stop = nx
        if y_stop > ny:
            y_stop = ny

        # Small-particle shortcut
        if r_pix_x < 0.5 and r_pix_y < 0.5:

            dxp = x - x_centers[x_idx]
            dyp = y - y_centers[y_idx]

            r2 = dxp * dxp + dyp * dyp

            if r2 < support2:
                q = np.sqrt(r2) * inv_h
                idx = x_idx * stride0 + y_idx * stride1
                grid[idx] += q_i * (_cubic_kernel(q) * norm)

            continue

        for i in range(x_start, x_stop):

            dxp = x - x_centers[i]
            dxp2 = dxp * dxp

            base = i * stride0

            for j in range(y_start, y_stop):

                dyp = y - y_centers[j]

                r2 = dxp2 + dyp * dyp

                if r2 >= support2:
                    continue

                # q = np.sqrt(r2) * inv_h

                # grid[base + j * stride1] += q_i * (_cubic_kernel(q) * norm)
                
                idx = int(r2 * inv_h2 * kernel_inv_dq2)
                if idx >= kernel_table.shape[0]:
                    idx = kernel_table.shape[0] - 1

                grid[base + j * stride1] += q_i * (kernel_table[idx] * norm)

    return grid


def bin_particles_scatter(pos, qty, h, mins=None, maxs=None, dims=None, nthreads=-1):
    """
    General N-dimensional scatter-splat binning algorithm.
    Spreads particle quantities over their local smoothing lengths onto a Cartesian grid. 

    Parameters
    ----------
    pos      : (N, D) array of particle positions
    qty      : (N,) quantity
    h        : (N,) or (N, D) smoothing length / search radius per particle
    mins     : (d,) lower bounds (DEFAULT: mins of each dimension)
    maxs     : (d,) upper bounds (DEFAULT: maxes of each dimension)
    dims     : (d,) number of bins per dimension (DEFAULT: 512x512 image)
    nthreads : number of threads to use (-1 for all available threads)
    """
    pos = np.asarray(pos, dtype=np.float64)
    qty = np.asarray(qty, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    
    if dims is None:
        dims = np.array([512, 512], dtype=np.int64)
    else:
        dims = np.asarray(dims, dtype=np.int64)
    d = dims.shape[0]
    
    if mins is None:
        mins = np.min(pos[:, :d], axis=0).astype(np.float64)
    else:
        mins = np.asarray(mins, dtype=np.float64)
        
    if maxs is None:
        maxs = np.max(pos[:, :d], axis=0).astype(np.float64)
    else:
        maxs = np.asarray(maxs, dtype=np.float64)
        
    if nthreads == -1:
        nthreads = numba.config.NUMBA_NUM_THREADS
        
    # Structural validations
    assert len(pos.shape) == 2, 'Invalid pos format.'
    assert len(qty.shape) == 1, 'Invalid qty format.'
    assert len(h.shape) == 1, 'Invalid h format.'
    assert len(mins.shape) == 1, 'Invalid mins format.'
    assert len(maxs.shape) == 1, 'Invalid maxs format.'
    assert len(dims.shape) == 1, 'Invalid dims format.'
    assert pos.shape[0] == qty.shape[0], 'Not matching number of particles in pos and qty.'
    assert mins.shape[0] == dims.shape[0], 'Not matching mins and dims shape.'
    assert maxs.shape[0] == dims.shape[0], 'Not matching maxs and dims shape.'
    assert dims.shape[0] <= pos.shape[1], 'Grid dimensions cannot exceed number of particle position dimensions.'

    # Calculate strides for flattening the output geometry
    strides = np.empty(d, dtype=np.int64)
    strides[d-1] = 1
    for i in range(d-2, -1, -1):
        strides[i] = strides[i+1] * dims[i+1]
    total_size = strides[0] * dims[0]
    
    if nthreads == 1:
        grid = np.zeros(total_size, dtype=np.float64)
        _bin_particles_scatter_serial(pos, qty, h, mins, maxs, dims, strides, grid)
    else:
        with numba_threads(nthreads):
            x_centers = mins[0] + (np.arange(dims[0]) + 0.5) * (maxs[0] - mins[0]) / dims[0]
            y_centers = mins[1] + (np.arange(dims[1]) + 0.5) * (maxs[1] - mins[1]) / dims[1]
            grid = _bin_particles_scatter_parallel(pos, qty, h, mins, maxs, dims, strides, kernel_table, kernel_inv_dq2,  x_centers, y_centers, total_size)
            
    return grid.reshape(tuple(dims))






#### Scanline rasterization method for Voronoi-like grids ####
@numba.njit(parallel=True, fastmath=True)
def _bin_particles_voronoi_2d(pos, qty, mins, maxs, dims, tile_size, search_radius):
    n_part = pos.shape[0]

    nx = dims[0]
    ny = dims[1]

    dx = (maxs[0] - mins[0]) / nx
    dy = (maxs[1] - mins[1]) / ny
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    nbx = (nx + tile_size - 1) // tile_size
    nby = (ny + tile_size - 1) // tile_size
    nbuckets = nbx * nby

    bucket_ids = np.empty(n_part, dtype=np.int64)
    counts = np.zeros(nbuckets, dtype=np.int64)

    x_min = mins[0]
    y_min = mins[1]

    # Bucket particles into tiles to keep candidate lists small.
    for n in range(n_part):
        x = pos[n, 0]
        y = pos[n, 1]

        ix = int((x - x_min) * inv_dx)
        iy = int((y - y_min) * inv_dy)

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            bucket_ids[n] = -1
            continue

        bx = ix // tile_size
        by = iy // tile_size
        bucket = bx + nbx * by

        bucket_ids[n] = bucket
        counts[bucket] += 1

    offsets = np.empty(nbuckets + 1, dtype=np.int64)
    offsets[0] = 0
    for b in range(nbuckets):
        offsets[b + 1] = offsets[b] + counts[b]

    bucket_particles = np.empty(offsets[nbuckets], dtype=np.int64)
    cursor = offsets[:-1].copy()

    for n in range(n_part):
        bucket = bucket_ids[n]
        if bucket >= 0:
            slot = cursor[bucket]
            bucket_particles[slot] = n
            cursor[bucket] = slot + 1

    result = np.zeros((nx, ny))

    # Each tile only searches nearby buckets to approximate a full Voronoi sweep.
    ntiles = nbx * nby
    for tile in numba.prange(ntiles):
        bx = tile % nbx
        by = tile // nbx

        x0 = bx * tile_size
        y0 = by * tile_size
        x1 = x0 + tile_size
        y1 = y0 + tile_size

        if x1 > nx:
            x1 = nx
        if y1 > ny:
            y1 = ny

        bx0 = bx - search_radius
        bx1 = bx + search_radius
        by0 = by - search_radius
        by1 = by + search_radius

        if bx0 < 0:
            bx0 = 0
        if by0 < 0:
            by0 = 0
        if bx1 >= nbx:
            bx1 = nbx - 1
        if by1 >= nby:
            by1 = nby - 1

        cand_count = 0
        for jbx in range(bx0, bx1 + 1):
            for jby in range(by0, by1 + 1):
                cand_count += counts[jbx + nbx * jby]

        if cand_count == 0:
            continue

        candidates = np.empty(cand_count, dtype=np.int64)
        p = 0
        for jbx in range(bx0, bx1 + 1):
            for jby in range(by0, by1 + 1):
                bucket = jbx + nbx * jby
                start = offsets[bucket]
                stop = offsets[bucket + 1]
                for s in range(start, stop):
                    candidates[p] = bucket_particles[s]
                    p += 1

        # If only one candidate exists, fill the tile directly.
        if cand_count == 1:
            n = candidates[0]
            val = qty[n]
            for i in range(x0, x1):
                for j in range(y0, y1):
                    result[i, j] = val
            continue

        for i in range(x0, x1):
            x_center = x_min + (i + 0.5) * dx
            for j in range(y0, y1):
                y_center = y_min + (j + 0.5) * dy

                best_d2 = 1.0e308
                best_qty = 0.0

                for c in range(cand_count):
                    n = candidates[c]
                    dxp = pos[n, 0] - x_center
                    dyp = pos[n, 1] - y_center
                    d2 = dxp * dxp + dyp * dyp

                    if d2 < best_d2:
                        best_d2 = d2
                        best_qty = qty[n]

                result[i, j] = best_qty

    return result


@numba.njit(parallel=True, fastmath=True)
def _bin_particles_voronoi_nd(pos, qty, mins, maxs, dims, tile_size, search_radius):
    n_part = pos.shape[0]
    d = dims.shape[0]

    dx = (maxs - mins) / dims
    inv_dx = 1.0 / dx

    # Build a regular tile grid for bucketed neighbor searches.
    nb = np.empty(d, dtype=np.int64)
    for k in range(d):
        nb[k] = (dims[k] + tile_size - 1) // tile_size

    bstrides = np.empty(d, dtype=np.int64)
    bstrides[0] = 1
    for k in range(1, d):
        bstrides[k] = bstrides[k - 1] * nb[k - 1]
    nbuckets = bstrides[d - 1] * nb[d - 1]

    bucket_ids = np.empty(n_part, dtype=np.int64)
    counts = np.zeros(nbuckets, dtype=np.int64)

    # Assign each particle to a bucket (or mark invalid if outside bounds).
    for n in range(n_part):
        valid = True
        bucket = 0
        for k in range(d):
            idx = int((pos[n, k] - mins[k]) * inv_dx[k])
            if idx < 0 or idx >= dims[k]:
                valid = False
                break
            b = idx // tile_size
            bucket += b * bstrides[k]
        if valid:
            bucket_ids[n] = bucket
            counts[bucket] += 1
        else:
            bucket_ids[n] = -1

    offsets = np.empty(nbuckets + 1, dtype=np.int64)
    offsets[0] = 0
    for b in range(nbuckets):
        offsets[b + 1] = offsets[b] + counts[b]

    bucket_particles = np.empty(offsets[nbuckets], dtype=np.int64)
    cursor = offsets[:-1].copy()
    for n in range(n_part):
        bucket = bucket_ids[n]
        if bucket >= 0:
            slot = cursor[bucket]
            bucket_particles[slot] = n
            cursor[bucket] = slot + 1

    # Flat result with strides so we can keep numba-friendly indexing.
    gstrides = np.empty(d, dtype=np.int64)
    gstrides[0] = 1
    for k in range(1, d):
        gstrides[k] = gstrides[k - 1] * dims[k - 1]
    total_size = gstrides[d - 1] * dims[d - 1]
    result = np.zeros(total_size)

    # Iterate tiles in parallel; each tile searches nearby buckets only.
    for tile in numba.prange(nbuckets): 
        tile_coord = np.empty(d, dtype=np.int64)
        rem = tile
        for k in range(d - 1, -1, -1):
            stride = bstrides[k]
            tile_coord[k] = rem // stride
            rem = rem - tile_coord[k] * stride

        cell_start = np.empty(d, dtype=np.int64)
        cell_stop = np.empty(d, dtype=np.int64)
        for k in range(d):
            start = tile_coord[k] * tile_size
            stop = start + tile_size
            if stop > dims[k]:
                stop = dims[k]
            cell_start[k] = start
            cell_stop[k] = stop

        bmin = np.empty(d, dtype=np.int64)
        bmax = np.empty(d, dtype=np.int64)
        for k in range(d):
            lo = tile_coord[k] - search_radius
            hi = tile_coord[k] + search_radius
            if lo < 0:
                lo = 0
            if hi >= nb[k]:
                hi = nb[k] - 1
            bmin[k] = lo
            bmax[k] = hi

        # Count and materialize candidates from neighboring buckets.
        cand_count = 0
        cur = bmin.copy()
        while True:
            b_lin = 0
            for k in range(d):
                b_lin += cur[k] * bstrides[k]
            cand_count += counts[b_lin]

            k = 0
            while k < d:
                if cur[k] < bmax[k]:
                    cur[k] += 1
                    for j in range(k):
                        cur[j] = bmin[j]
                    break
                k += 1
            if k == d:
                break

        if cand_count == 0:
            continue

        candidates = np.empty(cand_count, dtype=np.int64)
        p = 0
        cur = bmin.copy()
        while True:
            b_lin = 0
            for k in range(d):
                b_lin += cur[k] * bstrides[k]
            start = offsets[b_lin]
            stop = offsets[b_lin + 1]
            for s in range(start, stop):
                candidates[p] = bucket_particles[s]
                p += 1

            k = 0
            while k < d:
                if cur[k] < bmax[k]:
                    cur[k] += 1
                    for j in range(k):
                        cur[j] = bmin[j]
                    break
                k += 1
            if k == d:
                break

        # If only one candidate exists, fill the tile directly.
        if cand_count == 1:
            n = candidates[0]
            val = qty[n]
            cell = cell_start.copy()
            while True:
                lin = 0
                for k in range(d):
                    lin += cell[k] * gstrides[k]
                result[lin] = val

                k = 0
                while k < d:
                    if cell[k] + 1 < cell_stop[k]:
                        cell[k] += 1
                        for j in range(k):
                            cell[j] = cell_start[j]
                        break
                    k += 1
                if k == d:
                    break
            continue

        # Sweep grid cells inside the tile and pick nearest candidate.
        cell = cell_start.copy()
        center = np.empty(d, dtype=np.float64)
        while True:
            for k in range(d):
                center[k] = mins[k] + (cell[k] + 0.5) * dx[k]

            best_d2 = 1.0e308
            best_qty = 0.0
            for c in range(cand_count):
                n = candidates[c]
                d2 = 0.0
                for k in range(d):
                    diff = pos[n, k] - center[k]
                    d2 += diff * diff
                if d2 < best_d2:
                    best_d2 = d2
                    best_qty = qty[n]

            lin = 0
            for k in range(d):
                lin += cell[k] * gstrides[k]
            result[lin] = best_qty

            k = 0
            while k < d:
                if cell[k] + 1 < cell_stop[k]:
                    cell[k] += 1
                    for j in range(k):
                        cell[j] = cell_start[j]
                    break
                k += 1
            if k == d:
                break

    return result

@numba.njit(parallel=True, fastmath=True)
def _bin_particles_voronoi_2d(pos, qty, mins, maxs, dims, tile_size, search_radius):
    n_part = pos.shape[0]

    nx = dims[0]
    ny = dims[1]

    dx = (maxs[0] - mins[0]) / nx
    dy = (maxs[1] - mins[1]) / ny
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    nbx = (nx + tile_size - 1) // tile_size
    nby = (ny + tile_size - 1) // tile_size
    nbuckets = nbx * nby

    bucket_ids = np.empty(n_part, dtype=np.int64)
    counts = np.zeros(nbuckets, dtype=np.int64)

    x_min = mins[0]
    y_min = mins[1]

    for n in range(n_part):
        x = pos[n, 0]
        y = pos[n, 1]

        ix = int((x - x_min) * inv_dx)
        iy = int((y - y_min) * inv_dy)

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            bucket_ids[n] = -1
            continue

        bx = ix // tile_size
        by = iy // tile_size
        bucket = bx + nbx * by

        bucket_ids[n] = bucket
        counts[bucket] += 1

    offsets = np.empty(nbuckets + 1, dtype=np.int64)
    offsets[0] = 0
    for b in range(nbuckets):
        offsets[b + 1] = offsets[b] + counts[b]

    bucket_particles = np.empty(offsets[nbuckets], dtype=np.int64)
    cursor = offsets[:-1].copy()

    for n in range(n_part):
        bucket = bucket_ids[n]
        if bucket >= 0:
            slot = cursor[bucket]
            bucket_particles[slot] = n
            cursor[bucket] = slot + 1

    result = np.zeros((nx, ny))

    ntiles = nbx * nby
    for tile in numba.prange(ntiles):
        bx = tile % nbx
        by = tile // nbx

        x0 = bx * tile_size
        y0 = by * tile_size
        x1 = x0 + tile_size
        y1 = y0 + tile_size

        if x1 > nx:
            x1 = nx
        if y1 > ny:
            y1 = ny

        bx0 = bx - search_radius
        bx1 = bx + search_radius
        by0 = by - search_radius
        by1 = by + search_radius

        if bx0 < 0:
            bx0 = 0
        if by0 < 0:
            by0 = 0
        if bx1 >= nbx:
            bx1 = nbx - 1
        if by1 >= nby:
            by1 = nby - 1

        cand_count = 0
        for jbx in range(bx0, bx1 + 1):
            for jby in range(by0, by1 + 1):
                cand_count += counts[jbx + nbx * jby]

        if cand_count == 0:
            continue

        candidates = np.empty(cand_count, dtype=np.int64)
        p = 0
        for jbx in range(bx0, bx1 + 1):
            for jby in range(by0, by1 + 1):
                bucket = jbx + nbx * jby
                start = offsets[bucket]
                stop = offsets[bucket + 1]
                for s in range(start, stop):
                    candidates[p] = bucket_particles[s]
                    p += 1

        for i in range(x0, x1):
            x_center = x_min + (i + 0.5) * dx
            for j in range(y0, y1):
                y_center = y_min + (j + 0.5) * dy

                best_d2 = 1.0e308
                best_qty = 0.0

                for c in range(cand_count):
                    n = candidates[c]
                    dxp = pos[n, 0] - x_center
                    dyp = pos[n, 1] - y_center
                    d2 = dxp * dxp + dyp * dyp

                    if d2 < best_d2:
                        best_d2 = d2
                        best_qty = qty[n]

                result[i, j] = best_qty

    return result

def bin_particles_voronoi(pos, qty, mins=None, maxs=None, dims=None, weight=None,
                          tile_size=32, search_radius=1):
    """
    Fast Voronoi-style rasterizer for N-D grids.

    Each grid cell is assigned the value of the nearest particle, using a tiled
    local-neighborhood search over particle buckets. This is an O(N) expected
    method when bucket occupancy stays bounded, and it stays close in spirit to
    bin_particles_scatter while rendering a piecewise-constant Voronoi map.
    """
    pos = np.asarray(pos)
    qty = np.asarray(qty)

    if weight is not None:
        weight = np.asarray(weight)
        qty = qty * weight

    if dims is None:
        dims = np.array([512, 512], dtype=np.int64)
    else:
        dims = np.asarray(dims, dtype=np.int64)

    d = dims.shape[0]

    if mins is None:
        mins = np.min(pos[:, :d], axis=0)
    else:
        mins = np.asarray(mins)

    if maxs is None:
        maxs = np.max(pos[:, :d], axis=0)
    else:
        maxs = np.asarray(maxs)

    tile_size = int(tile_size)
    search_radius = int(search_radius)
    if tile_size < 1:
        tile_size = 1
    if search_radius < 0:
        search_radius = 0

    if d == 2:
        return _bin_particles_voronoi_2d(pos, qty, mins, maxs, dims, tile_size, search_radius)

    return _bin_particles_voronoi_nd(pos, qty, mins, maxs, dims, tile_size, search_radius).reshape(tuple(dims))
    

#### Kernel density estimation method for Voronoi-like ####
### -> really finnicky to get bandwidth right, seems kinda wrong, and not even fast. Originally based on what Phil did in fastkde, but i would avoid this for now until i can figure out its issues
def bin_particles_kde(pos, qty, mins=None, maxs=None, dims=None,
                          bandwidth_fix_factor=1, padding=True):
    """
    Fast KDE using O(N) binning + O(Ngrid log Ngrid) FFT convolution with Gaussian kernel.
    """
    pos = np.asarray(pos)
    qty = np.asarray(qty)
    N, D = pos.shape

    if dims is None:
        dims = np.array([512]*D)
    else:
        dims = np.asarray(dims)
    d = dims.shape[0]

    if mins is None:
        mins = np.min(pos[:, :d], axis=0)
    if maxs is None:
        maxs = np.max(pos[:, :d], axis=0)
    mins = np.asarray(mins)
    maxs = np.asarray(maxs)
    dx = (maxs - mins) / dims

    # handle global bandwidth
    if bandwidth_fix_factor is None:
        bandwidth_fix_factor = 1
    
    sigma = np.std(pos[:, :d], axis=0, ddof=1)
    bandwidth = bandwidth_fix_factor * sigma * N**(-1.0/(d + 4)) # Scott's rule of thumb
    bandwidth = np.atleast_1d(bandwidth)
    if bandwidth.ndim == 0:
        bandwidth = np.full(d, bandwidth)
    elif bandwidth.ndim == 1:
        bandwidth = bandwidth
    sigma_pixels = bandwidth / dx

    # --- Bin particles onto grid ---
    indices = ((pos[:, :d] - mins) / dx).astype(int)
    mask = np.all((indices >= 0) & (indices < dims), axis=1)
    indices = indices[mask]
    weights = qty[mask]
    grid = np.zeros(tuple(dims))
    for idx, w in zip(indices, weights):
        grid[tuple(idx)] += w

    # --- Optional symmetric padding (like fastkde) ---
    if padding:
        pad = [dim//2 for dim in dims]
        grid = np.pad(grid, [(p,p) for p in pad], mode='reflect')
        dims_pad = np.array(grid.shape)
    else:
        dims_pad = dims

    # --- Build Gaussian kernel in Fourier space ---
    k_axes = [np.fft.fftfreq(n) * 2*np.pi for n in dims_pad]
    K2 = np.zeros(dims_pad)
    for i in range(d):
        shape_i = np.ones(d, dtype=int)
        shape_i[i] = dims_pad[i]
        ki = k_axes[i].reshape(tuple(shape_i))
        K2 += (sigma_pixels[i]**2) * (ki**2)
    kernel_ft = np.exp(-0.5*K2)

    # --- FFT convolution ---
    grid_ft = np.fft.fftn(grid)
    smoothed = np.fft.ifftn(grid_ft * kernel_ft).real

    # --- Remove padding ---
    if padding:
        slices = tuple(slice(pad[i], pad[i]+dims[i]) for i in range(d))
        smoothed = smoothed[slices]

    # --- Normalize like fastkde ---
    norm_factor = np.prod(dx) * (2*np.pi)**(d/2) * np.prod(sigma_pixels)
    smoothed /= norm_factor

    return smoothed



#### Map back onto particles ####

@numba.njit
def _map_grid_to_particles_serial(indices, grid, result, default_value=0.0):
    N = indices.shape[0]
    dims = np.asarray(grid.shape)
    d = dims.shape[0]

    strides = np.empty(d, dtype=np.int64)
    strides[d - 1] = 1
    for i in range(d - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]

    flat_grid = grid.ravel()

    for n in range(N):
        idx = indices[n]
        valid = True
        lin = 0
        for k in range(d):
            if idx[k] < 0 or idx[k] >= dims[k]:
                valid = False
                break
            lin += idx[k] * strides[k]
        if valid:
            result[n] = flat_grid[lin]
        else:
            result[n] = default_value
    return result

def map_grid_to_particles(pos, grid, mins=None, maxs=None, default_value=0.0):
    """
    Map grid values back onto particles by using value in the nearest grid cell.
    All particles in a grid cell are given the same value. For conservative quantities, the 
    grid should be normalized by the number of particles in each cell beforehand.
    """
    pos = np.asarray(pos)
    grid = np.asarray(grid)
    N, D = pos.shape
    dims = np.asarray(grid.shape)
    d = dims.shape[0]

    if mins is None:
        mins = np.min(pos[:, :d], axis=0)
    if maxs is None:
        maxs = np.max(pos[:, :d], axis=0)
    mins = np.asarray(mins)
    maxs = np.asarray(maxs)
    dx = (maxs - mins) / dims

    indices = ((pos[:, :d] - mins) / dx).astype(int)
    result = np.zeros(N, dtype=grid.dtype)
    _map_grid_to_particles_serial(indices, grid, result, default_value=default_value)
    
    return result


### tree-based interpolation

@numba.njit(cache=True)
def _cartesian_grid_query(points, axes, k):
    distances = np.empty((len(points), k)); indices = np.empty((len(points), k), dtype=np.int64)
    for row in range(len(points)):
        best = np.zeros(k); best_indices = np.zeros(k, dtype=np.int64); count = 1
        for dim, axis in enumerate(axes):

            # order the nearest coordinates on this axis by physical distance
            size = len(axis); axis_count = min(k, size)
            right = np.searchsorted(axis, points[row, dim]); left = right-1
            axis_distances = np.empty(axis_count); axis_indices = np.empty(axis_count, dtype=np.int64)
            for i in range(axis_count):
                dl = (axis[left]-points[row, dim])**2 if left >= 0 else np.inf
                dr = (axis[right]-points[row, dim])**2 if right < size else np.inf
                if dl <= dr:
                    axis_distances[i] = dl; axis_indices[i] = left; left -= 1
                else:
                    axis_distances[i] = dr; axis_indices[i] = right; right += 1

            # retain the nearest combinations without materializing the Cartesian product
            merged = np.full(k, np.inf); merged_indices = np.empty(k, dtype=np.int64)
            found = 0
            for i in range(count):
                for j in range(axis_count):
                    distance = best[i]+axis_distances[j]
                    if distance >= merged[-1]: break
                    slot = min(found, k-1)
                    while slot > 0 and distance < merged[slot-1]:
                        merged[slot] = merged[slot-1]; merged_indices[slot] = merged_indices[slot-1]
                        slot -= 1
                    merged[slot] = distance; merged_indices[slot] = best_indices[i]*size+axis_indices[j]
                    found = min(found+1, k)
            best, best_indices, count = merged, merged_indices, found
        distances[row] = np.sqrt(best); indices[row] = best_indices
    return distances, indices


class CartesianGrid:
    """Nearest-neighbor queries on a Cartesian product, storing only its sorted axes.

    The query interface supports the Euclidean queries used by mfm_value_at_pos and
    raytrace_particles. Flatten fields in C order, as for meshgrid(indexing='ij').
    Distances and kernel weights match a KD-tree; ordering of equal-distance neighbors
    may differ. Queries run in compiled serial code; workers is accepted for compatibility.
    Axes may have unequal or nonuniform spacing. No full coordinate array or tree is built.
    """
    def __init__(self, axes):
        self.axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
        if not 1 <= len(self.axes) <= 3 or any(axis.ndim != 1 or not len(axis) or
                np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0) for axis in self.axes):
            raise ValueError('Use one to three finite, strictly increasing coordinate axes.')
        self.m = len(self.axes); self.n = math.prod(len(axis) for axis in self.axes)

    def query(self, points, k=1, workers=-1, distance_upper_bound=np.inf):
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != self.m or np.any(~np.isfinite(points)) or int(k) != k or not 1 <= k <= self.n:
            raise ValueError('Use finite query points with dimensions and neighbor count matching the Cartesian grid.')
        shape = points.shape[:-1]+(int(k),)
        distances, indices = _cartesian_grid_query(points.reshape(-1, self.m), self.axes, int(k))
        outside = distances >= distance_upper_bound
        distances[outside] = np.inf; indices[outside] = self.n
        distances, indices = distances.reshape(shape), indices.reshape(shape)
        return (distances[..., 0], indices[..., 0]) if k == 1 else (distances, indices)


def mfm_value_at_pos(query_pos, tree=None, pos=None, N_NGB=32, workers=-1, distance_upper_bound=np.inf, tol=1e-8, maxiter=50, 
                     qty=None, ret_h=False, ret_neighbours=False, ret_convergence_info=False, approx=False): # return options
    """
    Get interpolated quantity at specific position array using a tree-based approach for MFM or SPH with variable smoothing lengths.
    This method is more efficient for large datasets and can handle varying smoothing lengths.
    Note h is the maximum radius of support for each particle (r/h = 1 is maximum distance).
    tol is the absolute tolerance for the difference of the effective number of neighbors and the target number of neighbors (N_NGB). 
    Giving a qty array will return the interpolated quantity at the query positions. 
    """
    from scipy.optimize import newton
    if qty is not None:
        qty = np.asarray(qty, dtype=np.float64)
    query_pos = np.asarray(query_pos, dtype=np.float64)
    d = query_pos.shape[-1]
    assert d in (1, 2, 3), f"Only 1D, 2D, and 3D are supported (but d = {d})."
    if tree is None:
        # Build a KD-tree for efficient neighbor searches
        from scipy.spatial import cKDTree
        if pos is None:
            raise ValueError("Either a tree or particle positions must be provided.")
        pos = np.asarray(pos, dtype=np.float64)
        tree = cKDTree(pos)

    if not approx:
        # Query the tree for particles near the query positions
        neighbours_dist, neighbours_idx = tree.query(query_pos, k=2*N_NGB, workers=workers, distance_upper_bound=distance_upper_bound) # shape: (pts, neighbours)
        neighbours_dist = np.clip(neighbours_dist, 1e-15, None)

        # iteratively find h for each query point (Newton-Raphson)
        h0 = neighbours_dist[:, N_NGB-1] # first guess for h is the distance to the 32nd neighbour
        h1 = np.mean(neighbours_dist, axis=-1)

        class FunctionCache:
            def __init__(self, neighbours_dist, d):
                self.neighbours_dist = neighbours_dist
                self.d = d
                self._w = np.zeros_like(neighbours_dist)
                #self._wprime = np.zeros_like(neighbours_dist)
                Cd = 1.0 if d == 1 else (np.pi if d == 2 else (4.0/3.0 * np.pi))
                Kd = (3.0 / np.pi) if self.d == 3 else (15.0 / (7.0 * np.pi)) if self.d == 2 else 0.5 if self.d == 1 else None
                self.C = Cd*Kd

            def f(self, _h):
                # scipy always calls this function first and only once per iteration (see https://github.com/scipy/scipy/blob/c71fcecc1bdd07423de43da2588a5af83d95b5f7/scipy/optimize/_zeros_py.py#L319)
                # so we can update all cached values here and use them in the derivative functions without recomputing them
                _invhp = 1.0/_h[:, np.newaxis] # NOTE: tried using invh as parameter but it was less stable for probems I am interested in than using h directly
                self._w.fill(0.0)
                #self._wprime.fill(0.0)
                q = neighbours_dist * _invhp
                cubic_kernel(q, _invhp, d=self.d, w=self._w, wprime=False, wprime2=False, q_instead=True, dw_dq_instead=True)
                #self._wprime *= -q*_invhp # convert to dh rather than dr
                return (self.C * self._w.sum(axis=-1)) - N_NGB

            # def fprime(self, _h): # NOTE: tried using prime method but its fairly unstable compared to secant method used below
            #     return self.C * self._wprime.sum(axis=-1) 
            
        cache = FunctionCache(neighbours_dist, d)
        h, converged, zero_der = newton(cache.f, x0=h0, x1=h1, fprime=None, tol=tol, maxiter=maxiter, full_output=True, disp=False)
        h[~converged] = h0[~converged] # fallback to distance to initial guess if not converged

        if not np.all(converged):
            warnings.warn(f"Newton-Raphson did not converge for {np.sum(~converged)} query points.")
        if np.any(zero_der):
            warnings.warn(f"Zero derivative encountered for {np.sum(zero_der)} query points during Newton-Raphson.")
    else:
        # approximate h as the distance to the 32nd neighbour only, and skip iteration phase
        neighbours_dist, neighbours_idx = tree.query(query_pos, k=N_NGB, workers=workers, distance_upper_bound=distance_upper_bound) # shape: (pts, neighbours) 
        h = neighbours_dist[:, N_NGB-1]
        if ret_convergence_info:
            converged = np.zeros(len(h), dtype=bool)
            zero_der = np.zeros(len(h), dtype=bool)
    
    if qty is not None or ret_neighbours:
        invhp = 1.0/h[:, None]
        W = cubic_kernel(neighbours_dist, invh=invhp, d=d)

    ret = []
    if qty is not None:
        ret.append(np.sum(qty[neighbours_idx, ...] * W, axis=1) / np.sum(W, axis=1))
    if ret_h:
        ret.append(h)
    if ret_neighbours:
        ret.extend([neighbours_idx, W]) # note that when W = 0, the corresponding neighbour is not actually used in the interpolation
    if ret_convergence_info:
        ret.extend([converged, zero_der])

    return ret[0] if len(ret) == 1 else tuple(ret)



### Hybrid direct/sampled grids ###

def sample_particles_grid(qty, mins, maxs, weight=None, dims=None, sample_dims=None, sample_mask=None,
                          pos=None, tree=None, grid_to_particle=None, n_batches=128,
                          approx=True, N_NGB=32, workers=-1, distance_upper_bound=np.inf, 
                          rng=None, sampling='random', log=False, log_every=1, sample_state=None,
                          save_every=None, save_callback=None, ret_state=False, query_chunk_size=250_000,
                          eta_update_seconds=5.0):
    """
    Estimate a grid by uniformly sampling transformed coordinates and interpolating particles.

    If you set sampling to sobol, make sure that the number of batches is a power of two for best results. 
    Dimensions is the number of cells in each dimension of the grid. Sample dimensions is the number of samples in each dimension.
    By default, the number of samples in each dimension is equal to the number of grid cells in that dimension and prod(dims)^(1/(D*len(dims))) in other dimensions.
    query_chunk_size limits the number of query points held during each neighbor search.
    """
    # validate particle and interpolation inputs
    if isinstance(qty, tuple):
        qty = qty[0] if len(qty) == 1 else np.column_stack(qty)
    qty = np.asarray(qty, dtype=np.float64)
    if qty.ndim not in (1, 2):
        raise ValueError('qty must be a one- or two-dimensional particle array.')
    multiple_output = qty.ndim == 2
    n_quantities = 1 if qty.ndim == 1 else qty.shape[1]
    qty_values = qty[:, None] if qty.ndim == 1 else qty
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    dims = np.asarray([128, 128] if dims is None else dims, dtype=np.int64)
    D = mins.size
    assert D == maxs.size, 'mins and maxs must have the same number of dimensions.'
    assert D in (1, 2, 3), 'Only 1D, 2D, and 3D grids are supported.'
    d = dims.size
    assert d in (1, 2, 3), 'Only 1D, 2D, and 3D grids are supported.'
    if pos is not None:
        pos = np.asarray(pos, dtype=np.float64)
        assert pos.ndim == 2, 'pos must be a two-dimensional array of particle positions.'
        assert pos.shape[0] == qty.shape[0], 'pos must have the same number of rows as qty has elements.'
        assert pos.shape[1] >= len(dims), 'pos must have at least as many columns as the number of grid dimensions.'
        assert pos.shape[1] == D, 'pos must have the same number of columns as the number of grid dimensions.'
    if sample_dims is None:
        sample_dims = np.copy(dims)
        over_sampled_other_dim = max(1, int(np.prod(dims)**(1.0/(D*len(dims)))))
        sample_dims = np.append(sample_dims, [over_sampled_other_dim,] * (D - len(sample_dims)))
    else:
        sample_dims = np.asarray(sample_dims, dtype=np.int64)
    if weight is not None:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != (qty.shape[0],):
            raise ValueError('weight must have one value per particle.')
    if tree is None:
        if pos is None:
            raise ValueError('Either pos or tree must be supplied.')
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] != qty.shape[0]:
            raise ValueError('pos must have one row for every quantity value.')
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
    if grid_to_particle is None:
        grid_to_particle = lambda g: g

    # validate sampling-grid geometry
    if mins.ndim != 1 or maxs.shape != mins.shape or sample_dims.shape != mins.shape:
        raise ValueError('mins, maxs, and sample_dims must be one-dimensional arrays of equal length.')
    if dims.ndim != 1 or d > mins.size:
        raise ValueError('dims must describe no more dimensions than the sampling grid.')
    if np.any(maxs <= mins) or np.any(dims <= 0) or np.any(sample_dims <= 0):
        raise ValueError('Grid bounds and dimensions must be positive and ordered.')
    if n_batches < 1:
        raise ValueError('n_batches must be at least one.')
    if sampling not in ('random', 'sobol'):
        raise ValueError("sampling must be either 'random' or 'sobol'.")
    if log_every < 1:
        raise ValueError('log_every must be at least one.')
    if save_callback is not None and (save_every is None or save_every < 1):
        raise ValueError('save_every must be at least one when save_callback is supplied.')
    if query_chunk_size is not None and query_chunk_size < 1:
        raise ValueError('query_chunk_size must be positive or None.')
    if eta_update_seconds <= 0:
        raise ValueError('eta_update_seconds must be positive.')

    # select map cells that require sampling
    if sample_mask is None:
        sample_mask = np.ones(tuple(dims), dtype=bool)
    else:
        sample_mask = np.asarray(sample_mask, dtype=bool)
        if sample_mask.shape != tuple(dims):
            raise ValueError('sample_mask must have shape dims.')

    # construct sample cells directly for small maps and lazily for large maps
    axes = [np.linspace(mn, mx, n, endpoint=False) for mn, mx, n in zip(mins, maxs, sample_dims)]
    sample_total = int(np.prod(sample_dims))
    chunk_size = sample_total if query_chunk_size is None else query_chunk_size
    if sample_total <= chunk_size:
        grid_pos = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, mins.size)
        map_index = ((grid_pos[:, :d] - mins[:d]) * dims / (maxs[:d] - mins[:d])).astype(np.int64)
        np.minimum(map_index, dims - 1, out=map_index)
        sample_mask_flat = sample_mask[tuple(map_index.T)]
    else:
        grid_pos = None
        sample_mask_flat = None
    shape = tuple(dims)
    grid_shape = (n_quantities,) + shape if multiple_output else shape
    if sample_mask_flat is not None and not np.any(sample_mask_flat):
        sampled_grid = np.full(grid_shape, np.nan)
        grid_err = np.full(grid_shape, np.nan)
        state = {'mean': np.zeros(grid_shape), 'm2': np.zeros(grid_shape), 'batches': np.zeros(grid_shape, dtype=np.int64)}
        return (sampled_grid, grid_err, state) if ret_state else (sampled_grid, grid_err)

    # initialize batched sampled moments
    rng = np.random.default_rng(rng)
    sample_width = (maxs - mins) / sample_dims
    if sample_state is None:
        sample_mean = np.zeros(grid_shape)
        sample_m2 = np.zeros(grid_shape)
        sample_batches = np.zeros(grid_shape, dtype=np.int64)
    else:
        sample_mean = np.asarray(sample_state['mean'], dtype=np.float64).copy()
        sample_m2 = np.asarray(sample_state['m2'], dtype=np.float64).copy()
        sample_batches = np.asarray(sample_state['batches'], dtype=np.int64).copy()
        if sample_mean.shape != grid_shape or sample_m2.shape != grid_shape or sample_batches.shape != grid_shape:
            raise ValueError('sample_state arrays must have shape dims or (n_quantities, *dims).')
    previous_batches = int(np.max(sample_batches))

    # initialize random or low-discrepancy intra-cell jitters
    if sampling == 'sobol':
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=mins.size, scramble=True, seed=rng)
        if previous_batches:
            sobol.fast_forward(previous_batches)
        sobol_jitter = sobol.random_base2(int(np.log2(n_batches))) if not previous_batches and n_batches & (n_batches - 1) == 0 else sobol.random(n_batches)

    # initialize optional progress timers
    if log:
        import time
        time_spent_on_sampling = 0.0
        time_spent_on_query = 0.0
        time_spent_on_qty_calc = 0.0
        time_spent_on_bin = 0.0
        start_time = time.time()
        last_eta_update = start_time
    total_chunks = max(1, int(np.ceil(sample_total/chunk_size)))

    # print progress during long chunked batches
    def report_eta(progress, force=False):
        nonlocal last_eta_update
        if not log:
            return
        now = time.time()
        if force or now - last_eta_update >= eta_update_seconds:
            elapsed_time = now - start_time
            eta = elapsed_time * (n_batches - progress) / progress
            eta_str = f'ETA {int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d} / total ETA {elapsed_time*n_batches/progress/60:.2f} min'
            if save_callback is not None:
                eta_str += ' (saved {} times)'.format(int(progress) // save_every)
            print(f'\r{eta_str}', end='', flush=True)
            last_eta_update = now

    for batch in range(n_batches):
        # construct query chunks and jitter points within their sample cells
        if log:
            batch_start_time = time.time()
        if sample_mask_flat is not None:
            jitter = rng.uniform(0.0, sample_width, size=(sample_mask_flat.sum(), mins.size)) if sampling == 'random' else sobol_jitter[batch] * sample_width
            query_grid_pos = grid_pos[sample_mask_flat] + jitter
            query_real_pos = np.asarray(grid_to_particle(query_grid_pos), dtype=np.float64)
            if query_real_pos.shape != query_grid_pos.shape:
                raise ValueError('grid_to_particle must preserve the coordinate array shape.')
            query_chunks = ((query_grid_pos, query_real_pos),)
        else:
            def query_chunks():
                for chunk_start in range(0, sample_total, chunk_size):
                    chunk_stop = min(chunk_start + chunk_size, sample_total)
                    flat_index = np.arange(chunk_start, chunk_stop)
                    sample_index = np.column_stack(np.unravel_index(flat_index, tuple(sample_dims)))
                    chunk_grid_pos = mins + sample_index*sample_width
                    map_index = ((chunk_grid_pos[:, :d] - mins[:d]) * dims / (maxs[:d] - mins[:d])).astype(np.int64)
                    np.minimum(map_index, dims - 1, out=map_index)
                    chunk_mask = sample_mask[tuple(map_index.T)]
                    if not np.any(chunk_mask):
                        continue
                    chunk_grid_pos = chunk_grid_pos[chunk_mask]
                    jitter = rng.uniform(0.0, sample_width, size=chunk_grid_pos.shape) if sampling == 'random' else sobol_jitter[batch] * sample_width
                    chunk_grid_pos += jitter
                    chunk_real_pos = np.asarray(grid_to_particle(chunk_grid_pos), dtype=np.float64)
                    if chunk_real_pos.shape != chunk_grid_pos.shape:
                        raise ValueError('grid_to_particle must preserve the coordinate array shape.')
                    yield chunk_grid_pos, chunk_real_pos
            query_chunks = query_chunks()

        # process interpolation queries in bounded chunks to limit neighbor-list memory
        if log:
            time_spent_on_sampling += time.time() - batch_start_time
            batch_start_time = time.time()
        batch_sum = np.zeros(grid_shape)
        batch_weight = np.zeros(shape)
        for chunk_number, (chunk_grid_pos, chunk_real_pos) in enumerate(query_chunks):
            if approx:
                neighbours_idx, W = mfm_value_at_pos(chunk_real_pos, tree=tree, ret_neighbours=True,
                                                      approx=True, N_NGB=N_NGB, workers=workers,
                                                      distance_upper_bound=distance_upper_bound)
            else:
                neighbours_idx, W, converged, _ = mfm_value_at_pos(chunk_real_pos, tree=tree, ret_neighbours=True,
                                                                     ret_convergence_info=True, approx=False, N_NGB=N_NGB,
                                                                     workers=workers, distance_upper_bound=distance_upper_bound)
                chunk_grid_pos = chunk_grid_pos[converged]
                neighbours_idx = neighbours_idx[converged]
                W = W[converged]
            if log:
                time_spent_on_query += time.time() - batch_start_time
                batch_start_time = time.time()

            # interpolate particle values and bin the current query chunk
            kernel_sum = W.sum(axis=1)
            valid = kernel_sum > 0
            chunk_grid_pos = chunk_grid_pos[valid]
            neighbours_idx = neighbours_idx[valid]
            W = W[valid]
            kernel_sum = kernel_sum[valid]
            query_qty = np.sum(qty_values[neighbours_idx] * W[..., None], axis=1) / kernel_sum[:, None]
            if weight is None:
                query_weight = np.ones(query_qty.shape[0])
            else:
                query_weight = np.sum(weight[neighbours_idx] * W, axis=1) / kernel_sum
            if log:
                time_spent_on_qty_calc += time.time() - batch_start_time
                batch_start_time = time.time()
            query_values = query_qty * query_weight[:, None] if multiple_output else query_qty[:, 0] * query_weight
            binned = bin_particles_direct(chunk_grid_pos[:, :d], query_values, query_weight,
                                          mins=mins[:d], maxs=maxs[:d], dims=dims)
            batch_sum += binned[:-1] if multiple_output else binned[0]
            batch_weight += binned[-1] if multiple_output else binned[1]
            report_eta(batch + (chunk_number + 1)/total_chunks)

        # calculate the complete sampled map for this batch
        batch_grid = np.full(grid_shape, np.nan)
        np.divide(batch_sum, batch_weight, out=batch_grid, where=batch_weight != 0)
        del batch_sum, batch_weight
        valid = np.isfinite(batch_grid)
        next_batches = sample_batches[valid] + 1
        delta = batch_grid[valid] - sample_mean[valid]
        sample_mean[valid] += delta / next_batches
        sample_m2[valid] += delta * (batch_grid[valid] - sample_mean[valid])
        sample_batches[valid] = next_batches

        # report batch progress and estimated completion time
        if log:
            time_spent_on_bin += time.time() - batch_start_time
            if (batch + 1) % log_every == 0 or batch + 1 == n_batches:
                report_eta(batch + 1, force=True)
        if save_callback is not None and ((batch + 1) % save_every == 0 or batch + 1 == n_batches):
            sampled_grid = np.where(sample_batches != 0, sample_mean, np.nan)
            grid_err = np.full(shape, np.nan)
            valid_error = sample_batches > 1
            grid_err[valid_error] = np.sqrt(np.maximum(sample_m2[valid_error], 0.0) / (sample_batches[valid_error] * (sample_batches[valid_error] - 1)))
            save_callback(sampled_grid, grid_err, {'mean': sample_mean, 'm2': sample_m2, 'batches': sample_batches}, batch + 1)

    # calculate the sampled mean and its batch standard error
    sampled_grid = np.where(sample_batches != 0, sample_mean, np.nan)
    grid_err = np.full(grid_shape, np.nan)
    valid = sample_batches > 1
    grid_err[valid] = np.sqrt(np.maximum(sample_m2[valid], 0.0) / (sample_batches[valid] * (sample_batches[valid] - 1)))

    # report the sampling timing breakdown
    if log:
        total_time = time.time() - start_time
        other_time = total_time - time_spent_on_sampling - time_spent_on_query - time_spent_on_qty_calc - time_spent_on_bin
        print()
        print(f'Total time spent: {total_time:.2f} seconds')
        print(f' > Time spent on sampling: {time_spent_on_sampling:.2f} seconds ({time_spent_on_sampling/total_time*100:.2f}%)')
        print(f' > Time spent on queries: {time_spent_on_query:.2f} seconds ({time_spent_on_query/total_time*100:.2f}%)')
        print(f' > Time spent on quantity calculations: {time_spent_on_qty_calc:.2f} seconds ({time_spent_on_qty_calc/total_time*100:.2f}%)')
        print(f' > Time spent on quantity binning: {time_spent_on_bin:.2f} seconds ({time_spent_on_bin/total_time*100:.2f}%)')
        print(f' > Time spent on other misc tasks: {other_time:.2f} seconds ({other_time/total_time*100:.2f}%)')
    state = {'mean': sample_mean, 'm2': sample_m2, 'batches': sample_batches}
    return (sampled_grid, grid_err, state) if ret_state else (sampled_grid, grid_err)



def bin_particles_hybrid(pos, qty, mins, maxs, weight=None, dims=None, sample_dims=None,
                         tree=None, particle_to_grid=None, grid_to_particle=None,
                         n_batches=128, transition_occupancy=16, sample_eps=0.1, rng=None, sampling='random',
                         approx=True, N_NGB=32, workers=-1, distance_upper_bound=np.inf, 
                         ret_components=False, log=False, log_every=1, save_path=None, save_every=10,
                         query_chunk_size=250_000, eta_update_seconds=5.0, _test_size=1000):
    """
    Blend a directly binned particle map with an interpolated sampled map.

    If you set sampling to sobol, make sure that the number of batches is a power of two for best results. 
    Dimensions is the number of cells in each dimension of the grid. Sample dimensions is the number of samples in each dimension.
    Sample epsilon is the cutoff below which the fractional weight of the sampling grid is considered negligible and we do not sample that cell.
    """
    # validate particle and blending inputs
    pos = np.asarray(pos, dtype=np.float64)
    if isinstance(qty, tuple):
        qty = qty[0] if len(qty) == 1 else np.column_stack(qty)
    qty = np.asarray(qty, dtype=np.float64)
    if pos.ndim != 2 or qty.ndim not in (1, 2) or pos.shape[0] != qty.shape[0]:
        raise ValueError('pos and qty must contain the same number of particles; qty may be one- or two-dimensional.')
    multiple_output = qty.ndim == 2
    n_quantities = 1 if qty.ndim == 1 else qty.shape[1]
    if weight is not None and np.asarray(weight).shape != (qty.shape[0],):
        raise ValueError('weight must have one value per particle.')
    if transition_occupancy <= 0 or not 0 < sample_eps < 1:
        raise ValueError('transition_occupancy must be positive and sample_eps must be in (0, 1).')
    if save_path and save_every < 1:
        raise ValueError('save_every must be at least one when save_path is supplied.')
    rng = np.random.default_rng(rng)
    if particle_to_grid is None:
        particle_to_grid = lambda p: p
    if grid_to_particle is None:
        grid_to_particle = lambda g: g
    if _test_size > 0:
        _test_subset = rng.choice(pos, size=min(_test_size, pos.shape[0]), replace=False, axis=0) # minimal test, you should test it on the whole pos array to be sure
        if not np.isclose(grid_to_particle(particle_to_grid(_test_subset)), _test_subset).all():
            raise ValueError('Your particle to grid and grid to particle functions are not consistent with each other!')
    if tree is None:
        if pos.ndim != 2 or pos.shape[0] != qty.shape[0]:
            raise ValueError('pos must have one row for every quantity value.')
        from scipy.spatial import cKDTree
        print('Building KD-tree for particle positions...')
        tree = cKDTree(pos)

    # directly bin particle values and particle counts
    if log:
        import time
        start_time = time.time()
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    dims = np.asarray([128, 128] if dims is None else dims, dtype=np.int64)
    if sample_dims is None:
        sample_dims = np.copy(dims)
        sample_dims = np.append(sample_dims, [max(1, int(np.prod(dims)**(1.0/(mins.size*dims.size))))] * (mins.size - dims.size))
    else:
        sample_dims = np.asarray(sample_dims, dtype=np.int64)
    grid_pos = np.asarray(particle_to_grid(pos), dtype=np.float64)
    if grid_pos.ndim != 2 or grid_pos.shape[0] != qty.shape[0] or grid_pos.shape[1] < dims.size:
        raise ValueError('particle_to_grid must return one grid-coordinate row per particle.')
    particle_weight = np.ones(qty.shape[0]) if weight is None else np.asarray(weight, dtype=np.float64)
    weighted_qty = qty*particle_weight[:, None] if qty.ndim == 2 else qty*particle_weight
    particle_count_input = np.ones(qty.shape[0])
    binned = bin_particles_direct(grid_pos[:, :dims.size], weighted_qty, particle_weight, particle_count_input,
                                  mins=mins[:dims.size], maxs=maxs[:dims.size], dims=dims)
    grid_shape = (n_quantities,) + tuple(dims) if multiple_output else tuple(dims)
    direct_grid = np.zeros(grid_shape)
    direct_weight_grid = binned[n_quantities] if multiple_output else binned[1]
    direct_sum = binned[:n_quantities] if multiple_output else binned[0]
    np.divide(direct_sum, direct_weight_grid, out=direct_grid, where=direct_weight_grid != 0)
    particle_count = binned[n_quantities + 1] if multiple_output else binned[2]

    # sample low-occupancy cells and blend valid sampled estimates
    max_sample_occupancy = np.maximum(1, transition_occupancy * (1.0 / sample_eps - 1.0))
    sample_mask = particle_count < max_sample_occupancy

    # initialize resumable sampling state
    sample_state = None
    if save_path:
        import json, os
        from pathlib import Path
        save_path = Path(save_path)
        metadata = {
            'version': 1,
            'mins': mins.tolist(),
            'maxs': maxs.tolist(),
            'dims': dims.tolist(),
            'sample_dims': sample_dims.tolist(),
            'n_batches': n_batches,
            'transition_occupancy': transition_occupancy,
            'sample_eps': sample_eps,
            'sampling': sampling,
            'approx': approx,
            'N_NGB': N_NGB,
            'n_particles': pos.shape[0],
            'weighted': weight is not None,
            'n_quantities': n_quantities,
        }
        if save_path.exists():
            with np.load(save_path, allow_pickle=False) as previous:
                previous_metadata = json.loads(str(previous['metadata'].item()))
                metadata_match = all(previous_metadata.get(key) == value for key, value in metadata.items() if key != 'n_batches')
                mask_match = 'sample_mask' in previous and np.array_equal(previous['sample_mask'], sample_mask)
                if not metadata_match or not mask_match:
                    raise ValueError(f'Existing save file {save_path} has incompatible metadata.')
                required_state = ('sample_mean', 'sample_m2', 'sample_batches')
                if not all(key in previous for key in required_state):
                    raise ValueError(f'Existing save file {save_path} does not contain resumable sampling state.')
                sample_state = {'mean': previous['sample_mean'], 'm2': previous['sample_m2'], 'batches': previous['sample_batches']}
                if log:
                    print(f'Found compatible previous result at {save_path} with {previous["sample_batches"].max()} batches; combining it with this run.')

    # blend direct and sampled estimates
    direct_weight = particle_count / (particle_count + transition_occupancy)
    def blend_sampled(sampled_grid, grid_err):
        sample_valid = np.isfinite(sampled_grid)
        blend_mask = sample_mask & sample_valid
        grid = direct_grid.copy()
        if not multiple_output:
            grid[blend_mask] = (direct_weight[blend_mask] * direct_grid[blend_mask]
                                + (1.0 - direct_weight[blend_mask]) * sampled_grid[blend_mask])
        else:
            blend_mask = np.broadcast_to(blend_mask, grid_shape)
            direct_weight_broadcast = np.broadcast_to(direct_weight, grid_shape)
            grid[blend_mask] = (direct_weight_broadcast[blend_mask] * direct_grid[blend_mask]
                                + (1.0 - direct_weight_broadcast[blend_mask]) * sampled_grid[blend_mask])
        grid_err = grid_err.copy()
        grid_err[~blend_mask] = 0.0
        if not multiple_output:
            grid_err[blend_mask] *= 1.0 - direct_weight[blend_mask]
        else:
            grid_err[blend_mask] *= 1.0 - np.broadcast_to(direct_weight, grid_shape)[blend_mask]
        return grid, grid_err, sample_valid, blend_mask

    # save atomic checkpoints with the running sampling state
    def save_checkpoint(sampled_grid, grid_err, state, completed_batches):
        grid, grid_err, _, _ = blend_sampled(sampled_grid, grid_err)
        checkpoint_metadata = metadata | {'n_batches': int(np.max(state['batches']))}
        temporary_path = save_path.with_suffix(save_path.suffix + '.tmp')
        with open(temporary_path, 'wb') as file:
            np.savez(file, grid=grid, grid_err=grid_err, direct_grid=direct_grid, sampled_grid=sampled_grid,
                     particle_count=particle_count, sample_mask=sample_mask, sample_mean=state['mean'],
                     sample_m2=state['m2'], sample_batches=state['batches'], metadata=json.dumps(checkpoint_metadata))
        os.replace(temporary_path, save_path)

    # sample low-occupancy cells
    sampled_grid, grid_err, sample_state = sample_particles_grid(
        qty, mins, maxs, pos=pos, tree=tree, weight=weight, dims=dims, grid_to_particle=grid_to_particle,
        sample_dims=sample_dims, sample_mask=sample_mask, n_batches=n_batches, approx=approx,
        N_NGB=N_NGB, workers=workers, distance_upper_bound=distance_upper_bound, rng=rng,
        sampling=sampling, log=log, log_every=log_every, sample_state=sample_state, ret_state=True,
        save_every=save_every if save_path else None, save_callback=save_checkpoint if save_path else None,
        query_chunk_size=query_chunk_size,
        eta_update_seconds=eta_update_seconds,
    )
    grid, grid_err, sample_valid, blend_mask = blend_sampled(sampled_grid, grid_err)
    if log:
        print(f'Hybrid grid finished in {time.time() - start_time:.2f} seconds.')

    # save an empty-sample result that did not invoke the callback
    if save_path and not np.any(sample_mask):
        save_checkpoint(sampled_grid, grid_err, sample_state, 0)

    # return either compact results or diagnostics for inspection
    if not ret_components:
        return grid, grid_err
    return {
        'grid': grid,
        'grid_err': grid_err,
        'direct_grid': direct_grid,
        'sampled_grid': sampled_grid,
        'particle_count': particle_count,
        'direct_weight': direct_weight,
        'sample_mask': sample_mask,
        'sample_valid': sample_valid,
    }
