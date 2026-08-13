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

def cubic_kernel(q):
    return (q < 1.0)*(1.0 - 1.5*q*q + 0.75*q*q*q) + (1.0 < q)*(q < 2.0)*0.25*(2.0 - q)**3
    
@numba.njit(fastmath=True)
def _cubic_kernel(q):
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


def bin_particles_direct(pos, *qty, mins=None, maxs=None, dims=None, ret_bins=False, nthreads=None, _npart_multithread_threshold=1e6):
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
    
    If D > d, the extra dimensions of pos are ignored. If D < d, an error is raised.
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
    grid = np.zeros((M, total_size), dtype=np.float64, order='C')
    if nthreads == 1:
        _bin_particles_direct_serial(_pos, _qty, mins, maxs, dims, strides, grid)
    else:
        with numba_threads(nthreads):
            _bin_particles_direct_parallel(_pos, _qty, mins, maxs, dims, strides, grid, nthreads)
    grid = grid.reshape(tuple(dims)) if M == 1 else grid.reshape((M,) + tuple(dims))

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
