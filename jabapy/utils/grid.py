import numpy as np
import numba


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

#### Basic diagonistic direct binning method ####
@numba.njit(parallel=True, fastmath=True)
def _bin_particles_direct(pos, qty, mins, maxs, dims):
    N, D = pos.shape
    d = dims.shape[0]

    # cell sizes
    dx = (maxs - mins) / dims

    # strides for flattening
    strides = np.empty(d, dtype=np.int64)
    strides[0] = 1
    for i in range(1, d):
        strides[i] = strides[i-1] * dims[i-1]

    total_size = strides[d-1] * dims[d-1]
    grid = np.zeros(total_size)

    for n in numba.prange(N):
        linear_index = 0
        valid = True

        for k in range(d):
            coord = pos[n, k]
            idx = int((coord - mins[k]) / dx[k])

            if idx < 0 or idx >= dims[k]:
                valid = False
                break

            linear_index += idx * strides[k]

        if valid:
            grid[linear_index] += qty[n]

    return grid

def bin_particles_direct(pos, qty, mins=None, maxs=None, dims=None):
    """
    General N-dimensional direct binning algorithm.
    An extremely quick O(N) way to interpolate particles onto a Cartesian grid.
    Suitable in the limit of small smoothing length / high density relative to grid size,
    but otherwise will be noisy and not very accurate. 

    Parameters
    ----------
    pos  : (N, D) array of particle positions
    qty  : (N,) quantity
    mins : (d,) lower bounds (DEFAULT: mins of each dimension)
    maxs : (d,) upper bounds (DEFAULT: maxes of each dimension)
    dims : (d,) number of bins per dimension (DEFAULT: 512x512 image)
    """
    pos = np.asarray(pos)
    qty = np.asarray(qty)
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

    assert len(pos.shape) == 2, 'Invalid pos format.'
    assert len(qty.shape) == 1, 'Invalid qty format.'
    assert len(mins.shape) == 1, 'Invalid mins format.'
    assert len(maxs.shape) == 1, 'Invalid maxsformat.'
    assert len(dims.shape) == 1, 'Invalid dims format.'
    assert pos.shape[0] == qty.shape[0], 'Not matching number of particles in pos and qty.'
    assert mins.shape[0] == dims.shape[0], 'Not matching mins and dims shape.'
    assert maxs.shape[0] == dims.shape[0], 'Not matching maxs and dims shape.'
    assert dims.shape[0] <= pos.shape[1], 'Grid dimensions cannot exceed number of particle position dimensions.'

    return _bin_particles_direct(pos, qty, mins, maxs, dims).reshape(tuple(dims))


#### Scatter-splat method for SPH ####
# -> numba version of cython implementation in pynbody (tends to be slower than cython version)
# -> TODO: implement this in cython myself to avoid importing all of pynbody
@numba.njit(parallel=True, fastmath=True)
def _bin_particles_scatter(pos, qty, h, mins, maxs, dims):
    N, D = pos.shape
    d = dims.shape[0]

    dx = (maxs - mins) / dims

    # strides
    strides = np.empty(d, dtype=np.int64)
    strides[0] = 1
    for i in range(1, d):
        strides[i] = strides[i-1] * dims[i-1]

    total_size = strides[d-1] * dims[d-1]
    nthreads = numba.get_num_threads()
    thread_grids = np.zeros((nthreads, total_size))

    for n in numba.prange(N):
        tid = numba.get_thread_id()
        grid = thread_grids[tid]

        x = pos[n, 0]
        y = pos[n, 1]
        x_idx = int((x - mins[0]) / dx[0])
        y_idx = int((y - mins[1]) / dx[1])

        if x_idx < 0 or x_idx >= dims[0] or y_idx < 0 or y_idx >= dims[1]:
            continue

        h_i = h[n]

        # support radius = 2h (SPH kernel)
        r_support = 2.0 * h_i

        # compute bounding box
        x_min = mins[0]
        y_min = mins[1]

        x_start = int((pos[n,0] - r_support - x_min) / dx[0])
        x_stop  = int((pos[n,0] + r_support - x_min) / dx[0])
        y_start = int((pos[n,1] - r_support - y_min) / dx[1])
        y_stop  = int((pos[n,1] + r_support - y_min) / dx[1])

        # clamp
        if x_start < 0: x_start = 0
        if y_start < 0: y_start = 0
        if x_stop > dims[0]: x_stop = dims[0]
        if y_stop > dims[1]: y_stop = dims[1]

        norm = 10.0 / (7.0 * np.pi * h_i * h_i)  # 2D cubic spline normalization

        if (2.0 * h_i / dx[0] < 0.5) and (2.0 * h_i / dx[1] < 0.5):
            i = x_idx
            j = y_idx

            x_center = mins[0] + (i + 0.5) * dx[0]
            y_center = mins[1] + (j + 0.5) * dx[1]

            dxp = x - x_center
            dyp = y - y_center
            r = np.sqrt(dxp * dxp + dyp * dyp)
            q = r / h_i

            idx_lin = i * strides[0] + j * strides[1]
            if q < 2.0:
                grid[idx_lin] += qty[n] * _cubic_kernel(q) * norm
            continue

        for i in range(x_start, x_stop):
            x_center = mins[0] + (i + 0.5)*dx[0]

            for j in range(y_start, y_stop):
                y_center = mins[1] + (j + 0.5)*dx[1]

                # distance
                dxp = x - x_center
                dyp = y - y_center
                r = np.sqrt(dxp*dxp + dyp*dyp)

                q = r / h_i

                if q < 2.0:
                    w = _cubic_kernel(q) * norm

                    idx_lin = i*strides[0] + j*strides[1]
                    grid[idx_lin] += qty[n] * w

    grid = np.zeros(total_size)
    for t in range(nthreads):
        for i in range(total_size):
            grid[i] += thread_grids[t, i]

    return grid

def bin_particles_scatter(pos, qty, h, mins=None, maxs=None, dims=None):
    pos = np.asarray(pos)
    qty = np.asarray(qty)
    h = np.asarray(h)

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

    return _bin_particles_scatter(pos, qty, h, mins, maxs, dims).reshape(tuple(dims))


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
