import warnings

import numpy as np

##########################################
######## Translation and Rotation ########
##########################################

def rotation_matrix_transform_a_to_b(a, b, already_well_behaved=False):
    if already_well_behaved: # normalized, non-zero 3d
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != (3,):
            raise ValueError('Vector a must be 3d.')
        if b.shape != (3,):
            raise ValueError('Vector b must be 3d.')
        if np.all(a == 0):
            raise ValueError('Vector a must be non-zero.')
        if np.all(b == 0):
            raise ValueError('Vector b must be non-zero.')
        a = a/np.sqrt(np.sum(a*a))
        b = b/np.sqrt(np.sum(b*b))
    vx, vy, vz = np.cross(a, b)
    s2 = vx*vx + vy*vy + vz*vz
    if np.isclose(s2, 0):
        return np.eye(3)
    c = np.dot(a, b)
    W = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]) 
    return np.eye(3) + W + np.dot(W, W)*(1-c)/s2

def rotation_matrix_transform_a_to_b_and_c_to_d(a, b, c, d, already_well_behaved=False):
    if not already_well_behaved: # normalized, non-zero 3d, a&c perpendicular and b&d perpendicular
        a, b, c, d = np.asarray(a), np.asarray(b), np.asarray(c), np.asarray(d)
        if a.shape != (3,):
            raise ValueError('Vector a must be 3d.')
        if b.shape != (3,):
            raise ValueError('Vector b must be 3d.')
        if c.shape != (3,):
            raise ValueError('Vector c must be 3d.')
        if d.shape != (3,):
            raise ValueError('Vector d must be 3d.')
        if np.all(a == 0):
            raise ValueError('Vector a must be non-zero.')
        if np.all(b == 0):
            raise ValueError('Vector b must be non-zero.')
        if np.all(c == 0):
            raise ValueError('Vector c must be non-zero.')
        if np.all(d == 0):
            raise ValueError('Vector d must be non-zero.')
        a = a/np.sqrt(np.sum(a*a))
        b = b/np.sqrt(np.sum(b*b))
        c = c/np.sqrt(np.sum(c*c))
        d = d/np.sqrt(np.sum(d*d))
        if np.isclose(np.abs(np.dot(a, c)), 0):
            raise ValueError('The vectors a and c should be perpendicular.')
        if np.isclose(np.abs(np.dot(b, d)), 0):
            raise ValueError('The vectors b and d should be perpendicular.')
    R1 = rotation_matrix_transform_a_to_b(a, b, already_well_behaved=True)
    R2 = rotation_matrix_transform_a_to_b(R1 @ c, d, already_well_behaved=True)
    return R2 @ R1

def rotation_matrix_to(xdir=None, ydir=None, zdir=None):
    """Rotation matrix in Cartesian coordinate space to an orientation where the x, y and z directions are set."""
    # if all none entered
    if xdir is None and ydir is None and zdir is None:
        return np.eye(3)
    # make vectors well behaved
    if xdir is not None:
        xdir = np.asarray(xdir)
        if xdir.shape != (3,):
            raise ValueError('If entered, xdir must be 3d.')
        if np.all(xdir == 0):
            raise ValueError('If entered, xdir must be non-zero.')
        xdir = xdir/np.sqrt(np.sum(xdir*xdir))
        xdir2 = np.array([1.0, 0.0, 0.0])
    if ydir is not None:
        ydir = np.asarray(ydir)
        if ydir.shape != (3,):
            raise ValueError('If entered, ydir must be 3d.')
        if np.all(ydir == 0):
            raise ValueError('If entered, ydir must be non-zero.')
        ydir = ydir/np.sqrt(np.sum(ydir*ydir))
        ydir2 = np.array([0.0, 1.0, 0.0])
    if zdir is not None:
        zdir = np.asarray(zdir)
        if zdir.shape != (3,):
            raise ValueError('If entered, zdir must be 3d.')
        if np.all(zdir == 0):
            raise ValueError('If entered, zdir must be non-zero.')
        zdir = zdir/np.sqrt(np.sum(zdir*zdir))
        zdir2 = np.array([0.0, 0.0, 1.0])
    # if all three entered
    if xdir is not None and ydir is not None and zdir is not None:
        xcrossy = np.cross(xdir, ydir)
        if np.allclose(xcrossy - zdir, 0) and np.dot(xcrossy, zdir) >= 0:
             raise ValueError('If you specify all of xdir, ydir or zdir, they must all be orthogonal and follow the right-hand rule.')
        return rotation_matrix_transform_a_to_b_and_c_to_d(xdir, xdir2, ydir, ydir2, already_well_behaved=True)
    # if two entered
    if xdir is not None and ydir is not None:
        if np.isclose(np.abs(np.sum(xdir*ydir)), 0):
            raise ValueError('If both entered, xdir and ydir should be perpendicular.')
        return rotation_matrix_transform_a_to_b_and_c_to_d(xdir, xdir2, ydir, ydir2, already_well_behaved=True)
    if ydir is not None and zdir is not None:
        if np.isclose(np.abs(np.sum(ydir*zdir)), 0):
            raise ValueError('If both entered, ydir and zdir should be perpendicular.')
        return rotation_matrix_transform_a_to_b_and_c_to_d(ydir, ydir2, zdir, zdir2, already_well_behaved=True)
    if xdir is not None and zdir is not None:
        if np.isclose(np.abs(np.sum(xdir*zdir)), 0):
            raise ValueError('If both entered, xdir and zdir should be perpendicular.')
        return rotation_matrix_transform_a_to_b_and_c_to_d(xdir, xdir2, zdir, zdir2, already_well_behaved=True)
    # if only one entered
    if xdir is not None:
        return rotation_matrix_transform_a_to_b(xdir, xdir2, already_well_behaved=True)
    if ydir is not None:
        return rotation_matrix_transform_a_to_b(ydir, ydir2, already_well_behaved=True)
    if zdir is not None:
        return rotation_matrix_transform_a_to_b(zdir, zdir2, already_well_behaved=True)
    assert False, "Code should not reach this point.."

def rotation_matrix_by(z=None, y=None, x=None, in_radians=False):
    """Rotation in Cartesian coords along a given direction in degrees (unless in_radians=True). Order of operations is z, then y, then x."""
    if z is None and y is None and x is None:
        return np.eye(3)
    if z is not None:
        z = np.asarray(z) if in_radians else np.asarray(z)*np.pi/180.0
    if y is not None:
        y = np.asarray(y) if in_radians else np.asarray(y)*np.pi/180.0
    if x is not None:
        x = np.asarray(x) if in_radians else np.asarray(x)*np.pi/180.0
    R_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0],  [0, 0, 1]]) if z is not None else np.eye(3)
    R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]) if y is not None else np.eye(3)
    R_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]) if x is not None else np.eye(3)
    return R_x @ R_y @ R_z

def general_rotation_matrix(z=None, y=None, x=None, zdir=None, ydir=None, xdir=None, in_radians=False, strictness=1):
    """
    General rotation in Cartesian coordinates. 
    You can set the directions for the new coordinate axes AND/OR the rotation by set angles in degrees.
    Rotation will first be aligned to these axes and then rotated in z, then in y, then in x.
    A warning will be given if order matters and there is a chance that it could be isinterpreted.
    Adjust the strictness=0 to get rid of this warning, or =2 if you want it to explicitly error when this happens.
    """
    rotate_by = (z is not None) or (y is not None) or (x is not None)
    rotate_to = (zdir is not None) or (ydir is not None) or (xdir is not None)
    if strictness > 0 and rotate_to and rotate_by:
        if strictness <= 1:
            warnings.warn('You are attempting to rotate using both aligning to an axis and by set angle(s) in degrees.\nPlease note that the alignment will happen first, such that the angle rotation will be defined relative to those new axes.\nYou can set strictness<=0 to avoid this message in the future.')
        else:
            raise ValueError('You are attempting to rotate using both aligning to an axis and by set angle(s), which is not allowed under this level of strictness.\nYou can set strictness<=1 to avoid this error in the future.')
    if strictness > 0 and ((z is not None and y is not None) or (y is not None and x is not None) or (z is not None and x is not None)):
        if strictness <= 1:
            warnings.warn('You are attempting to rotate by multiple set angles in degrees.\nPlease note that the rotation will happen starting with z, then y, then x rotation.\nYou can set strictness<=0 to avoid this message in the future.')
        else:
            raise ValueError('You are attempting to rotate by multiple set angles, which is not allowed under this level of strictness.\nYou can set strictness<=1 to avoid this error in the future.')
    R_by = rotation_matrix_by(z=z, y=y, x=x, in_radians=in_radians)
    R_to = rotation_matrix_to(zdir=zdir, ydir=ydir, xdir=xdir)
    return R_by @ R_to


def translate_and_rotate_vectors(vectors, center=None, z=None, y=None, x=None, zdir=None, ydir=None, xdir=None, in_radians=False, strictness=2):
    """Translates and rotates vectors in Cartesian coordinate space."""
    if center is not None:
        vectors = vectors - center
    M = general_rotation_matrix(z=z, y=y, x=x, zdir=zdir, ydir=ydir, xdir=xdir, in_radians=in_radians, strictness=strictness)
    if M is None:
        return vectors
    return np.einsum('ij,kj->ki', M, vectors)
    

TRANSFORM_LIKE_OPTIONS = {'scalar', 'vector', '2-tensor', 'pseudovector'}
def transform_general_tensor(data, center=None, T=None, transforms_like='vector', reverse_order=False):
    if center is not None and not reverse_order:
        data -= center
    if transforms_like != 'scalar' and T is not None:
        TT = np.transpose(T)
        if transforms_like == 'vector':
            data = data @ TT
        elif transforms_like == '2-tensor':
            data = T @ data @ TT 
        elif transforms_like == 'pseudovector': # only affects flips
            data = np.linalg.det(T) * data @ TT
        else:
            raise ValueError('transforms_like must be in TRANSFORM_LIKE_OPTIONS.')
    if center is not None and reverse_order:
        data -= center
    return data



##########################################
########### Random Orientations ##########
##########################################

def random_cartesian_directions(N=1):
    phi = np.random.uniform(0, 2*np.pi, N)
    costheta = np.random.uniform(-1, 1, N)
    sintheta = np.sqrt(1 - costheta*costheta)
    return np.array([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta]).T