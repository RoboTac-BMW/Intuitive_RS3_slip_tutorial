import pybullet as p
import numpy as np
import math
import copy
import cv2


def plane_seg(point_cloud):
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.009,
                                                            ransac_n=3,
                                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = point_cloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([0.0, 0, 1.0])
    return inlier_cloud, outlier_cloud


def compute_object_rotation(rotm):
    euler_ang = rotm2euler(rotm)
    object_orn = [0, 0, euler_ang[2] - np.sign(euler_ang[2]) * math.pi / 2]
    return object_orn


def read_parameters(dbg_params):
    '''Reads values from debug parameters

    Parameters
    ----------
    dbg_params : dict
        Dictionary where the keys are names (str) of parameters and the values are
        the itemUniqueId (int) for the corresponing debug item in pybullet

    Returns
    -------
    dict
        Dictionary that maps parameter names (str) to parameter values (float)
    '''
    values = dict()
    for name, param in dbg_params.items():
        values[name] = p.readUserDebugParameter(param)

    return values


def interactive_camera_placement(camera_id, pos_scale=1.,
                                 max_dist=2.,
                                 show_plot=False,
                                 verbose=False,
                                 ):
    '''GUI for adjusting camera placement in pybullet. Use the scales to adjust
    intuitive parameters that govern view and projection matrix.  When you are
    satisfied, you can hit the print button and the values needed to recreate
    the camera placement will be logged to console.
    In addition to showing a live feed of the camera, there are also two visual
    aids placed in the simulator to help understand camera placement: the target
    position and the camera. These are both shown as red objects and can be
    viewed using the standard controls provided by the GUI.
    Note
    ----
    There must be a simulator running in GUI mode for this to work
    Parameters
    ----------
    pos_scale : float
        Position scaling that limits the target position of the camera.
    max_dist : float
        Maximum distance the camera can be away from the target position, you
        may need to adjust if you scene is large
    show_plot : bool, default to True
        If True, then a matplotlib window will be used to plot the generated
        image.  This is beneficial if you want different values for image width
        and height (since the built in image visualizer in pybullet is always
        square).
    verbose : bool, default to False
        If True, then additional parameters will be printed when print button
        is pressed.
    '''
    np.set_printoptions(suppress=True, precision=4)

    dbg = dict()
    # for view matrix
    dbg['camera_x'] = p.addUserDebugParameter('camera_x', -pos_scale, pos_scale, 0)
    dbg['camera_y'] = p.addUserDebugParameter('camera_y', -pos_scale, pos_scale, 0.85)
    dbg['camera_z'] = p.addUserDebugParameter('camera_z', -pos_scale, pos_scale, 0.3)

    dbg['print'] = p.addUserDebugParameter('print params', 1, 0, 1)
    old_print_val = 1
    while True:
        values = read_parameters(dbg)
        eye_pos = np.array([values[f'camera_{c}'] for c in 'xyz'])
        view_mtx = p.computeViewMatrix(cameraEyePosition=eye_pos,
                                                     cameraTargetPosition=[0, 0.3, 0.0],
                                                     cameraUpVector=[0, -1, 0])
        aspect = 480 / 480
        proj_mtx = p.computeProjectionMatrixFOV(90,
                                                 aspect,
                                                 0.01,
                                                 2)

        # update visual aid for camera
        view_mtx = np.array(view_mtx).reshape((4, 4), order='F')
        T_world_cam = np.linalg.inv(view_mtx)
        cam_orn = p.getQuaternionFromEuler(rotm2euler(T_world_cam[:3, :3]).tolist())
        cam_pos = T_world_cam[:3, 3]
        p.resetBasePositionAndOrientation(camera_id, cam_pos, cam_orn)
        view_mtx = view_mtx.reshape(-1, order='F')
        img = p.getCameraImage(480, 480, view_mtx, proj_mtx)[2]
        if old_print_val != values['print']:
            old_print_val = values['print']
            print("[x, y, z]: ", eye_pos)

# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Change Vector (Rotation) from Global Frame to Desired Frame
def changeFrame(vector, orn_des):
    sRw = euler2rotm(orn_des)
    wRs = np.linalg.inv(sRw)
    p = np.array([[vector[0]], [vector[1]], [vector[2]]])
    p_dash = np.matmul(wRs, p)

    return p_dash


# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotm2euler(R):
    assert (isRotm(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def angle2rotm(angle_axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke
    angle = angle_axis[0]
    axis = angle_axis[1:]

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]], dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M[:3, :3]


def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01  # Margin to allow for rounding errors
    epsilon2 = 0.1  # Margin to distinguish between 0 and 180 degrees

    assert (isRotm(R))

    if ((abs(R[0][1] - R[1][0]) < epsilon) and (abs(R[0][2] - R[2][0]) < epsilon) and (
            abs(R[1][2] - R[2][1]) < epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1] + R[1][0]) < epsilon2) and (abs(R[0][2] + R[2][0]) < epsilon2) and (
                abs(R[1][2] + R[2][1]) < epsilon2) and (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0]  # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if ((xx > yy) and (xx > zz)):  # R[0][0] is the largest diagonal term
            if (xx < epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif (yy > zz):  # R[1][1] is the largest diagonal term
            if (yy < epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2][2] is the largest diagonal term so base result on this
            if (zz < epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]  # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt(
        (R[2][1] - R[1][2]) * (R[2][1] - R[1][2]) + (R[0][2] - R[2][0]) * (R[0][2] - R[2][0]) + (R[1][0] - R[0][1]) * (
                R[1][0] - R[0][1]))  # used to normalise
    if (abs(s) < 0.001):
        s = 1

        # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]


# Convert from URx rotation format to axis angle
def urx2angle(v):
    angle = np.linalg.norm(v)
    axis = v / angle
    return np.insert(axis, 0, angle)


# Convert from angle axis format to URx rotation format
def angle2urx(angle_axis):
    angle_axis[1:4] = angle_axis[1:4] / np.linalg.norm(angle_axis[1:4])
    return angle_axis[0] * np.asarray(angle_axis[1:4])


# Convert from quaternion (w,x,y,z) to rotation matrix
def quat2rotm(quat):
    # From: Christoph Gohlke

    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quat, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


# Convert from rotation matrix to quaternion (w,x,y,z)
def rotm2quat(matrix, isprecise=False):
    # From: Christoph Gohlke

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


# Estimate rigid transform with SVD (from Nghia Ho)
def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0];  # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:  # Special reflection case
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    t.shape = (3, 1)
    return np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)


# Get nearest nonzero pixel
def nearest_nonzero_pix(img, y, x):
    r, c = np.nonzero(img)
    min_idx = ((r - y) ** 2 + (c - x) ** 2).argmin()
    return r[min_idx], c[min_idx]


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# modified by xzj
def transform_points(pts, transform):
    # pts = [3xN] array
    # trasform: [3x4]
    pts_t = np.dot(transform[0:3, 0:3], pts) + np.tile(transform[0:3, 3:], (1, pts.shape[1]))
    return pts_t


def get_mask(image_ini, image_RGB):
    h_ini, _, _ = cv2.split(cv2.cvtColor(image_ini, cv2.COLOR_RGB2HSV))
    h_cur, _, _ = cv2.split(cv2.cvtColor(image_RGB, cv2.COLOR_RGB2HSV))
    tmp = np.abs(h_ini - h_cur)
    tmp = np.minimum(tmp, 360 - tmp)

    mask = (tmp > 30).astype(np.int)
    mask[:, :150] = 0
    return mask


def get_length(vec):
    return np.sqrt((vec[0] ** 2) + (vec[1] ** 2))


def get_cos(vec0, vec1):
    return (vec0[0] * vec1[0] + vec0[1] * vec1[1]) / get_length(vec0) / get_length(vec1)


def invRt(Rt):
    # RtInv = [Rt(:,1:3)'  -Rt(:,1:3)'* Rt(:,4)];
    invR = Rt[0:3, 0:3].T
    invT = -1 * np.dot(invR, Rt[0:3, 3])
    invT.shape = (3, 1)
    RtInv = np.concatenate((invR, invT), axis=1)
    RtInv = np.concatenate((RtInv, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    return RtInv


def link_to_idx(body_id):
    d = {p.getBodyInfo(body_id)[0].decode('UTF-8'): -1, }

    for _id in range(p.getNumJoints(body_id)):
        _name = p.getJointInfo(body_id, _id)[12].decode('UTF-8')
        d[_name] = _id

    return d


def joint_to_idx(body_id):
    return {key.decode(): value for (value, key) in
            [p.getJointInfo(body_id, i)[:2] for i in range(p.getNumJoints(body_id))]}
