import numpy as np
import matplotlib.pyplot as plt


################################################################
#            DO NOT EDIT THESE HELPER FUNCTIONS                #
################################################################

# Plot 2D points
def displaypoints2d(points):
    plt.figure()
    plt.plot(points[0, :], points[1, :], '.b')
    plt.xlabel('Screen X')
    plt.ylabel('Screen Y')

# Plot 3D points
def displaypoints3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0, :], points[1, :], points[2, :], 'b')
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")


################################################################


def gettranslation(v):
    """ Returns translation matrix T in homogeneous coordinates
    for translation by v.

    Args:
        v: 3d translation vector

    Returns:
        Translation matrix in homogeneous coordinates
    """
    #
    # unit matrix
    translation_matrix = np.eye(4)

    # write translation part =v
    translation_matrix[:3, 3] = v

    return translation_matrix
    #


def getyrotation(d):
    """ Returns rotation matrix Ry in homogeneous coordinates for
    a rotation of d degrees around the y axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """
    #
    # degree 2 radian
    rad = d * np.pi / 180.0

    # calculate (cos) and (sin)
    cos_value = np.cos(rad)
    sin_value = np.sin(rad)
    # Y rotation matrix
    rotation_matrix = np.array([[cos_value, 0, sin_value, 0],
                                [0, 1, 0, 0],
                                [-sin_value, 0, cos_value, 0],
                                [0, 0, 0, 1]])

    return rotation_matrix
    #


def getxrotation(d):
    """ Returns rotation matrix Rx in homogeneous coordinates for a
    rotation of d degrees around the x axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """
    #
    # degree 2 radian
    rad = d * np.pi / 180.0

    # calculate (cos) and (sin)
    cos_value = np.cos(rad)
    sin_value = np.sin(rad)
    # X rotation matrix
    rotation_matrix = np.array([[1, 0, 0, 0],
                                [0, cos_value, -sin_value, 0],
                                [0, sin_value, cos_value, 0],
                                [0, 0, 0, 1]])

    return rotation_matrix
    #


def getzrotation(d):
    """ Returns rotation matrix Rz in homogeneous coordinates for a
    rotation of d degrees around the z axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """
    #
    # degree 2 radian
    rad = d * np.pi / 180.0

    # calculate (cos) and (sin)
    cos_value = np.cos(rad)
    sin_value = np.sin(rad)
    # Z rotation matrix
    rotation_matrix = np.array([[cos_value, -sin_value, 0, 0],
                                [sin_value, cos_value, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    return rotation_matrix
    #


def getcentralprojection(principal, focal):
    """ Returns the (3 x 4) matrix L that projects homogeneous camera
    coordinates on homogeneous image coordinates depending on the
    principal point and focal length.

    Args:
        principal: the principal point, 2d vector
        focal: focal length

    Returns:
        Central projection matrix
    """
    #

    # principal point
    cx, cy = principal

    # central projection matrix
    L = np.array([[focal, 0, cx, 0],
                  [0, focal, cy, 0],
                  [0, 0, 1, 0]])

    return L
    #


def getfullprojection(T, Rx, Ry, Rz, L):
    """ Returns full projection matrix P and full extrinsic
    transformation matrix M.

    Args:
        T: translation matrix
        Rx: rotation matrix for rotation around the x-axis
        Ry: rotation matrix for rotation around the y-axis
        Rz: rotation matrix for rotation around the z-axis
        L: central projection matrix

    Returns:
        P: projection matrix
        M: matrix that summarizes extrinsic transformations
    """
    #
    rotation_matrix = np.dot(np.dot(Rz, Ry), Rx)
    # extrinsic transformation matrix M
    M = np.dot(T,rotation_matrix)# maybe have problem about the rangfolge

    P = np.dot(L, M)

    return P, M
    # p = 3*4, M = 4*4
    #


def cart2hom(points):
    """ Transforms from cartesian to homogeneous coordinates.

    Args:
        points: a np array of points in cartesian coordinates

    Returns:
        A np array of points in homogeneous coordinates
    """
    #
    number_of_point, dimension = points.shape

    homogeneous = np.vstack((points, np.ones((number_of_point, 1))))  # Add a column of 1 to the right
    return homogeneous
    #


def hom2cart(points):
    """ Transforms from homogeneous to cartesian coordinates.

    Args:
        points: a np array of points in homogenous coordinates

    Returns:
        A np array of points in cartesian coordinates
    """
    #
    # number_of_point, dimension = points.shape

    Cartesian = points[:-1] / points[-1]
    return Cartesian
    #


def loadpoints(path):
    """ Load 2d points from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array
    """
    #
    points_2d = np.load(path)
    return points_2d
    #


def loadz(path):
    """ Load z-coordinates from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array
    """
    #
    z_coordinates = np.load(path)
    return z_coordinates
    #


def invertprojection(L, P2d, z):
    """
    Invert just the projection L of cartesian image coordinates
    P2d with z-coordinates z.

    Args:
        L: central projection matrix
        P2d: 2d image coordinates of the projected points
        z: z-components of the homogeneous image coordinates

    Returns:
        3d cartesian camera coordinates of the points
    """
    #
    f = L[0][0]
    px = L[0][2]
    py = L[1][2]
    x_img = P2d[0]
    y_img = P2d[1]
    x = (x_img-px)*z/f
    y = (y_img-py)*z/f
    point_cartesian = np.stack((x, y, z), axis=0)
    return point_cartesian
    #


# !!!!!!!!!!!!!!!!!!!!!!!!!!!PROBLEM!!!!!!!!!!!!!!!!!!!!!!!!

def inverttransformation(M, P3d):
    """ Invert just the model transformation in homogeneous
    coordinates for the 3D points P3d in cartesian coordinates.

    Args:
        M: matrix summarizing the extrinsic transformations
        P3d: 3d points in cartesian coordinates

    Returns:
        3d points after the extrinsic transformations have been reverted
    """
    #

    # extract from matrix M
    R = M[:3, :3]                   # rotate
    R_inv = np.matrix(R).I
    T = M[:3, 3].reshape((3, 1))     # translation


    T_extend = np.tile(T, (1, 2904))
    P3d_origin= P3d - T_extend
    P3d_original_homogeneous = np.array(R_inv * P3d_origin)

    # add
    number_of_point, dimension = (P3d_original_homogeneous.T).shape
    homogeneous = np.hstack((P3d_original_homogeneous.T, np.ones((number_of_point, 1))))  # Add a column of 1 to the right
    P3d_original_homogeneous = homogeneous.T

    return P3d_original_homogeneous
    #


def projectpoints(P, X):
    """ Apply full projection matrix P to 3D points X in cartesian coordinates.

    Args:
        P: projection matrix, p = 3*4
        X: 3d points in cartesian coordinates

    Returns:
        x: 2d points in cartesian coordinates
    """
    #
    # full projection matrix P, X is point`s real coordinate
    # homogeneous
    row_of_ones = np.ones((1, X.shape[1]))
    homogeneous_matrix = np.vstack((X, row_of_ones))
    #X_homogeneous = np.column_stack((X, np.ones(X.shape[0])))

    # Projection in + direction
    x_2D_homogeneous = np.dot(P, homogeneous_matrix)
    #print(f"this is x_2D_homogeneous : {x_2D_homogeneous}")

    # Cartesian
    x_2D_cartesian = x_2D_homogeneous[:-1] / x_2D_homogeneous[-1]
    return x_2D_cartesian
    #


def p3multiplechoice():
    '''
    Change the order of the transformations (translation and rotation).
    Check if they are commutative. Make a comment in your code.
    Return 0, 1 or 2:
    0: The transformations do not commute.
    1: Only rotations commute with each other.
    2: All transformations commute.
    '''

    # point coordinate (1, 0, 0)
    point = np.array([1, 0, 0])
    # translation (2, 3, 4)
    translation_matrix = np.array([[1, 0, 0, 2],
                                   [0, 1, 0, 3],
                                   [0, 0, 1, 4],
                                   [0, 0, 0, 1]])
    # for x axis
    rotation_x_matrix = np.array([[1, 0, 0, 0],
                                  [0, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                                  [0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                                  [0, 0, 0, 1]])

    # two orders
    result1 = translation_matrix @ (rotation_x_matrix @ point)  # rotation first, then translate
    result2 = (translation_matrix @ rotation_x_matrix) @ point  # translate first, then rotation
    # check
    if np.allclose(result1, result2):
        return 2  # can change position
    elif np.allclose(result1, point) and np.allclose(result2, point):
        return 1  # only the rotation operation produces the same result
    else:
        return 0  # cannot

    print(p3multiplechoice())

    return -1
