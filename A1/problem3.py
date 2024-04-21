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
    # You code here
    translation_matrix = np.eye(4)

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
    # You code here
    # initialize radian
    radian = d * np.pi / 180

    # calculate triangle functions
    cos_d = np.cos(radian)
    sin_d = np.sin(radian)
    # Y rotation matrix
    rotation_matrix = np.array([[cos_d, 0, sin_d, 0],
                                [0, 1, 0, 0],
                                [-sin_d, 0, cos_d, 0],
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
    # You code here
    radian = d * np.pi / 180

    # calculate triangle functions
    cos_d = np.cos(radian)
    sin_d = np.sin(radian)
    # X rotation matrix
    rotation_matrix = np.array([[1, 0, 0, 0],
                                [0, cos_d, -sin_d, 0],
                                [0, sin_d, cos_d, 0],
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
    # You code here
    radian = d * np.pi / 180

    # calculate triangle functions
    cos_d = np.cos(radian)
    sin_d = np.sin(radian)
    # X rotation matrix
    rotation_matrix = np.array([[cos_d, -sin_d, 0, 0],
                                [sin_d, cos_d, 0, 0],
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
    # You code here

    # principal point initialization
    px, py = principal

    # central projection matrix initialization
    K = np.array([[focal, 0, px, 0],
                  [0, focal, py, 0],
                  [0, 0, 1, 0]])

    return K
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
    # You code here
    rotation = np.dot(np.dot(Rz, Ry), Rx)
    # extrinsic transformation matrix
    M = np.dot(T, rotation)
    # projection matrix
    P = np.dot(L, M)
    return P, M
    #


def cart2hom(points):
    """ Transforms from cartesian to homogeneous coordinates.

    Args:
        points: a np array of points in cartesian coordinates

    Returns:
        A np array of points in homogeneous coordinates
    """
    #
    # You code here
    # initialization
    number_of_point, dimension = points.shape
    # add ones to every column
    homogeneous = np.vstack((points, np.ones((number_of_point, 1))))
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
    # You code here
    # cartesian transformation
    cartesian = points[:-1] / points[-1]
    return cartesian
    #


def loadpoints(path):
    """ Load 2d points from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array
    """
    #
    # You code here
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
    # You code here
    z = np.load(path)
    return z
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
    # You code here
    # parameters initialization
    f = L[0][0]
    px = L[0][2]
    py = L[1][2]
    x_img = P2d[0]
    y_img = P2d[1]
    # transformed parameters by focal point offset
    x = (x_img - px) * z / f
    y = (y_img - py) * z / f
    point_cartesian = np.stack((x, y, z), axis=0)
    return point_cartesian
    #


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
    # You code here
    rotation = M[:3, :3]  # rotation matrix
    rotation_inv = np.matrix(rotation).I
    translation = M[:3, 3].reshape((3, 1))  # translation matrix reshape

    # inverse translation
    for w in range(len(P3d[0])):
        P3d[0][w] = P3d[0][w] - translation[0]
    for w in range(len(P3d[1])):
        P3d[1][w] = P3d[1][w] - translation[1]
    for w in range(len(P3d[2])):
        P3d[2][w] = P3d[2][w] - translation[2]

    # inverse rotation and convert into homogeneous coordinates
    return np.append(np.array(rotation_inv * P3d), [np.ones(len(P3d[0]))], axis=0)


def projectpoints(P, X):
    """ Apply full projection matrix P to 3D points X in cartesian coordinates.

    Args:
        P: projection matrix
        X: 3d points in cartesian coordinates

    Returns:
        x: 2d points in cartesian coordinates
    """
    #
    # You code here

    # cartesian to homogeneous coordinates
    m = np.append(X, [np.ones(len(X[0]))], axis=0)
    # transformation
    m = np.dot(P, m)

    # homogeneous to cartesian coordinates
    return hom2cart(m)


def p3multiplechoice():
    '''
    Change the order of the transformations (translation and rotation).
    Check if they are commutative. Make a comment in your code.
    Return 0, 1 or 2:
    0: The transformations do not commute.
    1: Only rotations commute with each other.
    2: All transformations commute.
    '''
    t = np.array([-27.1, -2.9, -3.2])
    principal_point = np.array([8, -10])
    focal_length = 8

    # model transformations
    T = gettranslation(t)
    Rx = getxrotation(-45)
    Ry = getyrotation(110)
    Rz = getzrotation(120)

    K = getcentralprojection(principal_point, focal_length)

    rotation = np.dot(np.dot(Rz, Ry), Rx)
    # extrinsic transformation matrix
    M2 = np.dot(T, rotation)

    P1, M1 = getfullprojection(T, Rx, Ry, Rz, K)
    P3, M3 = getfullprojection(T, Ry, Rz, Rx, K)

    # the output is 2
    if np.array_equal(M1, M2):
        return 2  # All transformations commute
    elif np.array_equal(M1, M3):
        return 1  # Only rotations commute with each other
    else:
        return 0  # The transformations do not commute
