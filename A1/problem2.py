import numpy
import numpy as np
import scipy.ndimage
from scipy.ndimage import convolve


def loadbayer(path):
    """ Load data from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array (H,W)
    """
    #
    # You code here
    res = numpy.load(path)

    return res
    #


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        bayerdata: Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """
    #
    # You code here
    print(np.shape(bayerdata))
    red = []
    green = []
    blue = []
    # missing values are filled with zero by positions
    for i in range(0, len(bayerdata)):
        for j in range(0, len(bayerdata[0])):
            if i % 2 == 0 and j % 2 == 1:
                red.append(bayerdata[i][j])
                green.append(0)
                blue.append(0)
            elif i % 2 == 1 and j % 2 == 0:
                red.append(0)
                green.append(0)
                blue.append(bayerdata[i][j])
            else:
                red.append(0)
                green.append(bayerdata[i][j])
                blue.append(0)
    # reshape RGB matrix to the objective format
    red = np.array(red).reshape(len(bayerdata[0]), len(bayerdata[0]))
    green = np.array(green).reshape(len(bayerdata[0]), len(bayerdata[0]))
    blue = np.array(blue).reshape(len(bayerdata[0]), len(bayerdata[0]))

    return red, green, blue

    #


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        r: red channel as numpy array (H,W)
        g: green channel as numpy array (H,W)
        b: blue channel as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    #
    # You code here
    # initialize a new list for different channels
    res = np.zeros(len(r[0]) * len(r[0]) * 3)
    # reshape different color matrix to one matrix
    res = np.array(res).reshape(len(r[0]), len(r[0]), 3)
    for i in range(0, len(r)):
        for j in range(0, len(r[0])):
            res[i][j][0] = r[i][j]
            res[i][j][1] = g[i][j]
            res[i][j][2] = b[i][j]

    return res
    #


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        r: red channel as numpy array (H,W)
        g: green channel as numpy array (H,W)
        b: blue channel as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """
    #
    # You code here
    # use a 3*3 kernel
    # matrix parameters: https://www.sfu.ca/~gchapman/e895/e895l11.pdf
    k_red_blue = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])
    k_green = np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])
    r = scipy.ndimage.convolve(r, k_red_blue, mode='nearest')
    g = scipy.ndimage.convolve(g, k_green, mode='nearest')
    b = scipy.ndimage.convolve(b, k_red_blue, mode='nearest')

    # initialize a result matrix and reshape it
    res = np.zeros(len(r[0]) * len(r[0]) * 3)
    res = np.array(res).reshape(len(r[0]), len(r[0]), 3)

    for i in range(0, len(r)):
        for j in range(0, len(r[0])):
            res[i][j][0] = r[i][j]
            res[i][j][1] = g[i][j]
            res[i][j][2] = b[i][j]

    return res
    #
