import numpy
import numpy as np
import matplotlib.pyplot as plt


################################################################
#             DO NOT EDIT THIS HELPER FUNCTION                 #
################################################################

def load_image(path):
    return plt.imread(path)


################################################################

def display_image(img):
    """ Show an image with matplotlib

    Args:
        img: Image as numpy array (H,W,3)
    """
    #
    # You code here
    print(img.shape)
    plt.imshow(img)
    plt.show()
    #


def save_as_npy(path, img):
    """ Save the image array as a .npy file

    Args:
        img: Image as numpy array (H,W,3)
    """
    #
    # You code here
    numpy.save(path, img)
    #


def load_npy(path):
    """ Load and return the .npy file

    Args:
        path: Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """
    #
    # You code here
    res = numpy.load(path)
    return res
    #


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image

    Args:
        img: Image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """
    #
    # You code here
    # horizontally mirrored: rotate 180 degree and inverse rows
    img2 = img.copy()
    img2 = img.reshape(int(img.size / 3), 3)
    img2 = np.array
    img2 = img2.reshape(img.shape[0], img.shape[1], img.shape[2])
    return img2[::-1]
    #


def display_images(img1, img2):
    """ Display the normal and the mirrored image in one plot

    Args:
        img1: First image to display
        img2: Second image to display
    """
    #
    # You code here
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()
    #
