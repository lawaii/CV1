from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.ndimage import convolve


def load_img(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """
    img = Image.open(path)
    img_array = np.array(img)
    img_array_normalized = img_array / 255.0
    print(img.size)
    return img_array_normalized


def gauss_2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """
    size_x = fsize[0]
    size_y = fsize[1]
    center_x = size_x // 2
    center_y = size_y // 2

    # Create a grid of size x and size y
    x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))

    # The Gaussian function of the filter
    exp = -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
    gauss_filter = np.exp(exp)

    # Normalize the Gaussian filter
    gauss_filter = gauss_filter / np.sum(gauss_filter)
    print("For the gaussian filter: x=%2d y=%2d The filter is done." % (size_x, size_y))
    return gauss_filter


def binomial_2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """
    # Compute the coefficients
    size_x = fsize[0]
    size_y = fsize[1]

    # comb(x,k) k in range 0-N
    # define the function of factorial n!
    def factor(n):
        if n == 0 or n == 1:
            return 1
        return n * factor(n - 1)

    # define the comb function
    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return factor(n) // (factor(n - k) * factor(k))

    coefficients_x = [comb(size_x, k) for k in range(size_x + 1)]
    coefficients_y = [comb(size_y, k) for k in range(size_y + 1)]

    # Normalize coefficients
    norm_coefficients_x = coefficients_x / np.sum(coefficients_x)
    norm_coefficients_y = coefficients_y / np.sum(coefficients_y)

    # Firstly, create 1D arrays
    filter_x = np.sqrt(norm_coefficients_x)
    filter_y = np.sqrt(norm_coefficients_y)

    # Create 2D filter by outer product
    binomial_filter = np.outer(filter_y, filter_x)
    print("Binomial_filter created.")
    return binomial_filter


def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """
    # Smooth image with Gaussian filter
    # mode=constant cval=0.0 is used to avoid visual artifacts
    smoothed_image = convolve(img, f, mode='constant', cval=0.0)

    # Downsample the smoothed_image
    downsampled_image = smoothed_image[::2, ::2]
    print("smooth then down sample by 2 done.")
    return downsampled_image


def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """
    # Create zero rows and columns
    x = img.shape[0]
    y = img.shape[1]
    zeros = np.zeros((2 * x, 2 * y))
    # Add original image to zeros
    zeros[::2, ::2] = img
    upsampled_img = zeros

    # Filter the result with the binomial filter
    filtered_image = convolve(upsampled_img, f, mode='constant', cval=0.0)

    # Apply a scale factor of 4
    result_image = 4 * filtered_image

    return result_image


def gaussian_pyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    # Create the list of images
    pyramid = [img]
    # Build the pyramid
    for level in range(1, nlevel):
        # Downsample the previous level
        downsampled = downsample2(pyramid[level - 1], f)
        pyramid.append(downsampled)
    print("Gaussian pyramid done.")
    return pyramid


def laplacian_pyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    # Create the pyramid
    pyramid = []

    # Build the Laplacian pyramid
    for level in range(len(gpyramid) - 1):
        # Upsample the image in gaussian pyramid
        upsampled = upsample2(gpyramid[level + 1], f)

        # Calculate the Laplacian image by subtracting the upsampled image from the current level
        l_image = gpyramid[level] - upsampled
        pyramid.append(l_image)

    # The top level of laplacian pyramid and the gaussian pyramid should be the same image.
    pyramid.append(gpyramid[-1])

    return pyramid


def create_composite_image(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """
    max_height = max(img.shape[0] for img in pyramid)
    print(max_height)
    total_width = sum(img.shape[1] for img in pyramid)

    composite_image = np.zeros((max_height, total_width), dtype=np.float32)

    current_width = 0
    for img in pyramid:
        # normalize
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val)
        height, width = normalized_img.shape

        # Insert the normalized image into the composite image
        composite_image[:height, current_width:current_width + width] = normalized_img.astype(np.float32)
        current_width += width
    print("A composite image is created.")
    print(composite_image.shape)
    return composite_image


def amplify_high_freq(lpyramid, l0_factor=1.01, l1_factor=1.01):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """
    # Amplify frequencies of 2 layers of Laplacian pyramid
    amplified_pyramid = deepcopy(lpyramid)
    amplified_pyramid[-1] *= l0_factor
    amplified_pyramid[-2] *= l1_factor
    return amplified_pyramid


def reconstruct_image(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """
    # To obtain the reconstructed image, we add the image in this level of the pyramid to upsampled image in the next
    # level
    result_img = lpyramid[-1]
    for level in reversed(lpyramid[:-1]):
        result_img = upsample2(result_img, f)
        result_img += level
    return result_img
