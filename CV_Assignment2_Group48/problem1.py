import numpy as np
from scipy.ndimage import convolve


def generate_image():
    """ Generates cocentric simulated image in Figure 1.

    Returns:
        Concentric simulated image with the size (210, 210) with increasing intensity through the center
        as np.array.
    """
    #
    # constant parameters
    square_layers = 7
    mask_intensity = 30.0
    size_increment = 15
    size_layer = 30

    # range for adding intensity, iterated over the layers
    start = 0
    end = square_layers * size_increment

    # initialize the image for total dark
    image_size = size_layer * square_layers
    image = np.zeros((image_size, image_size))  # black background image for 210*210

    # generating picture
    for i in range(1, square_layers):
        # making the mask for the next inner layer
        start = size_increment * i - 1
        end = 210 - start - 1
        mask = np.ones((end - start - 1, end - start - 1)) * mask_intensity
        image[start + 1:end, start + 1:end] += mask

    # Display or save the image
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    return image
    #


def sobel_edge(img):
    """ Applies sobel edge filter on the image to obtain gradients in x and y directions and gradient map.
    (see lecture 5 slide 30 for filter coefficients)

    Args:
        img: image to be convolved
    Returns:
        Ix derivatives of the source image in x-direction as np.array
        Iy derivatives of the source image in y-direction as np.array
        Ig gradient magnitude map computed by sqrt(Ix^2+Iy^2) for each pixel
    """
    #

    # sobel operators for both side
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    # gradients in x and y directions
    Ix = convolve(img, sobel_x, mode='constant', cval=0)
    Iy = convolve(img, sobel_y, mode='constant', cval=0)
    Ig = np.sqrt(Ix ** 2 + Iy ** 2)

    return Ix, Iy, Ig
    #


def detect_edges(grad_map, threshold=15):
    """ Applies threshold on the edge map to detect edges.

    Args:
        grad_map: gradient map.
        threshold: threshold to be applied.
    Returns:
        edge_map: thresholded gradient map.
    """
    #
    # edges are marked as white(255) while the inside parts are marked as black(0)
    edges = np.where(grad_map > threshold, 255, 0)
    return edges
    #
    # In addition to the code, please include your response as a comment to
    # the following questions: Which threshold recovers the edge map of the
    # original image when working with the noisy image? How did you
    # determine this threshold value, and why did you choose it?
    # Threshold = 120-3*15 = 75
    # How?  it is 3 times of the given Gaussian noise's standard deviation
    # Why?  according to the gaussian distribution, when the variable x is greater than  3 times of the variance, 99.7%
    #       of the whole distribution will be concluded in the distribution
    #       that means, 99.7% of the noise influence could be filtered within this threshold.
    #       then the grad map has 120 gray level at the edges, so it is actually mean=120,
    #       and the threshold should be 120-3*15, because the standard deviation(sqrt(15)) in both directions should
    #       multiply and therefore the threshold is 75.


def add_noise(img, mean=0, variance=15):
    """ Applies Gaussian noise on the image.

    Args:
        img: image in np.array
        mean: mean of the noise distribution.
        variance: variance of the noise distribution.
    Returns:
        noisy_image: gaussian noise applied image.
    """
    #
    # generate Gaussian noise
    noise = np.random.normal(mean, np.sqrt(variance), np.shape(img))

    # add noise to the image
    noisy_img = noise + img
    return noisy_img
    #
