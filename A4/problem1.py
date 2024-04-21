import numpy as np
from numpy.linalg import norm


def cost_ssd(patch_l, patch_r):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """
    #
    # You code here
    ssd = np.sum((patch_l - patch_r) ** 2)
    return ssd
    #


def cost_nc(patch_l, patch_r):
    """Compute the normalized correlation cost (NC):

    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array

    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """
    #
    # You code here

    l = np.reshape(patch_l, (-1, 1))
    r = np.reshape(patch_r, (-1, 1))
    mean_l = np.mean(l)
    mean_r = np.mean(r)
    res = np.dot((l - mean_l).transpose(), (r - mean_r)) / (norm(l - mean_l) * norm(r - mean_r))
    return res


def cost_function(patch_l, patch_r, alpha):
    """Compute the cost between two input window patches
    
    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    #
    # You code here
    return 1 / (len(patch_l[0]) ** 2) * cost_ssd(patch_l, patch_r) + alpha * cost_nc(patch_l, patch_r)
    #


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Add padding to the input image based on the window size
    
    Args:
        input_img: input image as 2-dimensional (H,W) numpy array
        window_size: window size as a scalar value (always and odd number)
        padding_mode: padding scheme, it can be 'symmetric', 'reflect', or 'constant'.
            In the case of 'constant' assume zero padding.
        
    Returns:
        padded_img: padded image as a numpy array of the same type as input_img
    """
    #
    # You code here
    #
    # Calculate padding size
    pad_size = window_size // 2

    # Apply padding based on the specified mode
    if padding_mode == 'symmetric':
        padded_img = np.pad(input_img, pad_size, mode='symmetric')
    elif padding_mode == 'reflect':
        padded_img = np.pad(input_img, pad_size, mode='reflect')
    elif padding_mode == 'constant':
        padded_img = np.pad(input_img, pad_size, mode='constant', constant_values=0)
    else:
        return input_img

    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map using the window-based matching strategy    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """
    #
    # You code here
    #
    height, width = padded_img_l.shape
    half_window = window_size // 2
    disparities = np.zeros((height, width), dtype=padded_img_l.dtype)

    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            # Extract the patch from the left image
            patch_l = padded_img_l[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]

            min_cost = float('inf')
            best_disparity = 0

            for d in range(max_disp + 1):
                x_r = x - d
                if half_window <= x_r < width - half_window:
                    # Extract the corresponding patch from the right image
                    patch_r = padded_img_r[y - half_window:y + half_window + 1, x_r - half_window:x_r + half_window + 1]

                    # Calculate the cost using the weighted sum of SSD and NC
                    cost = cost_function(patch_l, patch_r, alpha)

                    # Update the best disparity if the current cost is lower
                    if cost < min_cost:
                        min_cost = cost
                        best_disparity = d

            # Assign the best disparity to the disparity map
            disparities[y, x] = best_disparity

    return disparities[half_window:-half_window, half_window:-half_window]


def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map
    
    Args:
        disparity_gt: ground truth of disparity map as (H, W) numpy array
        disparity_res: estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    #
    # You code here
    #
    # Calculate the absolute difference between ground truth and estimated disparity maps
    abs_diff = np.abs(disparity_gt - disparity_res)

    # Calculate the sum of absolute differences
    sum_abs_diff = np.sum(abs_diff)

    # Calculate the number of pixels
    num_pixels = np.prod(disparity_gt.shape)

    # Calculate the average end-point error (AEPE)
    aepe = sum_abs_diff / num_pixels

    return aepe


def optimal_alpha():
    """Return alpha that leads to the smallest EPE (w.r.t. other values)
    Note:
    Remember to check that max_disp = 15, window_size = 11, and padding_mode='symmetric'
    """
    #
    # Once you find the best alpha, you have to fix it
    #
    alpha = np.random.choice([-0.001, -0.01, -0.1, 0.1, 1, 10])
    alpha = -0.001  # aepe = 1.009 which is the smallest
    return alpha


"""
This is a multiple-choice question
"""


def window_based_disparity_matching():
    """Complete the following sentence by choosing the most appropriate answer 
    and return the value as a tuple.
    (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
    
    Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
        1: Using a bigger window size (e.g., 11x11)
        2: Using a smaller window size (e.g., 3x3)
        
    Q2. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
        1: symmetric
        2: reflect
        3: constant

    Q3. The inaccurate disparity estimation on the left image border happens due to [?].
        1: the inappropriate padding scheme
        2: the limitations of the fixed window size
        3: the absence of corresponding pixels
        
    Example or reponse: (1,1,1)
    """
    return (2, 3, 3)
