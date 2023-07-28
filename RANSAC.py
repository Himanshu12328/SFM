import random
from EstimateFundamentalMatrix import *
from helpers import homogenize_coords, unhomogenize_coords

random.seed(42)

def RANSAC(v1, v2, max_iter, threshold):
    """
    Input: 
    v1, v2 : non-homogenous image feature coordinates
    max_iter: number of iterations it should run ransac
    threshold: error threshold

    Output:
    inliers
    """
    v1 = homogenize_coords(v1)
    v2 = homogenize_coords(v2)

    max_inliers_idxs = []

    for iter in range(max_iter):
        sample_inds = random.sample(range(v1.shape[0]-1), 8)
        sample_v1 = v1[sample_inds, :]
        sample_v2 = v2[sample_inds, :]

        _, F = estimate_fundamental_matrix(sample_v1, sample_v2)

        error = F @ v1.T
        error = error.T
        error = np.multiply(v2, error)
        error = np.sum(error, axis=1)

        curr_inliers_idxs = abs(error) < threshold
        if np.sum(curr_inliers_idxs) > np.sum(max_inliers_idxs):
            max_inliers_idxs = curr_inliers_idxs
    return max_inliers_idxs
