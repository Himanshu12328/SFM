import numpy as np

def disambiguateCameraPose(Cs, Rs, Xs, orig_idxs):
    """
    Finds correct camera pose ontained from ExtractCameraPose.py
    inputs:
    Cs: All possible transaltions
    Rs: All possible rotaions
    Xs: All possible Xs corresponding to the poses
    orig_idxs: array of indicies of visibility matrix
    outputs:
    pose
    visibility_indxs: modified array of indicies of visibility matrix
    """
    correctC = None
    correctR = None
    correctX = None
    max_inliers = []
    for C, R, X in zip(Cs, Rs, Xs):
        r3 = R[:,2]
        C = C.reshape((3,1))
        cond1 = X[:,2].T
        cond2 = r3.T @ (X.T - C)

        inliers = np.logical_and(cond1 > 0, cond2 > 0)

        if np.sum(inliers) > np.sum(max_inliers):
            correctC = C
            correctR = R
            correctX = X[inliers]
            visibility_idxs = orig_idxs[inliers]
            max_inliers = inliers
    return correctC.flatten(), correctR, correctX, visibility_idxs