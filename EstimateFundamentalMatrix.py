import numpy as np

def estimate_fundamental_matrix(v1, v2):
    # Construct Ax = 0
    x1, y1 = v1[:,0], v1[:,1]
    x2, y2 = v2[:,0], v2[:,1]
    ones = np.ones(x1.shape[0])

    A = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, ones]
    A = np.vstack(A).T

    #get SVD of  A
    U, sigma, V = np.linalg.svd(A)
    f = V[np.argmin(sigma), :]

    F = f.reshape((3,3))

    UF, sigmaF, VF = np.linalg.svd(F)
    sigmaF[2] = 0 #enforcing rank 2 constraint
    reestimatedF = UF @ np.diag(sigmaF) @ VF

    return F, reestimatedF

def get_ij_fundamental_matrix(i,j,sfm_map):
    key = (i,j)
    v1, v2, _ = sfm_map.get_feat_matches(key)
    _, F = estimate_fundamental_matrix(v1, v2)

    return F

