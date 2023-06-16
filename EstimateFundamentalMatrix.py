import numpy as np

def estimate_fundamental_matrix(v1, v2):
    # Construct Ax = 0
    x1, y1 = v1[:,0], v1[:,1]                                   # N
    x2, y2 = v2[:,0], v2[:,1]                                   # N
    ones = np.ones(x1.shape[0])

    A = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, ones]      # N X 9
    A = np.vstack(A).T                                          # N X 9

    #get SVD of  A
    U, sigma, V = np.linalg.svd(A)                              # N x N, N x 9, 9 X 9
    f = V[np.argmin(sigma), :]                                  # 9

    #reconstruct F from singular vector
    F = f.reshape((3,3))

    #take SVD of A
    UF, sigmaF, VF = np.linalg.svd(F)
    sigmaF[2] = 0                                               # enforcing rank 2 constraint
    reestimatedF = UF @ np.diag(sigmaF) @ VF

    return F, reestimatedF

def get_ij_fundamental_matrix(i,j,sfm_map):
    key = (i,j)
    v1, v2, _ = sfm_map.get_feat_matches(key)
    _, F = estimate_fundamental_matrix(v1, v2)

    return F

def get_epipoles(F, homogenous=False):
    U, sigma, V = np.linalg.svd(F)
    e1 = V[2,:]                                                 # 3
    e1 = e1/e1[2]

    e2 = U[:,2]                                                 #3
    e2 = e2/e2[2]

    if not homogenous:
        e1 = e1[0:2]
        e2 = e2[0:2]

    return e1, e2

def get_epipolars(F, v1, v2):
    lines1 = F.T @ v2.T                                         # 3 X 3 @ 3 X N = 3 X N
    lines2 = F @ v1.T                                           # 3 X 3 @ 3 X N = 3 X N

    return lines1, lines2


