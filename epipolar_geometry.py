import numpy as np

def epipoles(F, homogenous = False):
    U, sigma, V = np.linalg.svd(F)
    e1 = V[2, :]
    e1 = e1/e1[2]

    e2 = U[:, 2]
    e2 = e2 / e2[2]

    if not homogenous:
        e1 = e1[:2]
        e2 = e2[:2]

    return e1, e2

def epipolars(F, v1, v2):
    lines1 = F.T @ v2.T                                         # 3 X 3 @ 3 X N = 3 X N
    lines2 = F @ v1.T                                           # 3 X 3 @ 3 X N = 3 X N

    return lines1, lines2
