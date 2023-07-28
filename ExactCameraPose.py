import numpy as np

def get_R_Matrix(M):
    UR, sigmaR, VR = np.linalg.svd(M)
    sigmaR = [1, 1, 1]
    R = UR @ np.diag(sigmaR) @ VR
    return R

def extractCameraPose(E):
    # SVD of E
    UE, sigmaE, VE = np.linalg.svd(E)
    # Estimate C1, C2, C3, C4
    # Possible transactions from left null space of E
    U = UE[:,2]
    Cs = [U, -U, U, -U]

    # Estimate R1, R2, R3, R4
    # Creating S
    S = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # Possible rotations
    R1 = UE @ S @ VE
    R1 = get_R_Matrix(R1)
    R2 = UE @ S.T @ VE
    R2 = get_R_Matrix(R2)
    Rs = [R1, R1, R2, R2]

    Cs_final = []
    Rs_final = []
    eps = 1e-2

    for C, R in zip(Cs, Rs):
        det_value = np.linalg.det(R)

        if -1 - eps < det_value < -1 + eps:
            Cs_final.append(-C)
            Rs_final.append(-R)
        else:
            Cs_final.append(C)
            Rs_final.append(R)

    return Cs_final, Rs_final