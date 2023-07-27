import numpy as np
import cv2
from utils.epipolar_geometry import epipoles
from utils.visualization_utils import plot_features

def essential_matrix(K, F, args):
    #estimate KT@F@K
    E = K.T @ F @ K

    #taking SVD of E
    UE, sigmaE, VE = np.linalg.svd(E)

    #modify and re-estimate E to make the rank 2
    corrected_sigmaE = np.array([1,1,0])
    newE = UE @ np.diag(corrected_sigmaE) @ VE
    if args.debug:
        print(f"before sigmaE: {sigmaE}")
        print(f"wo rank2 E:{E}")
        print(f"w rank2 E:{newE}")
        print(f"w rank2 E rank:{np.linalg.matrix_rank(newE)}")

    return newE

def test_E(K, F, E, img1, img2, window_name):
    Fe1, Fe2 = epipoles(F, True)
    Ee1, Ee2 = epipoles(E, True)

    Fe1_r = K@Ee1
    Fe2_r = K@Ee2

    #how are they exactly same till 1000th decimal point?
    print(f"from F:{Fe1}")
    print(f"from E:{Fe1_r}")
    print(f"from F:{Fe2}")
    print(f"from E:{Fe2_r}")

    img1_copy = img1.copy()
    plot_features(img1_copy, [Fe1[0:2]], color=(255, 0, 0), thickness=20)
    plot_features(img1_copy, [Fe1_r[0:2]])

    img2_copy = img2.copy()
    plot_features(img2_copy, [Fe2[0:2]], color=(255, 0, 0), thickness=20)
    plot_features(img2_copy, [Fe2_r[0:2]])

    concat = np.hstack((img1_copy, img2_copy))
    cv2.imshow(f"{window_name}", concat)

    