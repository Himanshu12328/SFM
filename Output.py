import numpy as np
import cv2
from visualization_utils import *
from EstimateFundamentalMatrix import *
from data_utils import *
from RANSAC import *
from matplotlib import pyplot as plt

def sample_matches_epipolars(imgs, test_key, matches):

    img1 = imgs[test_key[0]]
    img2 = imgs[test_key[1]]
    v1, v2, _ = matches

    random.seed(42)
    inliers = random.sample(range(v1.shape[0]-1), 8)
    v1_sample = v1[inliers, :]
    v2_sample = v2[inliers, :]

    show_matches2(img1, img2, [v1_sample, v2_sample], f"test_matches_{test_key}")

    F, reestimatedF = estimate_fundamental_matrix(v1_sample, v2_sample)

    show_epipolars(img1, img2, F, [v1_sample, v2_sample], f"test_without_rank2_{test_key}")
    show_epipolars(img1, img2, reestimatedF, [v1_sample, v2_sample], f"test_with_rank2_{test_key}")

def show_RANSAC(imgs, test_key, matches_before,matches_after):
    img1 = imgs[test_key[0]]
    img2 = imgs[test_key[1]]
    v1, v2, _ = matches_before
    v1_corrected, v2_corrected, _ = matches_after

    show_matches2(img1, img2, [v1, v2], f"Before_RANSAC_{test_key}")
    show_matches2(img1, img2, [v1_corrected, v2_corrected], f"After_RANSAC_{test_key}")

def disambiguated_and_corrected_poses(Xs_all_poses, X_linear, X_non_linear, C):
    plt.figure("Camera disambiguation")
    colors = ['red', 'brown', 'green', 'teal']
    for color, Xc in zip(colors, Xs_all_poses):
        plt.scatter(Xc[:,0], Xc[:,2], color = color, marker='.')

    plt.figure("Linear triangulation")
    plt.scatter(X_linear[:, 0], X_linear[:, 2], color = 'blue', marker='.')
    plt.scatter(0,0, marker='^', s=20)

    plt.figure("Non-Linear triangulation")
    plt.scatter(X_non_linear[:, 0], X_non_linear[:, 2], color = 'red', marker='x')
    plt.scatter(C[0],C[1], marker='^', s=20)