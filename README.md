# 📸 Structure from Motion (SfM) – 3D Reconstruction from Images  

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)  

---

## 🚀 Overview  
This project implements a complete **Structure from Motion (SfM)** pipeline from scratch, transforming 2D image sequences into accurate 3D point clouds. Designed to demonstrate core concepts of computer vision and 3D reconstruction, the project utilizes feature detection, matching, epipolar geometry, camera pose estimation, and triangulation to create dense and sparse reconstructions without relying on heavy external libraries.

---

## 📂 Project Features  

- 📌 **Feature Extraction:** SIFT/ORB keypoint detection and descriptor computation.  
- 🔗 **Feature Matching:** Brute-force and FLANN-based matching strategies.  
- 📐 **Epipolar Geometry:** Fundamental and Essential Matrix estimation using RANSAC.  
- 📸 **Camera Pose Recovery:** Decomposition of Essential Matrix for relative camera pose.  
- 📈 **Triangulation:** 3D point cloud generation using linear triangulation.  
- 🌍 **Visualization:** Matplotlib 3D plots and Open3D support for viewing reconstructed scenes.  
- ⚙️ **Modular Pipeline:** Clean, modular codebase for easy experimentation and extension.

