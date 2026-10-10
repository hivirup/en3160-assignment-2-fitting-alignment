# EN3160 Assignment 2: Fitting and Alignment

## Description
This repository contains the implementations and analysis for EN3160 Assignment 2. The project focuses on core computer vision techniques, specifically feature detection, robust model fitting in the presence of noise, and image alignment using homographies. 

## Topics Covered
* **Blob Detection:** Custom implementation to detect and draw circles in a natural image using Laplacian of Gaussians and scale-space extrema detection, without relying on standard Hough transform functions.
* **Robust Model Fitting (RANSAC):** A from-scratch implementation of the RANSAC algorithm to estimate best-fit lines and circles from noisy point clouds, including inlier consensus evaluation.
* **Homography & Image Warping:** Computing homographies from manually selected planar points to seamlessly superimpose and blend images onto architectural surfaces.
* **SIFT Feature Matching & Alignment:** Using SIFT features and RANSAC-based homography to align target Printed Circuit Board (PCB) images with a reference template, followed by image differencing to highlight 3D structural variations.

## Tools & Libraries
* Python
* OpenCV (cv2)
* NumPy
* SciPy (scipy.optimize)
* Matplotlib 

## Repository Structure
* `/images`: Contains the reference images, including the Wadham College architectural image, PCB templates, and the sunflower field.
* `assignment_code.py` (or `.ipynb`): The main executable scripts containing the algorithms for each of the four primary questions.
