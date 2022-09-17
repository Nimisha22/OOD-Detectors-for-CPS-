# Out-Of-Distribution Detectors for Cyber Physical Systems

Modern Cyber-Physical Systems (CPS) such as autonomous vehicles are expected to perform complex control functions under highly dynamic operating environments. Increasingly, to cope with the complexity and scale, these functions are being designed autonomously, often with the assistance of Machine Learning (ML) based techniques. 

In this project, techniques in ML for robust out-of-distribution (OOD) data detection and reasoning are being explored.
Deep learning based techniques for detecting OOD data from multi-modal inputs including high-dimensional images from video cameras are being developed. Implementation and deployment of our solutions is done on Duckietown platform.

# Problem
1. OOD Detectors for CPS have been trained for Rain. 
2. No validation is performed for other atmospheric conditions. 
3. The performance of CPS in fog and sandstorm is not known.

# Proposed Solution
Collect Dataset and Train existing OOD Detection models for Fog and Sandstorm.
Find out how well OOD Detectors work for Fog and Sandstorm.
Modify the variable to fine tune if desired results are not achieved.

# Methodology
## Augmentation of Fog in Dataset
Vignette filter is added to original image to produce depth in image
White tone is added to image to produce fog effect

## Augmentation of Sandstorm in Dataset
Sepia filter is used to create in-distribution images
Gaussian pepper noise is added to train model on more realistic images of sandstorm
Sepia filter is added for brown tone

# Model
https://entuedu-my.sharepoint.com/:f:/g/personal/n2202865j_e_ntu_edu_sg/EjqkEysBstFGsHp8noaYYrABmQ3BtOnwXMOP5y3RIzrAYA


