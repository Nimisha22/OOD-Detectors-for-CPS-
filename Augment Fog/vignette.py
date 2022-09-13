import numpy as np
import cv2
import os

	
#reading the image

def vg(input_image):
    input_image = cv2.resize(input_image, (640, 480))
    rows, cols = input_image.shape[:2]
# resizing the image according to our need
# resize() function takes 2 parameters,
# the image and the dimensions

# Extracting the height and width of an image

   
# generating vignette mask using Gaussian
# resultant_kernels
    X_resultant_kernel = cv2.getGaussianKernel(cols,200)
    Y_resultant_kernel = cv2.getGaussianKernel(rows,200)
   
# generating resultant_kernel matrix
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
   
# creating mask and normalising by using np.linalg
# function
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    output = np.copy(input_image)
   
# applying the mask to each channel in the input image
    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask
        os.chdir('/home/n2202864a/Downloads/vignette')
        cv2.imwrite(f"{img1}", output)

path ='/home/n2202864a/Downloads/Test'
for img1 in os.listdir(path):
     print(img1)
     
     # Dataset is stored in the Test folder in Downloads
     input_image = cv2.imread(f"/home/n2202864a/Downloads/Test/{img1}")
     vg(input_image)
