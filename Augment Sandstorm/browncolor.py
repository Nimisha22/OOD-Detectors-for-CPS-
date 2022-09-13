import numpy as np
import cv2
import os
from skimage import io

def sepia(src_image):
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32)/255
    #solid color
    sepia = np.ones(src_image.shape)
    sepia[:,:,0] *= 153 #B
    sepia[:,:,1] *= 204 #G
    sepia[:,:,2] *= 255 #R
    #hadamard
    sepia[:,:,0] *= normalized_gray #B
    sepia[:,:,1] *= normalized_gray #G
    sepia[:,:,2] *= normalized_gray #R
    return np.array(sepia, np.uint8)

path ='/home/n2202865j/Downloads/output2'
for image1 in os.listdir(path):
 print(image1)
 image = cv2.imread(f"/home/n2202865j/Downloads/output2/{image1}")
 image2 = sepia(image)
 image3=cv2.imshow('', image2)

 os.chdir('/home/n2202865j/Downloads/output3')
 cv2.imwrite(f"{image1}", image2)
