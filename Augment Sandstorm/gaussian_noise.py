import cv2
import numpy as np
from skimage.util import random_noise
import os

# Load the image
path ='/home/n2202865j/Downloads/Test'

for img in os.listdir(path):
 print(img)
 img1 = cv2.imread(f"/home/n2202865j/Downloads/Test/{img}")
 # Add salt-and-pepper noise to the image.
 noise_img = random_noise(img1, mode = 'pepper', amount = 0.05) 
 noise_img = np.array(255 * noise_img, dtype = 'uint8')
 cv2.imshow('blur', noise_img)
 os.chdir('/home/n2202865j/Downloads/output2')
 cv2.imwrite(f"{img}", noise_img)
