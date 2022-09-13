import cv2, math
import numpy as np
import os

def fog(img):
    
    img_f = img / 255
    (row, col, chs) = img.shape
    
    # brightness
    A = 0.5 

    # Fog concentration
    beta = 0.08 
    
    # Atomization size
    size = math.sqrt(max(row, col)) 

    # Atomization Center
    center = (row // 2, col // 2) 
    
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j-center[0])**2 + (l-center[1])**2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    os.chdir('/home/n2202864a/Downloads/fog')

    cv2.imshow("src", img)
    cv2.imshow("dst", img_f)
    img_f = (img_f * 255).astype(np.uint8)

    cv2.imwrite(f"{img1}", img_f)
    cv2.waitKey(300)

path ='/home/n2202864a/Downloads/vignette'
for img1 in os.listdir(path):
    print(img1)

    # Original image after adding vignette filter is passes in 
    # fog function to add white tone
    
    input_image = cv2.imread(f"/home/n2202864a/Downloads/vignette/{img1}")
    fog(input_image)
