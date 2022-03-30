import numpy as np 
from enum import Enum




#calculate integralimage
def integral_image(img_arr):
    row_sum = np.zeros((img_arr.shape[0],img_arr.shape[1]))
    integral_image_arr = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1))
    for x in range(img_arr.shape[1]):
        for y in range(img_arr.shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
            integral_image_arr[y+1, x+1] = integral_image_arr[y+1, x-1+1] + row_sum[y, x]
            
    return integral_image_arr





    #compute region
def compute_region(integral_image,topLeft,bottomRight):
    topLeft=(topLeft[1],topLeft[0])
    bottomRight=(bottomRight[1],bottomRight[0])
    if(topLeft==bottomRight):
        return integral_image[topLeft]
    else:
        topRight=(bottomRight[0],topLeft[1])
        bottomLeft=(topLeft[0],bottomRight[1])
        sumRegion=(integral_image[bottomRight]+integral_image[topLeft])-(integral_image[bottomLeft]+integral_image[topRight])
        return sumRegion
  