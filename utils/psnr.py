import numpy as np
import math

def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2) #MSE 구하는 코드
    if mse == 0:
        return 0, 100
    PIXEL_MAX = 1

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) #PSNR구하는 코드