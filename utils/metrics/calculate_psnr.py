from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original_img, after_img):
    mse = np.mean((original_img - after_img) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# Read image
original_img = cv2.imread("experience/2_apply_interpolation/img_after_detected/id10_trana.jpg")
after_img = cv2.imread("experience/2_apply_interpolation/img_after_resized/id10_trana_bicubic.jpg")

# Resize image
dim = (original_img.shape[1], original_img.shape[0])
after_img = cv2.resize(after_img,dim)

# Get PSNR metric
value = PSNR(original_img, after_img)
print(f"PSNR value is {value} dB")
