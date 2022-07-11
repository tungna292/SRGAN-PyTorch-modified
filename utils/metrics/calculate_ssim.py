from skimage.metrics import structural_similarity
import cv2
import numpy as np

def SSIM(original_img, after_img):
    # Resize img
    after_img = cv2.resize(after_img, (original_img.shape[1], original_img.shape[0]))
    
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(original_gray, after_gray, full=True)
    return score
    
original_img = cv2.imread("experience/2_apply_interpolation/img_after_detected/id10_trana.jpg")
after_img = cv2.imread("experience/1_apply_srgan/have_resolution/img_after_detected_no_border_have_solu/id10_trana.jpg")
score = SSIM(original_img, after_img)
print("Image similarity", score)