import argparse

import cv2
import numpy as np
import torch
import os
import utils.config as config
import src.imgproc as imgproc
from models.model import Generator
import time

def SRGAN(image, output_path, name_image, weights_path):
    """Module can improve quality of images.
    The results of elapsed time are saved in "results" folder
    Args:
        image: low-resolution input image
        name_image: the name of the image we want to save
        output_path: output path to save the image
        weights_path: where to save the pretrain model
    """
    
    # Initialize the model
    model = Generator()
    model = model.to(memory_format=torch.channels_last, device=config.device)

    # Load the SRGAN model weights
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    # Start the verification mode of the model.
    model.eval()

    start_time = time.time() 
    # Convert BGR channel image format data to RGB channel image format data
    lr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB channel image format data to Tensor channel image format data
    lr_tensor = imgproc.image_to_tensor(lr_image, False, False).unsqueeze_(0)

    # Transfer Tensor channel image format data to CUDA device
    lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    output_name = f"{output_path}/{name_image}.jpg"
    
    # Check output path
    isExist = os.path.exists(output_path)
    if not isExist:
        os.makedirs(output_path)
        cv2.imwrite(output_name, sr_image)
    else:
        cv2.imwrite(output_name, sr_image)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Save results
    result = f"SR image name {name_image} - {elapsed_time}s in CPU"
    with open('results/output.txt', "a+") as f:
        f.write(result + "\n")

# Test with multiple images in the folder
input_folder_path = "experience/1_apply_srgan/no_resolution/img_after_detected_with_border"
output_path="experience/1_apply_srgan/have_resolution/img_after_detected_have_border_have_solu"
weights_path="models/pretrained_models/SRResNet_x4-ImageNet-2096ee7f.pth.tar"

for index, filename in enumerate(os.listdir(input_folder_path)):
    name_image = os.path.splitext(filename)[0]
    if name_image != "output":
        image = cv2.imread(input_folder_path + "/" + filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        SRGAN(image, output_path, name_image, weights_path)

# Test with a single image
# output_path="test"
# weights_path="models/pretrained_models/SRResNet_x4-ImageNet-2096ee7f.pth.tar"
# image = cv2.imread("experience/1_apply_srgan/have_resolution/img_after_detected/no_border/1_thidang.jpg").astype(np.float32) / 255.0
# SRGAN(image=image, output_path=output_path, name_image="test", weights_path=weights_path)