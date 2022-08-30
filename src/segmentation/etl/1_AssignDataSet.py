"""Assign Dataset to Train and Test Mask / Mask Dilate Folder
"""

import os

import numpy as np
import cv2

from glob import glob
from tqdm import tqdm
import json

with open(os.path.join("SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

DILATE_KERNEL = np.ones((15, 15), np.uint8)

mask_images = glob(os.path.join(SETTINGS_JSON['MASK_DIR'],'*.png'))
mask_test = mask_images[:50]
mask_train = mask_images[50:]

for mask_image in tqdm(mask_images):
    base_file = os.path.basename(mask_image)
    image_file = os.path.join(SETTINGS_JSON['DATASET_DIR_SEGMENTATION'],base_file)
    
    image = cv2.imread(image_file)
    mask = cv2.imread(mask_image,cv2.IMREAD_GRAYSCALE)
    mask_dilate = cv2.dilate(mask,DILATE_KERNEL,iterations=2)

    if (mask_image in mask_train):
        cv2.imwrite(os.path.join(SETTINGS_JSON['TRAIN_IMAGES_DIR_SEGMENTATION'],base_file),image)
        cv2.imwrite(os.path.join(SETTINGS_JSON['TRAIN_MASK_DIR_SEGMENTATION'],base_file),mask)
        cv2.imwrite(os.path.join(SETTINGS_JSON['TRAIN_MASK_DILATE_DIR_SEGMENTATION'],base_file),mask_dilate)
    else:
        cv2.imwrite(os.path.join(SETTINGS_JSON['TEST_IMAGES_DIR_SEGMENTATION'],base_file),image)
        cv2.imwrite(os.path.join(SETTINGS_JSON['TEST_MASK_DIR_SEGMENTATION'],base_file),mask)
        cv2.imwrite(os.path.join(SETTINGS_JSON['TEST_MASK_DILATE_DIR_SEGMENTATION'],base_file),mask_dilate)