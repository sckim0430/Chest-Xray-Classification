"""Assign Dataset to Train and Test Mask / Mask Dilate Folder
"""

import os

import cv2

from glob import glob
from tqdm import tqdm

import json

with open(os.path.join("SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

with open(os.path.join("PARAMS.json")) as f2: 
    PARAMS_JSON = json.load(f2)


mask_images = glob(os.path.join(SETTINGS_JSON['TRAIN_MASK_DIR_SEGMENTATION'],'*.png'))
mask_val = mask_images[:50]
mask_train = mask_images[50:]

for mask_image in tqdm(mask_images):
    base_file = os.path.basename(mask_image)
    image_file = os.path.join(SETTINGS_JSON['TRAIN_IMAGES_DIR_SEGMENTATION'],base_file)
    mask_dilate_file = os.path.join(SETTINGS_JSON['TRAIN_MASK_DILATE_DIR_SEGMENTATION'],base_file)

    image = cv2.imread(image_file)
    mask = cv2.imread(mask_image,cv2.IMREAD_GRAYSCALE)
    mask_dilate = cv2.imread(mask_dilate_file,cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image,(PARAMS_JSON['UNET_INPUT_SIZE'],PARAMS_JSON['UNET_INPUT_SIZE']))
    mask = cv2.resize(mask,(PARAMS_JSON['UNET_INPUT_SIZE'],PARAMS_JSON['UNET_INPUT_SIZE']))
    mask_dilate = cv2.resize(mask_dilate,(PARAMS_JSON['UNET_INPUT_SIZE'],PARAMS_JSON['UNET_INPUT_SIZE']))

    if (mask_image in mask_train):
        cv2.imwrite(os.path.join(SETTINGS_JSON['UNET_TRAIN_DIR'],'images',base_file),image)
        cv2.imwrite(os.path.join(SETTINGS_JSON['UNET_TRAIN_DIR'],'masks',base_file),mask)
        cv2.imwrite(os.path.join(SETTINGS_JSON['UNET_TRAIN_DIR'],'masks_dilate',base_file),mask_dilate)
    else:
        cv2.imwrite(os.path.join(SETTINGS_JSON['UNET_VAL_DIR'],'images',base_file),image)
        cv2.imwrite(os.path.join(SETTINGS_JSON['UNET_VAL_DIR'],'masks',base_file),mask)
        cv2.imwrite(os.path.join(SETTINGS_JSON['UNET_VAL_DIR'],'masks_dilate',base_file),mask_dilate)