"""Delete Miss Match and Rename of File
"""
import os
import json

with open(os.path.join("SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

image_dir = SETTINGS_JSON['DATASET_DIR_SEGMENTATION']
mask_dir = SETTINGS_JSON['MASK_DIR']

for img in os.listdir(image_dir):
    if img.split('_')[0]=='CHNCXR':
        mask_img_dir = os.path.splitext(img)[0]+'_mask.png'
        mask = os.path.join(mask_dir,mask_img_dir)
        if not os.path.exists(mask):
            os.remove(os.path.join(image_dir,img))
        os.rename(mask,os.path.join(mask_dir,img))
    else:
        mask_img_dir = img
        mask = os.path.join(mask_dir,mask_img_dir)
        if not os.path.exists(mask):
            os.remove(os.path.join(image_dir,img))