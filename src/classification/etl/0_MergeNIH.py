"""Merge NIH Data Set to Merged Images Folder 
"""
import os
import shutil
import json

with open(os.path.join("SETTINGS.json")) as f:
    SETTINGS_JSON = json.load(f)

dataset_dir = SETTINGS_JSON['DATASET_DIR']
merged_image_dir = os.path.join(dataset_dir, 'merged_images')

if not os.path.isdir(merged_image_dir):
    os.mkdir(merged_image_dir)

for folder in os.listdir(dataset_dir):
    if folder.split('_')[0] == 'images':
        img_folder_dir = os.path.join(dataset_dir, folder)
        img_folder_dir = os.path.join(img_folder_dir, 'images')

        for img in os.listdir(img_folder_dir):
            shutil.move(os.path.join(img_folder_dir, img), merged_image_dir)

        os.rmdir(img_folder_dir)
        os.rmdir(os.path.join(dataset_dir, folder))
