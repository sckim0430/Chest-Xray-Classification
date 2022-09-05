"""Divide Train / Test Dataset
"""
import os
import shutil
import pandas as pd
import numpy as np
import json

with open(os.path.join("SETTINGS.json")) as f:
    SETTINGS_JSON = json.load(f)

dataset_df = pd.read_csv(SETTINGS_JSON['DATASET_CSV_DIR'])
# dataset_df = dataset_df[['Image Index','Finding Labels']]
dataset_df.columns = ['Image', 'Labels']

dataset_dir = os.path.join(SETTINGS_JSON['DATASET_DIR'], 'merged_images')
no_finding_dir = SETTINGS_JSON['NO_FINDING_DIR']
testset_dir = SETTINGS_JSON['TEST_IMAGES_DIR']

#move No Finding Labels
No_Finding_df = dataset_df[dataset_df.Labels == 'No Finding']
dataset_df = dataset_df[dataset_df.Labels != 'No Finding']

for _ in No_Finding_df.Image:
    src_img_path = os.path.join(dataset_dir, _)
    dst_img_path = os.path.join(no_finding_dir, _)
    shutil.move(src_img_path, dst_img_path)

No_Finding_df.to_csv(SETTINGS_JSON['NO_FINDING_CSV_DIR'], index=False)

#Assign Train/Test Set(Img,Csv)
per = 0.075
range_test = int(len(dataset_df)*per)
index = np.arange(len(dataset_df))
index = np.random.permutation(index)

test_index = index[:range_test]

test_df = dataset_df.iloc[test_index]
train_df = dataset_df.drop(dataset_df.index[test_index])

for _ in test_df.Image:
    src_img_path = os.path.join(dataset_dir, _)
    dst_img_path = os.path.join(testset_dir, _)
    shutil.move(src_img_path, dst_img_path)

for _ in os.listdir(dataset_dir):
    src_img_path = os.path.join(dataset_dir, _)
    dst_img_path = os.path.join(SETTINGS_JSON['TRAIN_IMAGES_DIR'], _)
    shutil.move(src_img_path, dst_img_path)

train_df.to_csv(SETTINGS_JSON['TRAIN_CSV_DIR'], index=False)
test_df.to_csv(SETTINGS_JSON['TEST_CSV_DIR'], index=False)

os.rmdir(dataset_dir)
os.remove(SETTINGS_JSON['DATASET_CSV_DIR'])
os.rmdir(SETTINGS_JSON['DATASET_DIR'])
