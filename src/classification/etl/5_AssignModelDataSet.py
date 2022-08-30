"""Generate Train Dataset
"""
import os 
import shutil
import pandas as pd 
import numpy as np
from PIL import Image
import json

with open("SETTINGS.json") as f: 
    SETTINGS_JSON = json.load(f)

with open("PARAMS.json") as f: 
    PARAMS_JSON = json.load(f)

def ConvertToResizedNumpy(data_dir,size,save_path):
    image = Image.open(data_dir).convert('RGB')
    resized_image = image.resize(size)
    resized_array = np.array(resized_image,'uint8')
    np.save(save_path,resized_array)


def convert_to_label(Target):
    """Convert str Label to int Label

    Args:
        Target (str): label of string

    Returns:
        list: label of int list
    """
    tmp_target = []

    for t in Target:
        tmp_target.append(t.split('|'))
    
    label = []

    for t in tmp_target:
        sub_label = []
        for target in t:
            if target == 'Consolidation':
                sub_label.append(0)
            elif target == 'Pneumothorax':
                sub_label.append(1)
            elif target == 'Edema':
                sub_label.append(2)
            elif target == 'Effusion':
                sub_label.append(3)
            elif target == 'Pneumonia':
                sub_label.append(4)
            elif target == 'Cardiomegaly':
                sub_label.append(5)
            else:
                print('Target Error! you should look csv file!')
                return None
        
        sub_label.sort()

        label_sub_str = ''

        for index,val in enumerate(sub_label):
            if index == len(sub_label)-1:
                label_sub_str += '{}'.format(val)
            else:
                label_sub_str += '{}_'.format(val)

        label.append(label_sub_str)        

    return label

train_dir = SETTINGS_JSON['TRAIN_CSV_DIR']
data_dir = os.path.join(SETTINGS_JSON['SEGMENTED_TRAIN_IMAGES_DIR'])
train_df = pd.read_csv(train_dir)

for index,img in enumerate(train_df.Image):
    print('Processing....{}'.format(index))
    # ConvertToResizedNumpy(os.path.join(data_dir,img),(PARAMS_JSON['INCEPTION_INPUT_SIZE'],PARAMS_JSON['INCEPTION_INPUT_SIZE']),os.path.join(SETTINGS_JSON['INCEPTION_TRAIN_DIR'],os.path.splitext(img)[0]+'.npy'))
    # ConvertToResizedNumpy(os.path.join(data_dir,img),(PARAMS_JSON['XCEPTION_INPUT_SIZE'],PARAMS_JSON['XCEPTION_INPUT_SIZE']),os.path.join(SETTINGS_JSON['XCEPTION_TRAIN_DIR'],os.path.splitext(img)[0]+'.npy'))
    ConvertToResizedNumpy(os.path.join(data_dir,img),(PARAMS_JSON['EFFICIENTNET_INPUT_SIZE'],PARAMS_JSON['EFFICIENTNET_INPUT_SIZE']),os.path.join(SETTINGS_JSON['EFFICIENTNET_TRAIN_DIR'],os.path.splitext(img)[0]+'.npy'))

train_df.Labels = convert_to_label(train_df.Labels)

for index,img in enumerate(train_df.Image):
    train_df.Image[index] = os.path.splitext(img)[0]

# train_df.to_csv(SETTINGS_JSON['INCEPTION_TRAIN_CSV_DIR'],index=False)
# train_df.to_csv(SETTINGS_JSON['XCEPTION_TRAIN_CSV_DIR'],index=False)
train_df.to_csv(SETTINGS_JSON['EFFICIENTNET_TRAIN_CSV_DIR'],index=False)