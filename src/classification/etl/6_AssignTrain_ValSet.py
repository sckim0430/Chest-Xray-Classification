"""Divide Train / Validation Dataset from Train Dataset
"""
import os
import shutil
import numpy as np
import pandas as pd
import json

with open("SETTINGS.json") as f: 
    SETTINGS_JSON = json.load(f)

def Assign_train_val(train_df_path,train_dir,val_df_path,val_dir,per = None):

    if per == None or per > 0.5:
        print('Value of per Error!')
        return None

    train_df = pd.read_csv(train_df_path)

    range_val = int(len(train_df)*per)

    k = np.arange(len(train_df))
    k = np.random.permutation(k)

    val_index = k[:range_val]

    val_df = train_df.iloc[val_index]
    train_df = train_df.drop(train_df.index[val_index])
    
    for val_img in val_df.Image:    
        src_path = os.path.join(train_dir,val_img)+'.npy'
        dst_path = os.path.join(val_dir,val_img)+'.npy'
        shutil.move(src_path,dst_path)
    
    train_df.to_csv(train_df_path,index=False)
    val_df.to_csv(val_df_path,index=False)

# inceptionv3_train_df_path = SETTINGS_JSON['INCEPTION_TRAIN_CSV_DIR']
# inceptionv3_train_dir = SETTINGS_JSON['INCEPTION_TRAIN_DIR']
# inceptionv3_val_df_path = SETTINGS_JSON['INCEPTION_VAL_CSV_DIR']
# inceptionv3_val_dir = SETTINGS_JSON['INCEPTION_VAL_DIR']

# Assign_train_val(inceptionv3_train_df_path,inceptionv3_train_dir,inceptionv3_val_df_path,inceptionv3_val_dir,0.1)

# xception_train_df_path = SETTINGS_JSON['XCEPTION_TRAIN_CSV_DIR']
# xception_train_dir = SETTINGS_JSON['XCEPTION_TRAIN_DIR']
# xception_val_df_path = SETTINGS_JSON['XCEPTION_VAL_CSV_DIR']
# xception_val_dir = SETTINGS_JSON['XCEPTION_VAL_DIR']

# Assign_train_val(xception_train_df_path,xception_train_dir,xception_val_df_path,xception_val_dir,0.1)

eff_train_df_path = SETTINGS_JSON['EFFICIENTNET_TRAIN_CSV_DIR']
eff_train_dir = SETTINGS_JSON['EFFICIENTNET_TRAIN_DIR']
eff_val_df_path = SETTINGS_JSON['EFFICIENTNET_VAL_CSV_DIR']
eff_val_dir = SETTINGS_JSON['EFFICIENTNET_VAL_DIR']

Assign_train_val(eff_train_df_path,eff_train_dir,eff_val_df_path,eff_val_dir,0.1)
