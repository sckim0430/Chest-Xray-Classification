"""Merge Chexpert Data Set to Merged Images Folder & Csv
"""

import os,sys
import shutil

import numpy as np

import pandas as pd
import json

#load json file for parameter information
with open(os.path.join("SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

with open(os.path.join("PARAMS.json")) as f2: 
    PARAMS_JSON = json.load(f2)

train_df = pd.read_csv(SETTINGS_JSON['DATASET_CHEXPERT_TRAIN_CSV_DIR'])
val_df = pd.read_csv(SETTINGS_JSON['DATASET_CHEXPERT_VAL_CSV_DIR'])

#train/val csv merge
df = pd.merge(train_df,val_df,how='outer')
df = df[['Path','Consolidation','Pneumothorax','Edema','Pleural Effusion','Pneumonia','Cardiomegaly']]
df.columns = ['Path','Consolidation','Pneumothorax','Edema','Effusion','Pneumonia','Cardiomegaly']

#unmentionned data and uncertain data labeling to negative
df.loc[df['Consolidation'] == -1,'Consolidation'] = 0
df.loc[np.isnan(df['Consolidation']),'Consolidation'] = 0

df.loc[df['Pneumothorax'] == -1,'Pneumothorax'] = 0
df.loc[np.isnan(df['Pneumothorax']),'Pneumothorax'] = 0

df.loc[df['Edema'] == -1,'Edema'] = 0
df.loc[np.isnan(df['Edema']),'Edema'] = 0

df.loc[df['Effusion'] == -1,'Effusion'] = 0
df.loc[np.isnan(df['Effusion']),'Effusion'] = 0

df.loc[df['Pneumonia'] == -1,'Pneumonia'] = 0
df.loc[np.isnan(df['Pneumonia']),'Pneumonia'] = 0

df.loc[df['Cardiomegaly'] == -1,'Cardiomegaly'] = 0
df.loc[np.isnan(df['Cardiomegaly']),'Cardiomegaly'] = 0


#delete not in (Consolidation,Pneumothorax,Edema,Effusion,Pneumonia,Cardiomegaly)
bool_df = df.astype('bool')
bool_df = bool_df.drop('Path',axis=1)

count = 0 
drop_list = []

for index in range(len(df)):
    print('Processing....{}'.format(index))
    
    if not bool_df.loc[index].any():    #delete index for no disease info in csv file
        drop_list.append(index)
    else:          #move image of having disease info 
        count = count + 1
        data_dir = df.loc[index][0]
        data_dir = data_dir.replace('CheXpert-v1.0-small/','')    
        data_dir = os.path.join(SETTINGS_JSON['DATASET_DIR'],data_dir)
        dst_dir = os.path.join(SETTINGS_JSON['DATASET_DIR'],'merged_images','Chexpert_{}.jpg'.format(count))
        df['Path'][index] = 'Chexpert_{}.jpg'.format(count)
        shutil.move(data_dir,dst_dir)

#delete for no disease info in csv file
df.drop(drop_list,inplace=True)

df.to_csv(SETTINGS_JSON['DATASET_CHEXPERT_CSV_DIR'],index=False)

#delete for no disease info in image folder & file
shutil.rmtree(os.path.join(SETTINGS_JSON['DATASET_DIR'],'valid'))
shutil.rmtree(os.path.join(SETTINGS_JSON['DATASET_DIR'],'train'))
os.remove(SETTINGS_JSON['DATASET_CHEXPERT_TRAIN_CSV_DIR'])
os.remove(SETTINGS_JSON['DATASET_CHEXPERT_VAL_CSV_DIR'])
