"""Merge Dataset NIH and Chexpert
"""
import os
import pandas as pd
import json

with open(os.path.join("SETTINGS.json")) as f:
    SETTINGS_JSON = json.load(f)

nih_df = pd.read_csv(SETTINGS_JSON['DATASET_CSV_DIR'])
nih_df = nih_df[['Image Index', 'Finding Labels']]

#get disease label in (Consolidation,Pneumothorax,Edema,Effusion,Pneumonia,Cardiomegaly)
for index, Labels in enumerate(list(nih_df['Finding Labels'])):
    print('Processing NIH....{}'.format(index))

    label_list = Labels.split('|')
    new_label_list = []

    for label in label_list:
        if label in ['Consolidation', 'Pneumothorax', 'Edema', 'Effusion', 'Pneumonia', 'Cardiomegaly', 'No Finding']:
            new_label_list.append(label)

    if len(new_label_list) == 0:
        nih_df['Finding Labels'][index] = 'No Finding'
    else:
        nih_df['Finding Labels'][index] = '|'.join(new_label_list)

chex_df = pd.read_csv(SETTINGS_JSON['DATASET_CHEXPERT_CSV_DIR'])
chex_df.columns = ['Image Index', 'Consolidation', 'Pneumothorax',
                   'Edema', 'Effusion', 'Pneumonia', 'Cardiomegaly']
chex_df['Finding Labels'] = ''

for index in range(len(chex_df)):
    print('Processing CHEXPERT....{}'.format(index))

    list_label = []

    if chex_df.loc[index][1] == 1:
        list_label.append('Consolidation')

    if chex_df.loc[index][2] == 1:
        list_label.append('Pneumothorax')

    if chex_df.loc[index][3] == 1:
        list_label.append('Edema')

    if chex_df.loc[index][4] == 1:
        list_label.append('Effusion')

    if chex_df.loc[index][5] == 1:
        list_label.append('Pneumonia')

    if chex_df.loc[index][6] == 1:
        list_label.append('Cardiomegaly')

    chex_df['Finding Labels'][index] = '|'.join(list_label)

chex_df = chex_df[['Image Index', 'Finding Labels']]

df = pd.merge(nih_df, chex_df, how='outer')

df.to_csv(SETTINGS_JSON['DATASET_CSV_DIR'], index=False)

os.remove(SETTINGS_JSON['DATASET_CHEXPERT_CSV_DIR'])
