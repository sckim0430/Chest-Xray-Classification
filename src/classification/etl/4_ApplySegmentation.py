"""Apply Segmentation Model(UNet) and Make Segmented Image Dataset
"""
import os
import sys

WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WDIR, "../../segmentation/"))

from model.unet import UNet
from tools.data import test_generator, save_results
from tools.image import normalize_mask

import tensorflow as tf
from keras import backend as K 

import cv2
import numpy as np

from tqdm import tqdm
import json

#load json file for parameter information
with open(os.path.join("SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

with open(os.path.join("PARAMS.json")) as f2: 
    PARAMS_JSON = json.load(f2)

#GPU Memory Assign
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
K.tensorflow_backend.set_session(tf.Session(config=config))

# Specify GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

img_height = PARAMS_JSON['UNET_INPUT_SIZE']
img_width = PARAMS_JSON['UNET_INPUT_SIZE']
img_size = (img_height, img_width)
train_path = SETTINGS_JSON['TRAIN_IMAGES_DIR']
model_weights_name = SETTINGS_JSON['UNET_PREDICT_DIR']
save_path = SETTINGS_JSON['SEGMENTED_TRAIN_IMAGES_DIR']

# build model
unet = UNet(
    input_size = (img_width,img_height,1),
    n_filters = 64,
    pretrained_weights = model_weights_name
)
unet.build()

src_gen = test_generator(train_path, img_size)
train_gen = test_generator(train_path, img_size)

KERNAL = np.ones((15, 15), np.uint8)

#Get Segmented Images from Dataset
for src,img_name in tqdm(zip(src_gen,os.listdir(train_path)),leave=False):
    
    result = unet.predict_on_batch(src)
    result = result[0]
    img = normalize_mask(result)
    img = (img*255).astype('uint8')

    img = cv2.erode(img,KERNAL,iterations=2)
    img = cv2.dilate(img,KERNAL,iterations=3)
    
    img = np.expand_dims(img,axis=2)

    img = cv2.bitwise_not(img)

    train_img = next(train_gen)
    train_img = train_img[0]
    train_img = (train_img*255).astype('uint8')
    
    train_img = cv2.add(train_img,img)
    train_img[train_img == img] = 0
    
    cv2.imwrite(os.path.join(save_path,img_name),train_img)
