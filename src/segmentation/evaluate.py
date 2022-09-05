"""Evaluate UNet Model
"""

import os

from model.unet import UNet
from tools.data import train_generator

import tensorflow as tf
from keras import backend as K

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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

img_height = PARAMS_JSON['UNET_INPUT_SIZE']
img_width = PARAMS_JSON['UNET_INPUT_SIZE']
img_size = (img_height, img_width)
test_dir = SETTINGS_JSON['TEST_DIR']
model_weights_name = SETTINGS_JSON['UNET_PREDICT_DIR']

# build model
unet = UNet(
    input_size=(img_width, img_height, 1),
    n_filters=64,
    pretrained_weights=model_weights_name
)
unet.build()

gen = train_generator(batch_size=1, train_path=test_dir, image_folder='images',
                      mask_folder='masks_dilate', target_size=img_size)

eval = unet.evaluate_generator(gen, steps=len(os.listdir(
    SETTINGS_JSON['TEST_IMAGES_DIR_SEGMENTATION'])), verbose=1)

print("loss : {}".format(eval[0]))
print("dice : {}".format(eval[1]))
