"""Train UNet Model
"""

import os

from model.unet import UNet
from tools.data import train_generator, save_results, is_file, prepare_dataset, show_image

import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import json

with open(os.path.join("SETTINGS.json")) as f:
    SETTINGS_JSON = json.load(f)

with open(os.path.join("PARAMS.json")) as f2:
    PARAMS_JSON = json.load(f2)

#GPU Memory Assign
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
K.tensorflow_backend.set_session(tf.Session(config=config))

# Specify GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

img_height = PARAMS_JSON['UNET_INPUT_SIZE']
img_width = PARAMS_JSON['UNET_INPUT_SIZE']
img_size = (img_height, img_width)
train_path = SETTINGS_JSON['UNET_TRAIN_DIR']
val_path = SETTINGS_JSON['UNET_VAL_DIR']
model_path = os.path.join(SETTINGS_JSON['UNET_SNAPSHOT_DIR'], '{epoch:02d}.h5')
model_weights_name = 'unet_weight_model.h5'
batch_size = 2
val_samples = 50

# generates training set
train_gen = train_generator(
    batch_size=batch_size,
    train_path=train_path,
    image_folder='images',
    mask_folder='masks_dilate',
    target_size=img_size
)

# generates training set
val_gen = train_generator(
    batch_size=batch_size,
    train_path=val_path,
    image_folder='images',
    mask_folder='masks_dilate',
    target_size=img_size
)

# check if pretrained weights are defined
if is_file(file_name=model_weights_name):
    pretrained_weights = model_weights_name
else:
    pretrained_weights = None

# build model
unet = UNet(
    input_size=(img_width, img_height, 1),
    n_filters=64,
    pretrained_weights=pretrained_weights
)
unet.build()

# creating a callback, hence best weights configurations will be saved
model_checkpoint = ModelCheckpoint(
    filepath=model_path, monitor='val_loss', verbose=1)

#lr schedule
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)
#callbacks of CsvLogger
logs_path = os.path.join(SETTINGS_JSON['UNET_LOG_DIR'], "log.csv")
csvlogger = CSVLogger(logs_path)
callbacks = [csvlogger, model_checkpoint, reduceLR]

# model training
# steps per epoch should be equal to number of samples in database divided by batch size
# in this case, it is 528 / 2 = 264
unet.fit_generator(
    generator=train_gen,
    steps_per_epoch=len(os.listdir(
        os.path.join(train_path, 'images')))//batch_size,
    epochs=200,
    validation_data=val_gen,
    validation_steps=val_samples//batch_size,
    callbacks=callbacks
)
