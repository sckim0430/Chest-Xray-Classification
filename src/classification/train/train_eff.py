################
#    import    #
################

import os,sys
WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WDIR, "gradient-checkpointing"))

import tensorflow as tf
import efficientnet.keras as eff
from efficientnet.keras import preprocess_input
from keras.callbacks import CSVLogger,ReduceLROnPlateau
from MultiCbk import MyCbk

import pandas as pd
import json

from util.model import *
from util.evaluation import *
from util.data import *

from ImageDataGenerator import DataGenerator
from albumentations import (Compose,HorizontalFlip,HueSaturationValue,OneOf,ToFloat,ShiftScaleRotate,
                            GridDistortion,ElasticTransform,JpegCompression, HueSaturationValue,
                            RGBShift,RandomBrightness, RandomContrast, RandomGamma,Blur, GaussianBlur, MotionBlur,
                            MedianBlur,GaussNoise, CenterCrop,IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,
                            RandomSizedCrop)

#load json file for parameter information
with open(os.path.join("SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

with open(os.path.join("PARAMS.json")) as f2: 
    PARAMS_JSON = json.load(f2)

###########
## Train ##
###########
def train(train_df, valid_df,
          input_size,
          epochs, batch_size,
          save_weights_path, save_logs_path,
          train_dir,valid_dir,
          model,multi_model=None
          ):

    """Train Classification Model

    Args:
        train_df (class): train dataset
        valid_df (class): validation dataset
        model_name (str): model name
        input_size (int): input size
        epochs (int): epoch num
        batch_size (int): batch size
        save_weights_path (str): save checkpoint path
        save_logs_path (str): save log path
        train_dir (str): train dataset path
        valid_dir (str): validation dataset path
        model (class): model class
        multi_model (class): multi model class
    """
  
    if not os.path.exists(save_weights_path):
        os.makedirs(save_weights_path)
    if not os.path.exists(save_logs_path): 
        os.makedirs(save_logs_path)

    AUGMENTATIONS_TRAIN = Compose([
        HorizontalFlip(p=0.35),
        RandomSizedCrop(min_max_height =(int(input_size*0.75),input_size),height=input_size,width=input_size,p=0.35),
        OneOf([
            ShiftScaleRotate(rotate_limit=25,p=0.35),
            ElasticTransform(alpha=120,sigma=120*0.05,alpha_affine=120*0.03,p=0.35),
            GridDistortion(p=0.35),
            OpticalDistortion(distort_limit=2,shift_limit=0.5,p=0.35),
        ],p=1),
        OneOf([
            RandomContrast(p=0.35),
            RandomGamma(p=0.35),
            RandomBrightness(p=0.35)
        ],p=1),
        OneOf([
            Blur(p=0.35),
            MedianBlur(p=0.35),
            GaussNoise(p=0.35),
            GaussianBlur(p=0.35)
            ],p=1)
        ],p=1)

    #Generator Parmas
    params_train = {
    'list_IDs':list(train_df.Image),
    'labels':list(train_df.Labels),
    'dim':(input_size,input_size),
    'data_dir':train_dir,
    'batch_size':batch_size,
    'n_channels':3,
    'n_classees':PARAMS_JSON['CLASS_NUM'],
    'aug':AUGMENTATIONS_TRAIN,
    'model_name':None,
    'preprocess_input':preprocess_input,
    'to_categori':to_categori,
    'shuffle':True}

    params_val = {
    'list_IDs':list(valid_df.Image),
    'labels':list(valid_df.Labels),
    'dim':(input_size,input_size),
    'data_dir':valid_dir,
    'batch_size':batch_size,
    'n_channels':3,
    'n_classees':PARAMS_JSON['CLASS_NUM'],
    'aug':None,
    'model_name':None,
    'preprocess_input':preprocess_input,
    'to_categori':to_categori,
    'shuffle':True}

    #Create Generator
    train_generator = DataGenerator(**params_train)
    validation_generator = DataGenerator(**params_val)

    #get class weight
    class_weight_dict = CalculateClassWeight(train_df,PARAMS_JSON['CLASS_NUM'],to_categori)

    #model check point
    model_path = os.path.join(save_weights_path,'{}.h5')
    # check_point = ModelCheckpoint(filepath=model_path, monitor='val_auc',verbose=1,mode=max)

    #lr schedule
    reduceLR = ReduceLROnPlateau(monitor='val_loss',factor=0.5,mode = min,patience=5)

    #callbacks of CsvLogger
    logs_path = os.path.join(save_logs_path,"log.csv")
    csvlogger = CSVLogger(logs_path)

    cbk = MyCbk(model,model_path)

    callbacks = [csvlogger,reduceLR,cbk]

    #Train
    if multi_model is not None:
        multi_model.fit_generator(generator = train_generator,
                            epochs=epochs,
                            validation_data=validation_generator,
                            callbacks=callbacks,
                            class_weight=class_weight_dict)
    else:
        model.fit_generator(generator = train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        class_weight=class_weight_dict)
################
# EfficientNet #
################

input_size_eff = PARAMS_JSON['EFFICIENTNET_INPUT_SIZE']

model_eff,multi_model_eff = get_eff_multi_model(eff,0,(input_size_eff,input_size_eff,3),PARAMS_JSON['CLASS_NUM'],lr=1e-3,dropout=0.2)

eff_train_df = pd.read_csv(SETTINGS_JSON['EFFICIENTNET_TRAIN_CSV_DIR'])
eff_valid_df = pd.read_csv(SETTINGS_JSON['EFFICIENTNET_VAL_CSV_DIR'])

train(eff_train_df,eff_valid_df,input_size_eff,
        300,16,
        SETTINGS_JSON['EFFICIENTNET_SNAPSHOT_DIR'],SETTINGS_JSON['EFFICIENTNET_LOG_DIR'],SETTINGS_JSON['EFFICIENTNET_TRAIN_DIR'],SETTINGS_JSON['EFFICIENTNET_VAL_DIR'],model_eff,multi_model_eff)
