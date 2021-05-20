################
#    import    #
################

import os,sys
WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WDIR, "gradient-checkpointing"))
import memory_saving_gradients

import tensorflow as tf
from keras.applications.densenet import DenseNet169
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K 
from keras import optimizers, layers, utils
from keras.engine import Model 
from keras.callbacks import CSVLogger,ReduceLROnPlateau
from MultiCbk import MyCbk
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.utils import multi_gpu_model


import pandas as pd
import numpy as np 
import json

from ImageDataGenerator import DataGenerator
from albumentations import (Compose,HorizontalFlip,HueSaturationValue,OneOf,ToFloat,ShiftScaleRotate,
                            GridDistortion,ElasticTransform,JpegCompression, HueSaturationValue,
                            RGBShift,RandomBrightness, RandomContrast, RandomGamma,Blur, GaussianBlur, MotionBlur,
                            MedianBlur,GaussNoise, CenterCrop,IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,
                            RandomSizedCrop)

#load json file for parameter information
with open(os.path.join("/home/ubuntu/DnnProject/ChestAI/SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

with open(os.path.join("/home/ubuntu/DnnProject/ChestAI/PARAMS.json")) as f2: 
    PARAMS_JSON = json.load(f2)

# K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

# #GPU Memory Assign
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# K.tensorflow_backend.set_session(tf.Session(config=config))

################
# KERAS MODELS #
################

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision

def get_model(base_model, 
              layer, 
              input_shape, 
              classes,
              lr=1e-3,
              activation="sigmoid",
              dropout=None, 
              pooling="avg", 
              weights=None,
              pretrained="imagenet"):

    base = base_model(input_shape=input_shape,
                      include_top=False,
                      weights=pretrained)

    if pooling == "avg": 
        x = GlobalAveragePooling2D()(base.output) 
    elif pooling == "max": 
        x = GlobalMaxPooling2D()(base.output) 
    elif pooling is None: 
        x = Flatten()(base.output) 
    if dropout is not None: 
        x = Dropout(dropout)(x) 
    x = Dense(classes, activation=activation)(x) 

    model = Model(inputs=base.input, outputs=x)
    multi_model = multi_gpu_model(model,gpus=2)
    
    if weights is not None:
        model.load_weights(weights)

    for l in multi_model.layers[:layer]:
        l.trainable = False

    multi_model.compile(loss='binary_crossentropy', metrics=["binary_accuracy",auc,precision,recall], 
                  optimizer=optimizers.Adam(lr)) 
    return model,multi_model

#############
#   UTILS   #
#############

#UTILS
def to_categori(array,num_class=None):
    
    if num_class == None:
        print('num_class Error! You shoud give val!')
        return None

    categorical = []
    
    for a in array:
        cat = np.zeros(num_class)
        for idx in a:
            cat[idx] = 1
        categorical.append(cat)

    return categorical    

#PREPROCESSING
def preprocess_input(x, model):
    x = x.astype("float64")
    if model in ("inception","xception","mobilenet"): 
        x /= 255.
        x -= 0.5
        x *= 2.
    if model in ("densenet"): 
        x /= 255.
        if x.shape[-1] == 3:
            x[..., 0] -= 0.485
            x[..., 1] -= 0.456
            x[..., 2] -= 0.406 
            x[..., 0] /= 0.229 
            x[..., 1] /= 0.224
            x[..., 2] /= 0.225 
        elif x.shape[-1] == 1: 
            x[..., 0] -= 0.449
            x[..., 0] /= 0.226
    elif model in ("resnet","vgg"):
        if x.shape[-1] == 3:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.680
        elif x.shape[-1] == 1: 
            x[..., 0] -= 115.799
    return x

def CalculateClassWeight(train_df,class_num,to_categori):
    
    labels = []
    class_frequency = []
    class_weight_dict = {}

    for lst in train_df.Labels:
        labels.append(list(int(i) for i in lst.split('_')))

    label_of_onehot = np.asarray(to_categori(labels,class_num))
    label_of_onehot = label_of_onehot.sum(axis=0)

    for each_class in range(class_num):
        class_frequency.append(label_of_onehot[each_class]/float(len(train_df)))
    for each_class in range(class_num):
        class_weight_dict[each_class] = np.max(class_frequency)/class_frequency[each_class]

    return class_weight_dict

###########
## Train ##
###########
def train(train_df, valid_df,
          model,multi_model, model_name,input_size,
          epochs, batch_size,
          save_weights_path, save_logs_path,
          train_dir,valid_dir,
          ):
  
    if not os.path.exists(save_weights_path):
        os.makedirs(save_weights_path)
    if not os.path.exists(save_logs_path): 
        os.makedirs(save_logs_path)

    AUGMENTATIONS_TRAIN = Compose([
        HorizontalFlip(p=0.25),
        RandomSizedCrop(min_max_height =(int(input_size*0.75),input_size),height=input_size,width=input_size,p=0.25),
        OneOf([
            ShiftScaleRotate(rotate_limit=25),
            ElasticTransform(alpha=120,sigma=120*0.05,alpha_affine=120*0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2,shift_limit=0.5),
        ],p=0.5),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness()
        ],p=0.5),
        OneOf([
            Blur(),
            MedianBlur(),
            GaussNoise(),
            GaussianBlur()
            ],p=0.5)
        ],p=0.5)

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
    'model_name':model_name,
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
    'model_name':model_name,
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
    multi_model.fit_generator(generator = train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        class_weight=class_weight_dict)
                        
# #####################
# # InceptionResNetV3 #
# #####################

# input_size_inception = PARAMS_JSON['INCEPTION_INPUT_SIZE']

# model_inception = get_model(InceptionV3,0,(input_size_inception,input_size_inception,3),PARAMS_JSON['CLASS_NUM'],lr=1e-4,dropout=0.5)
# model_inception_name = 'inception'
# inception_train_df = pd.read_csv(SETTINGS_JSON['INCEPTION_TRAIN_CSV_DIR'])
# inception_valid_df = pd.read_csv(SETTINGS_JSON['INCEPTION_VAL_CSV_DIR'])


# train(inception_train_df,inception_valid_df,model_inception,model_inception_name,input_size_inception,
#         300,16,
#         SETTINGS_JSON['INCEPTION_SNAPSHOT_DIR'],SETTINGS_JSON['INCEPTION_LOG_DIR'],SETTINGS_JSON['INCEPTION_TRAIN_DIR'],SETTINGS_JSON['INCEPTION_VAL_DIR'])

# #####################
# #     XCEPTION      #
# #####################

# input_size_xception = PARAMS_JSON['XCEPTION_INPUT_SIZE']

# model_xception = get_model(Xception,0,(input_size_xception,input_size_xception,3),PARAMS_JSON['CLASS_NUM'],lr=1e-4,dropout=0.5)
# model_xception_name = 'xception'
# xception_train_df = pd.read_csv(SETTINGS_JSON['XCEPTION_TRAIN_CSV_DIR'])
# xception_valid_df = pd.read_csv(SETTINGS_JSON['XCEPTION_VAL_CSV_DIR'])

# train(xception_train_df,xception_valid_df,model_xception,model_xception_name,input_size_xception,
#         300,16,
#         SETTINGS_JSON['XCEPTION_SNAPSHOT_DIR'],SETTINGS_JSON['XCEPTION_LOG_DIR'],SETTINGS_JSON['XCEPTION_TRAIN_DIR'],SETTINGS_JSON['XCEPTION_VAL_DIR'])

###############
# DenseNet169 #
###############

input_size_densenet = PARAMS_JSON['DENSENET_INPUT_SIZE']

model_densenet,multi_model_densenet = get_model(DenseNet169,0,(input_size_densenet,input_size_densenet,3),PARAMS_JSON['CLASS_NUM'],lr=1e-3,dropout=0.2)

model_densenet_name = 'densenet'
densenet_train_df = pd.read_csv(SETTINGS_JSON['DENSENET_TRAIN_CSV_DIR'])
densenet_valid_df = pd.read_csv(SETTINGS_JSON['DENSENET_VAL_CSV_DIR'])

train(densenet_train_df,densenet_valid_df,model_densenet,multi_model_densenet,model_densenet_name,input_size_densenet,
        300,32,
        SETTINGS_JSON['DENSENET_SNAPSHOT_DIR'],SETTINGS_JSON['DENSENET_LOG_DIR'],SETTINGS_JSON['DENSENET_TRAIN_DIR'],SETTINGS_JSON['DENSENET_VAL_DIR'])
