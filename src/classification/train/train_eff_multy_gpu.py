################
#    import    #
################

import os,sys
WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WDIR, "gradient-checkpointing"))
import memory_saving_gradients

import tensorflow as tf
import efficientnet.keras as eff
from efficientnet.keras import preprocess_input
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
with open(os.path.join("SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

with open(os.path.join("PARAMS.json")) as f2: 
    PARAMS_JSON = json.load(f2)

# K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

# #GPU Memory Assign
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# K.tensorflow_backend.set_session(tf.Session(config=config))

# # Specify GPU
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

################
# KERAS MODELS #
################

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max ?????? ??????????????? ?????? ??????
    # round : ???????????????
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # ???????????? 0(Negative) ?????? 1(Positive)??? ????????????
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # ???????????? 0(Negative) ?????? 1(Positive)??? ????????????

    # True Positive??? ?????? ?????? ?????? ?????? ?????? 1(Positive)??? ????????????
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = ?????? ?????? 1(Positive) ??????
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()??? 'divide by zero error' ?????????????????? ?????? ?????? ?????????
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max ?????? ??????????????? ?????? ??????
    # round : ???????????????
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # ???????????? 0(Negative) ?????? 1(Positive)??? ????????????
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # ???????????? 0(Negative) ?????? 1(Positive)??? ????????????

    # True Positive??? ?????? ?????? ?????? ?????? ?????? 1(Positive)??? ????????????
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = ?????? ?????? 1(Positive) ??????
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()??? 'divide by zero error' ?????????????????? ?????? ?????? ?????????
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision

def get_model(base, 
            layer, 
            input_shape, 
            classes,
            lr=1e-3,
            activation="sigmoid",
            dropout=None, 
            pooling="avg", 
            weights=None,
            pretrained="noisy-student"):

    efficientnet = base.EfficientNetB6(weights=pretrained,include_top=False,classes = classes,input_shape=input_shape,pooling=pooling)

    avgpool = efficientnet.get_layer(name='avg_pool').output
    drop = Dropout(dropout)(avgpool)

    DENSE_KERNEL_INITIALIZER = {
        'class_name' : 'VarianceScaling',
        'config':{
            'scale':1./3.,
            'mode':'fan_out',
            'distribution':'uniform'
        }
    }

    output_layer = Dense(classes,activation='sigmoid',kernel_initializer=DENSE_KERNEL_INITIALIZER,name='output')(drop)
    
    model = Model(inputs=efficientnet.input, outputs=output_layer)
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
          model,multi_model,input_size,
          epochs, batch_size,
          save_weights_path, save_logs_path,
          train_dir,valid_dir,
          ):
  
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
    multi_model.fit_generator(generator = train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        class_weight=class_weight_dict)
                        
################
# EfficientNet #
################

input_size_eff = PARAMS_JSON['EFFICIENTNET_INPUT_SIZE']

model_eff,multi_model_eff = get_model(eff,0,(input_size_eff,input_size_eff,3),PARAMS_JSON['CLASS_NUM'],lr=1e-3,dropout=0.2)

eff_train_df = pd.read_csv(SETTINGS_JSON['EFFICIENTNET_TRAIN_CSV_DIR'])
eff_valid_df = pd.read_csv(SETTINGS_JSON['EFFICIENTNET_VAL_CSV_DIR'])

train(eff_train_df,eff_valid_df,model_eff,multi_model_eff,input_size_eff,
        300,16,
        SETTINGS_JSON['EFFICIENTNET_SNAPSHOT_DIR'],SETTINGS_JSON['EFFICIENTNET_LOG_DIR'],SETTINGS_JSON['EFFICIENTNET_TRAIN_DIR'],SETTINGS_JSON['EFFICIENTNET_VAL_DIR'])
