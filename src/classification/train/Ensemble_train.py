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
from keras.callbacks import CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from keras.layers import Dropout, Flatten, Dense, Input,Conv2D,Reshape
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D,Conv1D
from keras.layers.merge import concatenate

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

K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

#GPU Memory Assign
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

# Specify GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

################
# KERAS MODELS #
################

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def get_model(base_models, 
              layer, 
              input_sizes,
              classes,
              channels=3, 
              lr=1e-3,
              activation="sigmoid",
              dropout=None, 
              pooling="avg", 
              weights=None,
              pretrained="imagenet"):
    
    layers = []
    inputs = []

    for base_model,input_size in zip(base_models,input_sizes):
        base = base_model(input_shape=(input_size,input_size,channels),
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
        
        x = Dense(300,activation='relu')(x)

        layers.append(x)
        inputs.append(base.input)

    merge_layer = concatenate(layers)
    
    x = Dense(classes, activation=activation)(x)

    model = Model(inputs=inputs, outputs=x)

    if weights is not None:
        model.load_weights(weights)

    for l in model.layers[:layer]:
        l.trainable = False

    model.compile(loss='binary_crossentropy', metrics=["binary_accuracy",auc], 
                  optimizer=optimizers.Adam(lr)) 
    return model

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
    
    class_weight = {}

    for model_index in range(len(train_df)):

        labels = []
        class_frequency = []
        

        for lst in train_df[model_index].Labels:
            labels.append(list(int(i) for i in lst.split('_')))

        label_of_onehot = np.asarray(to_categori(labels,class_num))
        label_of_onehot = label_of_onehot.sum(axis=0)

        for each_class in range(class_num):
            class_frequency.append(label_of_onehot[each_class]/float(len(train_df[model_index])))
        
        if model_index == 0:
            for each_class in range(class_num):
                class_weight[each_class] = np.max(class_frequency)/class_frequency[each_class]
        else:
            for each_class in range(class_num):
                class_weight[each_class] = class_weight[each_class] + np.max(class_frequency)/class_frequency[each_class]        

    for each_class in range(class_num):
        class_weight[each_class] = class_weight[each_class]/3

    return class_weight

###########
## Train ##
###########

def train(train_df, valid_df,
          model, model_name,input_size,
          epochs, batch_size,channels,
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

    list_train_IDs = []
    list_train_labels = []

    for df in train_df:
        list_train_IDs.append(list(df.Image))
        list_train_labels.append(list(df.Labels))

    list_val_IDs = []
    list_val_labels = []

    for df in valid_df:
        list_val_IDs.append(list(df.Image))
        list_val_labels.append(list(df.Labels))

    #Generator Parmas
    params_train = {
    'list_IDs':list_train_IDs,
    'labels':list_train_labels,
    'dim':input_sizes,
    'data_dir':train_dir,
    'batch_size':batch_size,
    'n_channels':channels,
    'n_classees':PARAMS_JSON['CLASS_NUM'],
    'aug':AUGMENTATIONS_TRAIN,
    'model_name':model_names,
    'preprocess_input':preprocess_input,
    'to_categori':to_categori,
    'shuffle':True}

    params_val = {
    'list_IDs':list_val_IDs,
    'labels':list_val_labels,
    'dim':input_sizes,
    'data_dir':valid_dir,
    'batch_size':batch_size,
    'n_channels':channels,
    'n_classees':PARAMS_JSON['CLASS_NUM'],
    'aug':None,
    'model_name':model_names,
    'preprocess_input':preprocess_input,
    'to_categori':to_categori,
    'shuffle':True}

    #Create Generator
    train_generator = DataGenerator(**params_train)
    validation_generator = DataGenerator(**params_val)
    
    #get class weight
    class_weight_dict = CalculateClassWeight(train_df,PARAMS_JSON['CLASS_NUM'],to_categori)

    #model check point
    model_path = os.path.join(save_weights_path,'{epoch:02d}-{val_auc:.4f}.h5')
    check_point = ModelCheckpoint(filepath=model_path,monitor='val_auc',verbose=1,model=max,save_best_only=True,save_weights_only=True)

    #lr schedule
    reduceLR = ReduceLROnPlateau(monitor='val_auc',factor=0.5,patience=10)

    #callbacks of CsvLogger
    logs_path = os.path.join(save_logs_path,"log.csv")
    csvlogger = CSVLogger(logs_path)
    callbacks = [csvlogger,check_point,reduceLR]

    #Train
    model.fit_generator(generator = train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        class_weight=class_weight_dict)


###############
#   Ensemble  #
###############

input_size_inception = PARAMS_JSON['INCEPTION_INPUT_SIZE']
input_size_xception = PARAMS_JSON['XCEPTION_INPUT_SIZE']
input_size_densenet = PARAMS_JSON['DENSENET_INPUT_SIZE']

input_sizes =[input_size_inception,input_size_xception,input_size_densenet]

models = [InceptionV3,Xception,DenseNet169]
channels = 3

Ensemble_model = get_model(models,0,input_sizes,PARAMS_JSON['CLASS_NUM'],channels=channels,lr=1e-4,dropout=0.5)

model_names = ['inception','xception','densenet']

inception_train_df = pd.read_csv(SETTINGS_JSON['INCEPTION_TRAIN_CSV_DIR'])
inception_valid_df = pd.read_csv(SETTINGS_JSON['INCEPTION_VAL_CSV_DIR'])
xception_train_df = pd.read_csv(SETTINGS_JSON['XCEPTION_TRAIN_CSV_DIR'])
xception_valid_df = pd.read_csv(SETTINGS_JSON['XCEPTION_VAL_CSV_DIR'])
densenet_train_df = pd.read_csv(SETTINGS_JSON['DENSENET_TRAIN_CSV_DIR'])
densenet_valid_df = pd.read_csv(SETTINGS_JSON['DENSENET_VAL_CSV_DIR'])

train_df = [inception_train_df,xception_train_df,densenet_train_df]
valid_df = [inception_valid_df,xception_valid_df,densenet_valid_df]

train_dir = [SETTINGS_JSON['INCEPTION_TRAIN_DIR'],SETTINGS_JSON['XCEPTION_TRAIN_DIR'],SETTINGS_JSON['DENSENET_TRAIN_DIR']]
valid_dir = [SETTINGS_JSON['INCEPTION_VAL_DIR'],SETTINGS_JSON['XCEPTION_VAL_DIR'],SETTINGS_JSON['DENSENET_VAL_DIR']]

train(train_df,valid_df,Ensemble_model,model_names,input_sizes,
        300,16,channels,
        SETTINGS_JSON['ENSEMBLE_SNAPSHOT_DIR'],SETTINGS_JSON['ENSEMBLE_LOG_DIR'],train_dir,valid_dir)