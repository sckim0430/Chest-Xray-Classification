################
#    import    #
################

import os,sys

WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WDIR, "../../segmentation/"))

from model.unet import UNet
from tools.image import reshape_image, normalize_mask
import skimage.io as io

import tensorflow as tf
import efficientnet.keras as eff
from efficientnet.keras import preprocess_input
from keras import backend as K 
from keras import optimizers, layers, utils
from keras.engine import Model 
from keras.callbacks import CSVLogger
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from PIL import Image

from albumentations import (Compose,HorizontalFlip,HueSaturationValue,OneOf,ToFloat,ShiftScaleRotate,
                            GridDistortion,ElasticTransform,JpegCompression, HueSaturationValue,
                            RGBShift,RandomBrightness, RandomContrast, RandomGamma,Blur, GaussianBlur, MotionBlur,
                            MedianBlur,GaussNoise, CenterCrop,IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,
                            RandomSizedCrop)

import pandas as pd
import numpy as np
import cv2

import glob
import json

#json file load
with open(os.path.join("SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

with open(os.path.join("PARAMS.json")) as f2: 
    PARAMS_JSON = json.load(f2)

#Specify GPU & Usage Percent
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#GPU Memory Assign
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
K.tensorflow_backend.set_session(tf.Session(config=config))

################
#  KERAS EVAL  #
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

################
# KERAS  MODEL #
################

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
    
    if weights is not None:
        model.load_weights(weights)

    for l in model.layers[:layer]:
        l.trainable = False

    model.compile(loss='binary_crossentropy', metrics=["binary_accuracy",auc,precision,recall], 
                optimizer=optimizers.Adam(lr)) 
    return model

##########
# SCRIPT #
##########

#get segmentation model
unet_height = PARAMS_JSON['UNET_INPUT_SIZE']
unet_width = PARAMS_JSON['UNET_INPUT_SIZE']
unet_size = (unet_height, unet_width)
model_weights_name = SETTINGS_JSON['UNET_PREDICT_DIR']

unet = UNet(
    input_size = (unet_width,unet_height,1),
    n_filters = 64,
    pretrained_weights = model_weights_name
)
unet.build()

#get classification model file names
efficient_file = os.path.join(SETTINGS_JSON['MODEL_PREDICT_DIR'],'efficientnet.h5')
efficient_input_size = PARAMS_JSON['EFFICIENTNET_INPUT_SIZE']

#get classification model
efficient = get_model(eff,0,(efficient_input_size,efficient_input_size,3),PARAMS_JSON['CLASS_NUM'],lr=1e-3,dropout=0.2,weights=efficient_file)

#predict target
path_to_img_folder = ''
dst_image = ''

KERNEL = np.ones((15, 15), np.uint8)

for _ in os.listdir(path_to_img_folder):
    path_to_image = _+'_'+dst_image
    path_to_image = os.path.join(path_to_img_folder,_,path_to_image)

    #total prediction
    img = io.imread(path_to_image,as_gray = True)
    img = img/255.
    img = reshape_image(img, unet_size)

    result = unet.predict_on_batch(img)
    result = result[0]
    new_img = normalize_mask(result)
    new_img = (new_img*255).astype('uint8')

    new_img = cv2.erode(new_img,KERNEL,iterations=2)
    new_img = cv2.dilate(new_img,KERNEL,iterations=3)
    new_img = np.expand_dims(new_img,axis=2)

    new_img = cv2.bitwise_not(new_img)

    img = img[0]
    img = (img*255).astype('uint8')

    img = cv2.add(img,new_img)
    img[img == new_img] = 0

    test_image = Image.fromarray(img).convert('RGB')

    #each model prediction mean
    resized_image = test_image.resize((efficient_input_size,efficient_input_size))
    resized_image = np.array(resized_image,'uint8')
    ########################################################################
    resized_image = preprocess_input(resized_image)
    resized_image = np.expand_dims(resized_image,axis=0)
    prediction = efficient.predict(resized_image)[0]
    ########################################################################
            
    print(prediction)
    print(prediction.argmax())
    print(prediction.max())

    output = efficient.output[:,prediction.argmax()]

    last_conv_layer = efficient.get_layer('top_conv')

    grads = K.gradients(output,last_conv_layer.output)[0]
    pooled_grads = K.mean(grads,axis=(0,1,2))

    iterate = K.function([efficient.input],[pooled_grads,last_conv_layer.output[0]])
            
    pooled_grads_value,conv_layer_output_value = iterate([resized_image])

    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value,axis=-1)
    heatmap /= np.max(heatmap)
    heatmap = np.maximum(heatmap,0)

    orig_img = cv2.imread(path_to_image,cv2.IMREAD_COLOR)

    heatmap = cv2.resize(heatmap,(orig_img.shape[1],orig_img.shape[0]))

    heatmap = np.uint8(255*heatmap)

    heatmap = (heatmap/255.)*(prediction.max() * (255-150) +150)
    heatmap = heatmap.astype('uint8')

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    heatmap_img = heatmap * 0.4 + orig_img
    path_to_heatmap = os.path.splitext(path_to_image)[0]
    path_to_heatmap = path_to_heatmap+'_heatmap.png'
    cv2.imwrite(path_to_heatmap,heatmap_img)