import os,sys

WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WDIR, "../../segmentation/"))

from model.unet import UNet
from tools.image import normalize_mask
from tools.image import reshape_image, normalize_mask
import skimage.io as io

import tensorflow as tf
from keras.applications.densenet import DenseNet169
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers, layers, utils
from keras import backend as K
from keras.engine import Model 
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

import numpy as np  

import cv2
from PIL import Image

import argparse
import json

#json file load
with open(os.path.join("/home/hclee/DeepLearning_SCKIM/ChestClassification/SETTINGS.json")) as f: 
    SETTINGS_JSON = json.load(f)

with open(os.path.join("/home/hclee/DeepLearning_SCKIM/ChestClassification/PARAMS.json")) as f2: 
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
    
    if weights is not None:
        model.load_weights(weights)

    for l in model.layers[:layer]:
        l.trainable = False

    model.compile(loss='binary_crossentropy', metrics=["binary_accuracy",auc,precision,recall], 
                  optimizer=optimizers.Adam(lr)) 
    return model

#############
#   UTILS   #
#############

def preprocess_input(x, model):
    x = x.astype("float32")
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

parser = argparse.ArgumentParser()

parser.add_argument('--label',type = int,default=None)

args = parser.parse_args()

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

#get model file names
model_file_names = []

for _ in os.listdir(SETTINGS_JSON['MODEL_PREDICT_DIR']):
    if os.path.splitext(_)[-1] == '.h5':
        model_file_names.append(os.path.splitext(_)[0])

#get model
models = []

for model_name in model_file_names:
    base_model_name = model_name.split('_')[0]
    input_size = int(model_name.split('_')[1])
    
    if base_model_name == 'inception':
        base_model_name = 'InceptionV3'
    elif base_model_name == 'xception':
        base_model_name = 'Xception'
    elif base_model_name == 'densenet':
        base_model_name = 'DenseNet169'
    else:
        print('model name error!!')
        break

    models.append(get_model(eval(base_model_name),0,(input_size,input_size,3),
                            PARAMS_JSON['CLASS_NUM'],
                            lr=1e-4,
                            weights=os.path.join(SETTINGS_JSON['MODEL_PREDICT_DIR'],model_name+'.h5'),dropout=0.5))

src_path = SETTINGS_JSON['TEST_IMAGES_DIR']
dst_path = SETTINGS_JSON['HEATMAP_DIR']

last_conv_dict = {
    "inception":"conv2d_94",
    "densenet":"conv5_block32_2_conv",
    "xception":"block14_sepconv2"
}

#HeatMap Processing
DILATE_KERNEL = np.ones((15, 15), np.uint8)

for img_path in os.listdir(SETTINGS_JSON['TEST_IMAGES_DIR']):
    orig_img = cv2.imread(os.path.join(SETTINGS_JSON['TEST_IMAGES_DIR'],img_path),cv2.IMREAD_COLOR)
    img = io.imread(os.path.join(SETTINGS_JSON['TEST_IMAGES_DIR'],img_path),as_gray = True)
    img = img/255.
    img = reshape_image(img, unet_size)

    result = unet.predict_on_batch(img)
    result = result[0]
    result = normalize_mask(result)
    result = (result*255).astype('uint8')

    result = cv2.dilate(result,DILATE_KERNEL,iterations=3)
    result = np.expand_dims(result,axis=2)

    result = cv2.bitwise_not(result)

    img = img[0]
    img = (img*255).astype('uint8')
    
    result = cv2.add(img,result)
    result = cv2.bitwise_not(result)
    
    result = cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
    
    heatmaps = np.zeros((len(model_file_names),orig_img.shape[0],orig_img.shape[1]))
    
    for index,(model,model_name) in enumerate(zip(models,model_file_names)):
        model_name = model_name.split('_')[0]

        if args.label != None:
            output = model.output[:,args.label]
        else:
            output = 0.0

            for _ in range(PARAMS_JSON['CLASS_NUM']):
                output = output + model.output[:,_]

        last_conv_layer = model.get_layer(last_conv_dict[model_name])

        grads = K.gradients(output,last_conv_layer.output)[0]
        pooled_grads = K.mean(grads,axis=(0,1,2))

        iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
        
        test_image = Image.fromarray(result)
        
        model_input_size = int(model.get_input_at(0).get_shape()[1])
        resized_image = test_image.resize((model_input_size,model_input_size))
        resized_image = np.array(resized_image,'uint8')
        preprocessed_image = preprocess_input(resized_image,model_name)
        preprocessed_image = np.expand_dims(preprocessed_image,axis=0)

        pooled_grads_value,conv_layer_output_value = iterate([preprocessed_image])

        for i in range(pooled_grads_value.shape[0]):
            conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
    
        heatmap = np.mean(conv_layer_output_value,axis=-1)
        heatmap /= np.max(heatmap)
        heatmap = np.maximum(heatmap,0)
        heatmap = cv2.resize(heatmap,(orig_img.shape[1],orig_img.shape[0]))
        heatmap = np.uint8(255*heatmap)
        heatmaps[index] = heatmap

    heatmap = np.mean(heatmaps,axis=0)
    
    pred = models[0].predict(np.expand_dims(result,axis=0))
    print(pred)
    pred = pred[0][args.label]
    heatmap = (heatmap/255.)*(pred*100*2+55)

    heatmap = heatmap.astype('uint8')
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    
    ###################################################
    # #b값이 높은 요소 제거 혹은 약화
    # height,width = heatmap.shape[:2]
    # img_hsv = cv2.cvtColor(heatmap,cv2.COLOR_BGR2HSV)

    # low_blue = (120-10,30,30)
    # upper_blue = (120+10,255,255)

    # blue_mask = cv2.inRange(img_hsv,low_blue,upper_blue)
    # blue_mask = cv2.bitwise_not(blue_mask)

    # heatmap = cv2.bitwise_and(heatmap,heatmap,mask=blue_mask)

    ###################################################

    # result = cv2.resize(result,(heatmap.shape[1],heatmap.shape[0]))
    # heatmap[np.where(result == 0)]=0
    
    heatmap_img = heatmap * 0.4 + orig_img
    # cv2.imwrite(os.path.join(SETTINGS_JSON['HEATMAP_DIR'],img_path),result)
    cv2.imwrite(os.path.join(SETTINGS_JSON['HEATMAP_DIR'],img_path),heatmap_img)