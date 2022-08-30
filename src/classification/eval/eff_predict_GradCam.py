"""Get GradCam of EfficientNet Model
"""

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

from PIL import Image

from albumentations import (Compose,HorizontalFlip,HueSaturationValue,OneOf,ToFloat,ShiftScaleRotate,
                            GridDistortion,ElasticTransform,JpegCompression, HueSaturationValue,
                            RGBShift,RandomBrightness, RandomContrast, RandomGamma,Blur, GaussianBlur, MotionBlur,
                            MedianBlur,GaussNoise, CenterCrop,IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,
                            RandomSizedCrop)

import numpy as np
import cv2
import json

from util.data import *
from util.evaluation import *
from util.model import *

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

#Set Unet Parameters
unet_height = PARAMS_JSON['UNET_INPUT_SIZE']
unet_width = PARAMS_JSON['UNET_INPUT_SIZE']
unet_size = (unet_height, unet_width)
model_weights_name = SETTINGS_JSON['UNET_PREDICT_DIR']

#Load Segmentation Model
unet = UNet(
    input_size = (unet_width,unet_height,1),
    n_filters = 64,
    pretrained_weights = model_weights_name
)
unet.build()

#Set EfficientNet Parameters
efficient_file = os.path.join(SETTINGS_JSON['MODEL_PREDICT_DIR'],'efficientnet.h5')
efficient_input_size = PARAMS_JSON['EFFICIENTNET_INPUT_SIZE']

#Load Classification Model
efficient = get_eff_model(eff,0,(efficient_input_size,efficient_input_size,3),PARAMS_JSON['CLASS_NUM'],lr=1e-3,dropout=0.2,weights=efficient_file)

#Dst Path
path_to_img_folder = ''
dst_image = ''

#Kernel of erode and dilate
KERNEL = np.ones((15, 15), np.uint8)

#Apply GradCam
for _ in os.listdir(path_to_img_folder):
    path_to_image = _+'_'+dst_image
    path_to_image = os.path.join(path_to_img_folder,_,path_to_image)

    #Apply Segmentation and Classification
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

    resized_image = test_image.resize((efficient_input_size,efficient_input_size))
    resized_image = np.array(resized_image,'uint8')
    resized_image = preprocess_input(resized_image)
    resized_image = np.expand_dims(resized_image,axis=0)

    prediction = efficient.predict(resized_image)[0]
            
    print(prediction)
    print(prediction.argmax())
    print(prediction.max())

    output = efficient.output[:,prediction.argmax()]

    #Apply GradCam(Use BackPropagation)
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