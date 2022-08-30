"""Get GradCam of Ensemble Models
"""

import os,sys

WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WDIR, "../../segmentation/"))

from model.unet import UNet
from tools.image import normalize_mask
from tools.image import reshape_image, normalize_mask

import tensorflow as tf
from keras import backend as K

import cv2
from PIL import Image
import skimage.io as io
import numpy as np  

import argparse
import json

from util.data import *
from util.evaluation import *
from util.model import *

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

parser = argparse.ArgumentParser()

parser.add_argument('--label',type = int,default=None)

args = parser.parse_args()

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

#Set Ensemble Parameters
model_file_names = []

for _ in os.listdir(SETTINGS_JSON['MODEL_PREDICT_DIR']):
    if os.path.splitext(_)[-1] == '.h5':
        model_file_names.append(os.path.splitext(_)[0])

#Load Classification Models
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

#Kernel of dilate
DILATE_KERNEL = np.ones((15, 15), np.uint8)

#Apply GradCam
for img_path in os.listdir(SETTINGS_JSON['TEST_IMAGES_DIR']):
    orig_img = cv2.imread(os.path.join(SETTINGS_JSON['TEST_IMAGES_DIR'],img_path),cv2.IMREAD_COLOR)
    img = io.imread(os.path.join(SETTINGS_JSON['TEST_IMAGES_DIR'],img_path),as_gray = True)
    img = img/255.
    img = reshape_image(img, unet_size)

    #Apply Segmentation and Classification
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

    #Apply GradCam to Each Models(Use BackPropagation)
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