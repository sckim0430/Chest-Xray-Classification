"""Evaluate the EfficientNet Model
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

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from PIL import Image

from albumentations import (Compose,HorizontalFlip,HueSaturationValue,OneOf,ToFloat,ShiftScaleRotate,
                            GridDistortion,ElasticTransform,JpegCompression, HueSaturationValue,
                            RGBShift,RandomBrightness, RandomContrast, RandomGamma,Blur, GaussianBlur, MotionBlur,
                            MedianBlur,GaussNoise, CenterCrop,IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,
                            RandomSizedCrop)

import json
import pandas as pd
import numpy as np
import cv2

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

#Set Segmentation Parameter
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

#Set Classification Parameter
efficient_file = os.path.join(SETTINGS_JSON['MODEL_PREDICT_DIR'],'efficientnet.h5')
efficient_input_size = PARAMS_JSON['EFFICIENTNET_INPUT_SIZE']

#Load Classification Model
efficient = get_eff_model(eff,0,(efficient_input_size,efficient_input_size,3),PARAMS_JSON['CLASS_NUM'],lr=1e-3,dropout=0.2,weights=efficient_file)

#Dst Target
path_to_test_images = SETTINGS_JSON['TEST_IMAGES_DIR']
true_df = pd.read_csv(SETTINGS_JSON['TEST_CSV_DIR'],keep_default_na=False)

test_num = 3000
true_df = true_df[:test_num]

prediction = []
test_image_names = []

for _ in true_df.Image:
    test_image_names.append(_)

KERNEL = np.ones((15, 15), np.uint8)

#Total Prediction
for index,test_image_name in enumerate(test_image_names):
    sys.stdout.write("Predicting: {}/{} ...\r".format(index+1, len(test_image_names)))
    sys.stdout.flush()

    img = io.imread(os.path.join(path_to_test_images,test_image_name),as_gray = True)
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
    # test_image = Image.open(os.path.join(path_to_test_images,test_image_name)).convert('RGB')
    
    #each model prediction mean
    resized_image = test_image.resize((efficient_input_size,efficient_input_size))
    resized_image = np.array(resized_image,'uint8')
    ########################################################################
    resized_image = preprocess_input(resized_image)
    resized_image = np.expand_dims(resized_image,axis=0)
    prediction.append(efficient.predict(resized_image)[0])
    ########################################################################
        
prediction = np.array(prediction)

prediction = np.array(prediction>0.5,dtype=np.uint8)

labels = []

for lst in list(convert_to_label(true_df.Labels)):
    labels.append(list(int(i) for i in lst.split('_')))

true = np.asarray(to_categori(labels,PARAMS_JSON['CLASS_NUM']))

#Each Label Evaluation
print('=======================================================================')
print('Consolidation Accuracy: {}'.format(Accuracy(true[:,0].reshape(len(true),1),prediction[:,0].reshape(len(prediction),1))))
print('Consolidation Precision: {}'.format(Precision(true[:,0].reshape(len(true),1),prediction[:,0].reshape(len(prediction),1))))
print('Consolidation Recall: {}'.format(Recall(true[:,0].reshape(len(true),1),prediction[:,0].reshape(len(prediction),1))))

print('Pneumothorax Accuracy: {}'.format(Accuracy(true[:,1].reshape(len(true),1),prediction[:,1].reshape(len(prediction),1))))
print('Pneumothorax Precision: {}'.format(Precision(true[:,1].reshape(len(true),1),prediction[:,1].reshape(len(prediction),1))))
print('Pneumothorax Recall: {}'.format(Recall(true[:,1].reshape(len(true),1),prediction[:,1].reshape(len(prediction),1))))

print('Edema Accuracy: {}'.format(Accuracy(true[:,2].reshape(len(true),1),prediction[:,2].reshape(len(prediction),1))))
print('Edema Precision: {}'.format(Precision(true[:,2].reshape(len(true),1),prediction[:,2].reshape(len(prediction),1))))
print('Edema Recall: {}'.format(Recall(true[:,2].reshape(len(true),1),prediction[:,2].reshape(len(prediction),1))))

print('Effusion Accuracy: {}'.format(Accuracy(true[:,3].reshape(len(true),1),prediction[:,3].reshape(len(prediction),1))))
print('Effusion Precision: {}'.format(Precision(true[:,3].reshape(len(true),1),prediction[:,3].reshape(len(prediction),1))))
print('Effusion Recall: {}'.format(Recall(true[:,3].reshape(len(true),1),prediction[:,3].reshape(len(prediction),1))))

print('Pneumonia Accuracy: {}'.format(Accuracy(true[:,4].reshape(len(true),1),prediction[:,4].reshape(len(prediction),1))))
print('Pneumonia Precision: {}'.format(Precision(true[:,4].reshape(len(true),1),prediction[:,4].reshape(len(prediction),1))))
print('Pneumonia Recall: {}'.format(Recall(true[:,4].reshape(len(true),1),prediction[:,4].reshape(len(prediction),1))))

print('Cardiomegaly Accuracy: {}'.format(Accuracy(true[:,5].reshape(len(true),1),prediction[:,5].reshape(len(prediction),1))))
print('Cardiomegaly Precision: {}'.format(Precision(true[:,5].reshape(len(true),1),prediction[:,5].reshape(len(prediction),1))))
print('Cardiomegaly Recall: {}'.format(Recall(true[:,5].reshape(len(true),1),prediction[:,5].reshape(len(prediction),1))))
print('=======================================================================')

#Total Label Evaluation
acc = Accuracy(true,prediction)
precision = Precision(true,prediction)
recall = Recall(true,prediction)

print('total acc : {}'.format(acc))
print('total precision : {}'.format(precision))
print('total recall : {}'.format(recall))
print('=======================================================================')


#Sklearn Total Label Evaluation
sklearn_report = classification_report(true,prediction)
sklearn_roc_auc_score = roc_auc_score(true,prediction)

print(sklearn_report)
print('total sklean roc auc score : {}'.format(sklearn_roc_auc_score))
print('=======================================================================')