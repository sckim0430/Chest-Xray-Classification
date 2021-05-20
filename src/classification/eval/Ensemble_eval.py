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
from keras.applications.densenet import DenseNet169
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
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

def convert_to_label(Target):
    
    tmp_target = []

    for t in Target:
        tmp_target.append(t.split('|'))
    
    label = []

    for t in tmp_target:
        sub_label = []
        for target in t:
            if target == 'Consolidation':
                sub_label.append(0)
            elif target == 'Pneumothorax':
                sub_label.append(1)
            elif target == 'Edema':
                sub_label.append(2)
            elif target == 'Effusion':
                sub_label.append(3)
            elif target == 'Pneumonia':
                sub_label.append(4)
            elif target == 'Cardiomegaly':
                sub_label.append(5)
            else:
                print('Target Error! you should look csv file!')
                return None
        
        sub_label.sort()

        label_sub_str = ''

        for index,val in enumerate(sub_label):
            if index == len(sub_label)-1:
                label_sub_str += '{}'.format(val)
            else:
                label_sub_str += '{}_'.format(val)

        label.append(label_sub_str)        

    return label

#PREPROCESSING
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

#TTA
def TTA(img,model,model_name,seed=88,niter=4):
    
    input_size = int(model.get_input_at(0).get_shape()[1])

    AUGMENTATIONS = Compose([
    HorizontalFlip(p=0.25),
    RandomSizedCrop(min_max_height =(int(input_size*0.75),input_size),height=input_size,width=input_size,p=0.5),
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

    np.random.seed(seed)
    original_img = img.copy()
    inverted_img = np.invert(img.copy())
    hflipped_img = np.fliplr(img.copy())
    original_img_array = np.empty((niter+1, img.shape[0], img.shape[1], img.shape[2]))
    inverted_img_array = original_img_array.copy() 
    hflipped_img_array = original_img_array.copy()
    original_img_array[0] = original_img 
    inverted_img_array[0] = inverted_img 
    hflipped_img_array[0] = hflipped_img
    for each_iter in range(niter):
        original_img_array[each_iter+1] = AUGMENTATIONS(image=original_img)['image']
        inverted_img_array[each_iter+1] =  AUGMENTATIONS(image=inverted_img)['image']
        hflipped_img_array[each_iter+1] = AUGMENTATIONS(image=hflipped_img)['image']
    tmp_array = np.vstack((original_img_array, inverted_img_array, hflipped_img_array))
    tmp_array = preprocess_input(tmp_array, model_name)
    
    prediction = np.mean(model.predict(tmp_array),axis=0)

    return prediction

#Evaluation
def Accuracy(y_true,y_pred):

    all_accuracy = np.zeros(y_true.shape[0])

    for index_of_img,each_img in enumerate(y_true):
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for index_of_label,label in enumerate(each_img):
            if label == 1 and label == y_pred[index_of_img][index_of_label]:
                tp = tp + 1
            elif label == 1 and label != y_pred[index_of_img][index_of_label]:
                fn = fn + 1
            elif label == 0 and label == y_pred[index_of_img][index_of_label]:
                tn = tn + 1
            elif label == 0 and label != y_pred[index_of_img][index_of_label]:
                fp = fp + 1

        try:
            all_accuracy[index_of_img]  = float(tp+tn)/(tp+tn+fp+fn)
        except ZeroDivisionError:
            all_accuracy[index_of_img] = 0
    
    return np.mean(all_accuracy)

def Accuracy_Each_Label(y_true,y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for index_of_img,each_img in enumerate(y_true):    
        if each_img == 1 and each_img == y_pred[index_of_img]:
            tp = tp + 1
        elif each_img == 1 and each_img != y_pred[index_of_img]:
            fn = fn + 1
        elif each_img == 0 and each_img == y_pred[index_of_img]:
            tn = tn + 1
        elif each_img == 0 and each_img != y_pred[index_of_img]:
            fp = fp + 1

    try:
        all_accuracy = float(tp+tn)/(tp+tn+fp+fn)
    except ZeroDivisionError:
        all_accuracy = 0
    
    return all_accuracy

def Precision(y_true,y_pred):

    all_precision = np.zeros(y_true.shape[0])

    for index_of_img,each_img in enumerate(y_true):
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for index_of_label,label in enumerate(each_img):
            if label == 1 and label == y_pred[index_of_img][index_of_label]:
                tp = tp + 1
            elif label == 1 and label != y_pred[index_of_img][index_of_label]:
                fn = fn + 1
            elif label == 0 and label == y_pred[index_of_img][index_of_label]:
                tn = tn + 1
            elif label == 0 and label != y_pred[index_of_img][index_of_label]:
                fp = fp + 1

        try:
            all_precision[index_of_img]  = float(tp)/(tp+fp)
        except ZeroDivisionError:
            all_precision[index_of_img] = 0

    return np.mean(all_precision)

def Precision_Each_Label(y_true,y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for index_of_img,each_img in enumerate(y_true):    
        if each_img == 1 and each_img == y_pred[index_of_img]:
            tp = tp + 1
        elif each_img == 1 and each_img != y_pred[index_of_img]:
            fn = fn + 1
        elif each_img == 0 and each_img == y_pred[index_of_img]:
            tn = tn + 1
        elif each_img == 0 and each_img != y_pred[index_of_img]:
            fp = fp + 1

    try:
        all_precision = float(tp)/(tp+fp)
    except ZeroDivisionError:
        all_precision = 0
    
    return all_precision

def Recall(y_true,y_pred):

    all_recall = np.zeros(y_true.shape[0])

    for index_of_img,each_img in enumerate(y_true):
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for index_of_label,label in enumerate(each_img):
            if label == 1 and label == y_pred[index_of_img][index_of_label]:
                tp = tp + 1
            elif label == 1 and label != y_pred[index_of_img][index_of_label]:
                fn = fn + 1
            elif label == 0 and label == y_pred[index_of_img][index_of_label]:
                tn = tn + 1
            elif label == 0 and label != y_pred[index_of_img][index_of_label]:
                fp = fp + 1
        try:
            all_recall[index_of_img]  = float(tp)/(tp+fn)
        except ZeroDivisionError:
            all_recall[index_of_img] = 0

    return np.mean(all_recall)

def Recall_Each_Label(y_true,y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for index_of_img,each_img in enumerate(y_true):    
        if each_img == 1 and each_img == y_pred[index_of_img]:
            tp = tp + 1
        elif each_img == 1 and each_img != y_pred[index_of_img]:
            fn = fn + 1
        elif each_img == 0 and each_img == y_pred[index_of_img]:
            tn = tn + 1
        elif each_img == 0 and each_img != y_pred[index_of_img]:
            fp = fp + 1

    try:
        all_recall = float(tp)/(tp+fn)
    except ZeroDivisionError:
        all_recall = 0
    
    return all_recall


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
model_file_names = []

for _ in os.listdir(SETTINGS_JSON['MODEL_PREDICT_DIR']):
    if os.path.splitext(_)[-1] == '.h5':
        model_file_names.append(os.path.splitext(_)[0])


#get classification model
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

#predict target
path_to_test_images = SETTINGS_JSON['TEST_IMAGES_DIR']
true_df = pd.read_csv(SETTINGS_JSON['TEST_CSV_DIR'],keep_default_na=False)
####################################################
# test_num = 1000
# true_df = true_df[:test_num]
####################################################

test_image_names = []

for _ in true_df.Image:
    test_image_names.append(_)

test_predictions = []

KERNEL = np.ones((15, 15), np.uint8)

#total prediction
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
    
    test_individual_predictions = []

    #each model prediction mean
    for model_index,model in enumerate(models):
        model_name = model_file_names[model_index].split('_')[0]
        model_input_size = int(model.get_input_at(0).get_shape()[1])
        resized_image = test_image.resize((model_input_size,model_input_size))
        resized_image = np.array(resized_image,'uint8')
        # test_individual_predictions.append(TTA(resized_image,model,model_name))
        ########################################################################
        resized_image = preprocess_input(resized_image,model_name)
        resized_image = np.expand_dims(resized_image,axis=0)
        test_individual_predictions.append(model.predict(resized_image))
        ########################################################################
        
    test_predictions.append(test_individual_predictions)

#model prediction mean
prediction = np.zeros((len(test_predictions),PARAMS_JSON['CLASS_NUM']))

for index,pred in enumerate(test_predictions):
    prediction[index] = np.mean(pred,axis=0)

prediction = np.array(prediction>0.5,dtype=np.uint8)

labels = []

for lst in list(convert_to_label(true_df.Labels)):
    labels.append(list(int(i) for i in lst.split('_')))

true = np.asarray(to_categori(labels,PARAMS_JSON['CLASS_NUM']))

#each label evaluation
print('=======================================================================')
print('Consolidation Accuracy: {}'.format(Accuracy_Each_Label(true[:,0],prediction[:,0])))
print('Consolidation Precision: {}'.format(Precision_Each_Label(true[:,0],prediction[:,0])))
print('Consolidation Recall: {}'.format(Recall_Each_Label(true[:,0],prediction[:,0])))

print('Pneumothorax Accuracy: {}'.format(Accuracy_Each_Label(true[:,1],prediction[:,1])))
print('Pneumothorax Precision: {}'.format(Precision_Each_Label(true[:,1],prediction[:,1])))
print('Pneumothorax Recall: {}'.format(Recall_Each_Label(true[:,1],prediction[:,1])))

print('Edema Accuracy: {}'.format(Accuracy_Each_Label(true[:,2],prediction[:,2])))
print('Edema Precision: {}'.format(Precision_Each_Label(true[:,2],prediction[:,2])))
print('Edema Recall: {}'.format(Recall_Each_Label(true[:,2],prediction[:,2])))

print('Effusion Accuracy: {}'.format(Accuracy_Each_Label(true[:,3],prediction[:,3])))
print('Effusion Precision: {}'.format(Precision_Each_Label(true[:,3],prediction[:,3])))
print('Effusion Recall: {}'.format(Recall_Each_Label(true[:,3],prediction[:,3])))

print('Pneumonia Accuracy: {}'.format(Accuracy_Each_Label(true[:,4],prediction[:,4])))
print('Pneumonia Precision: {}'.format(Precision_Each_Label(true[:,4],prediction[:,4])))
print('Pneumonia Recall: {}'.format(Recall_Each_Label(true[:,4],prediction[:,4])))

print('Cardiomegaly Accuracy: {}'.format(Accuracy_Each_Label(true[:,5],prediction[:,5])))
print('Cardiomegaly Precision: {}'.format(Precision_Each_Label(true[:,5],prediction[:,5])))
print('Cardiomegaly Recall: {}'.format(Recall_Each_Label(true[:,5],prediction[:,5])))
print('=======================================================================')

#total evaluation
acc = Accuracy(true,prediction)
precision = Precision(true,prediction)
recall = Recall(true,prediction)

print('total acc : {}'.format(acc))
print('total precision : {}'.format(precision))
print('total recall : {}'.format(recall))
print('=======================================================================')


#sklearn evaluation
sklearn_report = classification_report(true,prediction)
sklearn_roc_auc_score = roc_auc_score(true,prediction)

print(sklearn_report)
print('total sklean roc auc score : {}'.format(sklearn_roc_auc_score))
print('=======================================================================')