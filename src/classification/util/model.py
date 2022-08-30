"""Model Utils
"""
from keras.layers import Dropout, Dense
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten
from keras.engine import Model 
from keras import optimizers
from evalutaion import auc, precision, recall

################
# KERAS  MODEL #
################

def get_eff_model(base, 
            layer, 
            input_shape, 
            classes,
            lr=1e-3,
            activation="sigmoid",
            dropout=None, 
            pooling="avg", 
            weights=None,
            pretrained="noisy-student"):
    """Load Model

    Args:
        base (class): efficientnet model base class
        layer (class): target layer not to train(freeze)
        input_shape (int): input size
        classes (int): num of classes
        lr (float, optional): learing rate. Defaults to 1e-3.
        activation (str, optional): activation function. Defaults to "sigmoid".
        dropout (float, optional): drop out ratio. Defaults to None.
        pooling (str, optional): pooling layer name. Defaults to "avg".
        weights (str, optional): model weight file path to load. Defaults to None.
        pretrained (str, optional): pretrained model. Defaults to "noisy-student".

    Returns:
        class: Loaded Model Class
    """

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

def get_eff_multi_model(base, 
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
    """Load Model

    Args:
        base (class): model base class
        layer (class): target layer not to train(freeze)
        input_shape (int): input size
        classes (int): num of classes
        lr (float, optional): learing rate. Defaults to 1e-3.
        activation (str, optional): activation function. Defaults to "sigmoid".
        dropout (float, optional): drop out ratio. Defaults to None.
        pooling (str, optional): pooling layer name. Defaults to "avg".
        weights (str, optional): model weight file path to load. Defaults to None.
        pretrained (str, optional): pretrained model. Defaults to "noisy-student".

    Returns:
        class: Loaded Model Class
    """

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


def get_multi_model(base_model, 
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