import tensorflow as tf
from keras import backend as K 
from keras.models import Model
from keras.optimizers import *
from model.utils import input_tensor, single_conv, double_conv, deconv, pooling, merge, callback

# def Mean_IOU_tensorflow_1(y_true, y_pred):
#         nb_classes = K.int_shape(y_pred)[-1]
#         iou = []
#         true_pixels = K.argmax(y_true, axis=-1)
#         pred_pixels = K.argmax(y_pred, axis=-1)
#         void_labels = K.equal(K.sum(y_true, axis=-1), 0)
#         for i in range(0, nb_classes): # exclude first label (background) and last label (void)
#             true_labels = K.equal(true_pixels, i) & ~void_labels
#             pred_labels = K.equal(pred_pixels, i) & ~void_labels
#             inter = tf.to_int32(true_labels & pred_labels)
#             union = tf.to_int32(true_labels | pred_labels)
#             legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
#             ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
#             iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
#         iou = tf.stack(iou)
#         legal_labels = ~tf.debugging.is_nan(iou)
#         iou = tf.gather(iou, indices=tf.where(legal_labels))
#         return K.mean(iou)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


class UNet(Model):
    """ U-Net atchitecture
    Creating a U-Net class that inherits from keras.models.Model
    In initializer, CNN layers are defined using functions from model.utils
    Then parent-initializer is called wuth calculated input and output layers
    Build function is also defined for model compilation and summary
    checkpoint returns a ModelCheckpoint for best model fitting
    """
    def __init__(
        self,
        input_size,
        n_filters,
        pretrained_weights = None
    ):
        # define input layer
        input = input_tensor(input_size)

        # begin with contraction part
        conv1 = double_conv(input, n_filters * 1)
        pool1 = pooling(conv1)

        conv2 = double_conv(pool1, n_filters * 2)
        pool2 = pooling(conv2)

        conv3 = double_conv(pool2, n_filters * 4)
        pool3 = pooling(conv3)

        conv4 = double_conv(pool3, n_filters * 8)
        pool4 = pooling(conv4)

        conv5 = double_conv(pool4, n_filters * 16)

        # expansive path
        up6 = deconv(conv5, n_filters * 8)
        up6 = merge(conv4, up6)
        conv6 = double_conv(up6, n_filters * 8)

        up7 = deconv(conv6, n_filters * 4)
        up7 = merge(conv3, up7)
        conv7 = double_conv(up7, n_filters * 4)

        up8 = deconv(conv7, n_filters * 2)
        up8 = merge(conv2, up8)
        conv8 = double_conv(up8, n_filters * 2)

        up9 = deconv(conv8, n_filters * 1)
        up9 = merge(conv1, up9)
        conv9 = double_conv(up9, n_filters * 1)

        # define output layer
        output = single_conv(conv9, 1, 1)

        # initialize Keras Model with defined above input and output layers
        super(UNet, self).__init__(inputs = input, outputs = output)
        
        # load preatrained weights
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def build(self):
        self.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[dice_coef])
        self.summary()

    def save_model(self, name):
        self.save_weights(name)

    @staticmethod
    def checkpoint(name):
        return callback(name)

        
    
# TODO: FIX SAVING MODEL: AT THIS POINT, ONLY SAVING MODEL WEIGHTS IS AVAILBILE
# SINCE SUBSCLASSING FROM KERAS.MODEL RESTRICTS SAVING MODEL AS AN HDF5 FILE
