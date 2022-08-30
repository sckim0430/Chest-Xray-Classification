"""Data Generator Class
"""

import os
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    """Data Generator Class

    Args:
        keras (class): keras.utils.Sequence
    """
    def __init__(self,list_IDs,labels,dim,data_dir,batch_size=32,n_channels=3,n_classees=15,aug = None,model_name=None,preprocess_input=None,to_categori=None,shuffle=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.dim = dim
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classees = n_classees
        self.aug = aug
        self.preprocess_input = preprocess_input
        self.to_categori = to_categori
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        image_of_array,label_of_onehot = self.__data_generation(list_IDs_temp,labels_temp)
        
        return image_of_array, label_of_onehot
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self,list_IDs_temp,labels_temp):
        """Generate the Data

        Args:
            list_IDs_temp (np.ndarray): image name array
            labels_temp (np.ndarray): label array

        Returns:
            np.ndarray, np.ndarray: image, label array
        """
        image_of_array = np.empty((self.batch_size,*self.dim,self.n_channels),dtype=np.uint8)
        label_of_onehot = np.empty((self.batch_size,self.n_classees),dtype=np.uint8)

        #load image
        for i, ID in enumerate(list_IDs_temp):
            image_of_array[i,] = np.load(os.path.join(self.data_dir,ID)+'.npy')
        
        #augmentation
        if self.aug is not None:
            for index,img in enumerate(image_of_array):
                image_of_array[index,] = self.aug(image=img)['image']
        
        #preprocess
        if self.preprocess_input is not None and self.model_name is not None:
            image_of_array = self.preprocess_input(image_of_array,self.model_name)
        elif self.preprocess_input is not None and self.model_name is None:
            image_of_array = self.preprocess_input(image_of_array)
        else:
            image_of_array = image_of_array/255.
        
        #load label
        if self.to_categori is not None:
            labels = []
            for lst in labels_temp:
                labels.append(list(int(i) for i in lst.split('_')))

            label_of_onehot = np.asarray(self.to_categori(labels,self.n_classees))
        else:
            label_of_onehot = keras.utils.to_categorical(labels_temp)

        return image_of_array,label_of_onehot

