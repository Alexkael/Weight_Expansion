from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Noise_Layer
from keras.layers import Noise_Layer 
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import callmodel
import save_weights
import imagenet

from keras.utils.training_utils import multi_gpu_model
from keras import Model

import tensorflow as tf
import os
    
class vgg:
    def __init__(self, fname, x_train, y_train, x_test, y_test):
        self.num_classes = 1000
        self.x_shape = [32,32,3]
        self.fname = fname
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = self.build_model()
        #self.model = multi_gpu_model(self.model, 2)
        self.model = self.train(self.model)


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        dt=0.5
        
        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Noise_Layer(dt))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Noise_Layer(dt))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Noise_Layer(dt))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Noise_Layer(dt))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Noise_Layer(dt))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Noise_Layer(dt))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Noise_Layer(dt))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Noise_Layer(dt))
        
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Noise_Layer(dt))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        #model.add(Noise_Layer(dt))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model

    def train(self,model):

        #training parameters
        batch_size = 1024
        maxepoches = 100
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        

        def lr_scheduler(epoch):
            if epoch<50:
                return learning_rate * (0.5 ** (epoch // lr_drop))
            else:
                return 0.0005
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
       
        #optimization details
        sgd = optimizers.SGD_test(add_ah=0.0, add_noise=0.0, a_corr=np.zeros([12,512,512]), h_corr=np.zeros([12,512,512]), lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy','top_k_categorical_accuracy'])
        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit(x=x_train,y=y_train,epochs=maxepoches,
                                batch_size=batch_size,
                                validation_data=(x_test, y_test),callbacks=[reduce_lr,save_weights.save_weights(fname=self.fname),callmodel.callmodel(inputs=x_train,outputs=y_train)],verbose=2)
        
        return model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="3" 
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = tf.Session(config=config)

    def normalize(X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        del mean, std
        return X_train, X_test

    (x_train, y_train), (x_test, y_test) = imagenet.load_data()
    x_train = np.reshape(x_train, [x_train.shape[0], 3, 32, 32]).astype('float32') / 255
    x_test = np.reshape(x_test , [x_test.shape[0] , 3, 32, 32]).astype('float32') / 255
    
    x_train = x_train.transpose([0,2,3,1])
    x_test = x_test.transpose([0,2,3,1])
    
    y_train = keras.utils.to_categorical(y_train, 1000)
    y_test = keras.utils.to_categorical(y_test, 1000)
    
    x_train, x_test = normalize(x_train, x_test)
    
    for i in [1]:
        K.clear_session()
        fname = 'rawdata/add_volume'
        model = vgg(fname, x_train, y_train, x_test, y_test)