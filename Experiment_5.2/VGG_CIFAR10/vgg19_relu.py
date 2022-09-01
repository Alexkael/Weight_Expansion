from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
import save_weights
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

from keras import optimizers

import tensorflow as tf
import os
    
class cifar10vgg:
    def __init__(self, lr, dropout_rate, width, batch_size, weight_decay, train=True):
        self.num_classes = 10
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.width = width
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.act = 'relu'
        
        self.x_shape = [32,32,3]
        
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        model = Sequential()
        weight_decay = self.weight_decay
        act = self.act
        
        model.add(Conv2D(int(64*self.width), (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(int(64*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(int(128*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(int(128*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(int(256*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(int(256*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        model.add(Conv2D(int(256*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(int(256*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(int(512*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(int(512*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        model.add(Conv2D(int(512*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(int(512*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(int(512*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(int(512*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        model.add(Conv2D(int(512*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(int(512*self.width), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout_rate))

        model.add(Flatten())
        model.add(Dense(int(512*self.width),kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation(act))
        model.add(BatchNormalization())

        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = self.batch_size
        maxepoches = 190
        learning_rate = self.lr
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            if epoch<140:
                return learning_rate * (0.5 ** (epoch // lr_drop))
            else:
                return 0.0005
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD_test(add_ah=0.0, add_noise=0.0, vgg_alex=0.5, a_corr=np.zeros([12,512,512]), h_corr=np.zeros([12,512,512]), lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        # training process in a for loop with learning rate drop every 25 epoches.
        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,validation_data=(x_test, y_test),callbacks=[reduce_lr,save_weights.save_weights(fname='rawdata/vgg19_relu/'+str(self.lr)+'_'+str(self.dropout_rate)+'_'+str(self.width)+'_'+str(self.batch_size)+'_'+str(self.weight_decay)+'_'+str(self.act))],verbose=2)
        return model
        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="3" 
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    session = tf.Session(config=config)
    
    act = ['relu', 'tanh']
    depth = [11, 16, 19]
    
    lr_ = [0.1, 0.03]
    drop_rate_ = [0.0, 0.15, 0.3]
    width_ = [1., 0.5]
    batch_size_ = [64, 256]
    weight_decay_ = [0.0, 0.0005]
    
    for weight_decay in weight_decay_:
        for drop_rate in drop_rate_:
            for width in width_:
                for batch_size in batch_size_:
                    for lr in lr_:
                        K.clear_session()
                        print('#############################')
                        print('#############################')
                        print('#############################')
                        print('#############################')
                        print('#############################')
                        print('#############################')
                        model = cifar10vgg(lr, drop_rate, width, batch_size, weight_decay)
                        
                        