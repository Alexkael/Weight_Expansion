#### Load Data ####
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from numpy import linalg as LA

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
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
    return X_train, X_test

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = normalize(x_train, x_test)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

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

from keras import optimizers
from keras import backend as K
import copy

def co_measure(fname,drop_rate):
    K.clear_session()
    model = keras.models.load_model(fname+'/final_139.h5',compile=False)
    sgd = optimizers.SGD_test(add_ah=False, add_noise=0.0, a_corr=np.zeros([12,512,512]), h_corr=np.zeros([12,512,512]), lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    orig_weights = model.get_weights()
    for i in range(len(orig_weights)):
        orig_weights[i] = orig_weights[i]**2
        
    model.set_weights(orig_weights)
    path_loss, path_acc = model.evaluate(x_train, y_train,verbose=0)
    
    return path_acc


lr_ = [0.1, 0.03]
drop_rate_ = [0.0, 0.15, 0.3]
width_ = [1., 0.5]
batch_size_ = [64, 256]
weight_decay_ = [0.0, 0.0005]

act = 'relu'
for net in ['vgg11_relu/','vgg16_relu/','vgg19_relu/']:
    path_norm_ = [] 

    for weight_decay in weight_decay_:
        for drop_rate in drop_rate_:
            for width in width_:
                for batch_size in batch_size_:
                    for lr in lr_:
                        fname = net+str(lr)+'_'+str(drop_rate)+'_'+str(width)+'_'+str(batch_size)+'_'+str(weight_decay)+'_'+str(act)
                        path_norm = co_measure(fname,drop_rate)
                        path_norm_.append(path_norm)
                        print('complete '+fname)
                        
    np.save(net+'path_norm.npy',np.array(path_norm_))
                        
                        
                        
                        
                        
