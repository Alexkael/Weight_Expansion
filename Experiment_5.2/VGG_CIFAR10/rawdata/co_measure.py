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
    ori_loss, ori_acc = model.evaluate(x_train, y_train,verbose=0)
    tst_loss, tst_acc = model.evaluate(x_test, y_test,verbose=0)
    
    ##############################################f_norm,p_norm#########################################################
    model_initial = keras.models.load_model(fname+'/initial.h5',compile=False)
    sgd = optimizers.SGD_test(add_ah=False, add_noise=0.0, a_corr=np.zeros([12,512,512]), h_corr=np.zeros([12,512,512]), lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model_initial.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    initial_weights = model_initial.get_weights()
    
    sum_f_norm = 0
    sum_p_norm = 0
    pro_f_norm = 1
    pro_p_norm = 1
    for jj in range(len(orig_weights)):
        if len(orig_weights[jj].shape)>1:
            sum_f_norm += LA.norm(orig_weights[jj]-initial_weights[jj])
            sum_p_norm += LA.norm(orig_weights[jj])
            pro_f_norm *= LA.norm(orig_weights[jj]-initial_weights[jj])
            pro_p_norm *= LA.norm(orig_weights[jj])
    
    ##############################################spectral_norm, nop###################################################
    nop = 0
    sum_s_norm = 0
    pro_s_norm = 1
    for i in range(len(orig_weights)):
        wf = orig_weights[i]-initial_weights[i]
        if len(wf.shape) == 2:
            if wf.shape[0] > wf.shape[1]:
                aa = wf.T.dot(wf)
            else:
                aa = wf.dot(wf.T)     
            sum_s_norm += np.sqrt(np.max(np.sum(np.abs(aa),axis=1)))
            pro_s_norm *= np.sqrt(np.max(np.sum(np.abs(aa),axis=1)))
            nop += wf.shape[0] * wf.shape[1]
            
        elif len(wf.shape) == 4:
            aa = 0
            for w1 in wf:
                for w2 in w1:
                    zz = w2.dot(w2.T)
                    aa = aa + np.max(np.sum(np.abs(zz),axis=1))
            sum_s_norm += np.sqrt(aa)
            pro_s_norm *= np.sqrt(aa)
            nop += wf.shape[0] * wf.shape[1] * wf.shape[2] * wf.shape[3]
    
    
    
    ##############################################laplace############################################################
    def get_weight_grad(model, inputs, outputs):
        """ Gets gradient of model for given inputs and outputs for all weights"""
        grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y + sample_weight)
        return output_grad

    def get_layer_output_grad(model, inputs, outputs, layer=-1):
        """ Gets gradient a layer output for given inputs and outputs"""
        grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y + sample_weight)
        return output_grad

    def tran_a(a_1):
        a_1 = np.mean(a_1, axis=0)
        a_1 = a_1.reshape(a_1.shape[0]*a_1.shape[1],-1)
        a_1 = np.mean(a_1,axis=0)
        return a_1

    def tran_h(h_2):
        h_2 = np.mean(h_2, axis=0)
        h_2 = h_2.reshape(h_2.shape[0]*h_2.shape[1],-1)
        h_2 = np.mean(h_2,axis=0)+10**-13
        return h_2

    def load(a):
        a_1 = []
        h_2 = []
        num = 0
        layer_1 = K.function([model.layers[0].input], [model.layers[a].output])
        for num in [0,20,40,60,80,100,120]:
            a_1.append(tran_a(layer_1([x_train[num:num+20]])[0]))

        for num in [0,20,40,60,80,100,120]:
            h_2.append(tran_h(get_layer_output_grad(model, x_train[num:num+20], y_train[num:num+20], a+1)[0]))
        return a_1, h_2
    
    a_1, h_2 = load(-13)
    split = 16
    deet_a = []
    for i in range(int(len(a_1[0])/split)):
        a = np.zeros((split,split))
        for e in range(50):
            noise = (np.random.uniform(0,1,split)>drop_rate).astype(float)
            a += np.kron(a_1[0][i*split:i*split+split]*noise, np.array([a_1[0][i*split:i*split+split]*noise]).reshape(-1,1))
            for j in range(1,len(a_1)):
                a += np.kron(a_1[j][i*split:i*split+split]*noise, np.array([a_1[j][i*split:i*split+split]*noise]).reshape(-1,1))    
        a = np.abs(np.linalg.inv(a/50))
        a = a/np.max(a)
        for j in range(len(a)):
            a[j][j] = 1
        deet_a.append(np.linalg.det(a))
    deet_a = np.mean(deet_a)


    deet_h = []
    for e in range(20):
        for i in range(int(len(a_1[0])/split)):
            a = np.kron(h_2[0][i*split:i*split+split], np.array([h_2[0][i*split:i*split+split]]).reshape(-1,1))  
            for j in range(1,len(h_2)):
                a += np.kron(h_2[j][i*split:i*split+split], np.array([h_2[j][i*split:i*split+split]]).reshape(-1,1))
            a = a/np.max(a)
            a += np.random.normal(loc=0.0, scale=0.00001, size=(len(a),len(a)))
            a = np.abs(np.linalg.inv(a))
            a = a/np.max(a)
            for j in range(len(a)):
                a[j][j] = 1
            deet_h.append(np.linalg.det(a))
    deet_h = np.mean(deet_h)     

    ##############################################sharpness############################################################
    def add_noise(weights, sigma):
        for i in range(len(weights)):
            n = np.random.normal(0, sigma, weights[i].shape)
            weights[i] += weights[i]*n
        model.set_weights(weights)
        index = np.random.randint(0,50000,5000)
        loss, acc = model.evaluate(x_train[index], y_train[index],verbose=0)
        return acc

    lower = 0
    sigma = 0.1    
    acc = add_noise(copy.deepcopy(orig_weights), sigma)

    ssigma = []
    aacc = []
    ssigma.append(sigma)
    aacc.append(acc)
    for i in range(10):
        if ori_acc-acc>0.05:
            sigma = sigma-(sigma-lower)/2
            acc1 = []
            for j in range(5):
                acc1.append(add_noise(copy.deepcopy(orig_weights), sigma))
            acc = min(acc1)
        else:
            a = (sigma - lower)/2
            lower = copy.deepcopy(sigma)
            sigma = sigma*1.1
            acc1 = []
            for j in range(5):
                acc1.append(add_noise(copy.deepcopy(orig_weights), sigma))
            acc = min(acc1)
        ssigma.append(sigma)
        aacc.append(acc)

    aacc = np.array(aacc)
    aacc = aacc - ori_acc + 0.05
    minacc = 1
    mindex = 0
    for i in range(len(aacc)):
        if aacc[i]>0 and aacc[i]<minacc:
            minacc = aacc[i]
            mindex = i


    return sum_f_norm, pro_f_norm, sum_s_norm, pro_s_norm, sum_p_norm, pro_p_norm, nop, ssigma[mindex], deet_a*deet_h,tst_loss-ori_loss


lr_ = [0.1, 0.03]
drop_rate_ = [0.0, 0.15, 0.3]
width_ = [1., 0.5]
batch_size_ = [64, 256]
weight_decay_ = [0.0, 0.0005]

act = 'relu'
for net in ['vgg11_relu/','vgg16_relu/','vgg19_relu/']:
    sum_f_norm_ = [] 
    pro_f_norm_ = [] 
    sum_s_norm_ = [] 
    pro_s_norm_ = [] 
    sum_p_norm_ = [] 
    pro_p_norm_ = [] 
    nop_ = [] 
    sigma_ = []
    laplace_det_ = []
    ge_ = []
    for weight_decay in weight_decay_:
        for drop_rate in drop_rate_:
            for width in width_:
                for batch_size in batch_size_:
                    for lr in lr_:
                        fname = net+str(lr)+'_'+str(drop_rate)+'_'+str(width)+'_'+str(batch_size)+'_'+str(weight_decay)+'_'+str(act)
                        sum_f_norm, pro_f_norm, sum_s_norm, pro_s_norm, sum_p_norm, pro_p_norm, nop, sigma, laplace_det,ge = co_measure(fname,drop_rate)
                        sum_f_norm_.append(sum_f_norm)
                        pro_f_norm_.append(pro_f_norm)
                        sum_s_norm_.append(sum_s_norm)
                        pro_s_norm_.append(pro_s_norm)
                        sum_p_norm_.append(sum_p_norm)
                        pro_p_norm_.append(pro_p_norm)
                        nop_.append(nop)
                        sigma_.append(sigma)
                        laplace_det_.append(laplace_det)
                        ge_.append(ge)
                        print('complete '+fname)
                        
    np.save(net+'sum_f_norm.npy',np.array(sum_f_norm_))
    np.save(net+'pro_f_norm.npy',np.array(pro_f_norm_))
    np.save(net+'sum_s_norm.npy',np.array(sum_s_norm_))
    np.save(net+'pro_s_norm.npy',np.array(pro_s_norm_))
    np.save(net+'sum_p_norm.npy',np.array(sum_p_norm_))
    np.save(net+'pro_p_norm.npy',np.array(pro_p_norm_))
    np.save(net+'nop.npy',np.array(nop_))
    np.save(net+'sigma.npy',np.array(sigma_))
    np.save(net+'laplace_det.npy',np.array(laplace_det_))
    np.save(net+'ge.npy',np.array(ge_))
                        
                        
                        
                        
                        
