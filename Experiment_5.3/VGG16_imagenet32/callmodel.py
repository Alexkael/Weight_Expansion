import keras
import keras.backend as K
import numpy as np
import time
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
import build_model

class callmodel(keras.callbacks.Callback):
    def __init__(self, inputs=None, outputs=None, *kargs, **kwargs):
        super(callmodel, self).__init__(*kargs, **kwargs)
        self.inputs = inputs
        self.outputs = outputs
    
    def on_epoch_begin(self, epoch, logs={}):
        def get_layer_output_grad(model, inputs, outputs, layer):
            """ Gets gradient a layer output for given inputs and outputs"""
            grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
            symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
            f = K.function(symb_inputs, grads)
            x, y, sample_weight = model._standardize_user_data(inputs, outputs)
            output_grad = f(x + y + sample_weight)
            return output_grad
        
        ##############################h_value#######################################
        
        if epoch == 49:
            K.set_value(self.model.optimizer.add_ah, np.array(0.0))
        elif epoch == 20:
            mmodel = build_model.build_model()
            mmodel.set_weights(self.model.get_weights())
            sgd = optimizers.SGD_test(add_ah=0., add_noise=0., h_corr=np.zeros([12,512,512]),a_corr=np.zeros([12,512,512]), lr=0.001, momentum=0.9, nesterov=True)
            mmodel.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
            
            index_input = [np.random.randint(0,50000) for _ in range(100)]
            cov = []
            for l_i in [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]:
                h_value = get_layer_output_grad(mmodel, [self.inputs[index_input,:]], [self.outputs[index_input,:]], l_i)[0]
                h_value = h_value.reshape(h_value.shape[0]*h_value.shape[1]*h_value.shape[2],-1)
                h_value = np.mean(h_value,axis=0)
                cov.append(h_value)
            
            h_corr = []
            for l_i in range(len(cov)):
                split_data = 64
                zz = np.zeros([512,512])
                for i in range(int(len(cov[l_i])/split_data)):
                    z = np.kron(cov[l_i][i*split_data:i*split_data+split_data], np.array([cov[l_i][i*split_data:i*split_data+split_data]]).reshape(-1,1))
                    z = z / (z.max(axis=1).reshape(-1,1)+0.0001) + np.random.normal(loc=0.0, scale=0.2, size=(split_data,split_data))
                    z = np.linalg.inv(z)
                    z = z * (-np.ones([len(z),len(z)]) + 2*np.eye(len(z), dtype=float))
                    #z = np.abs(z)  
                    z = z / np.abs(z).sum(axis=1).reshape(-1,1)
                    if not np.isnan(np.min(z)):
                        zz[i*split_data:i*split_data+split_data,i*split_data:i*split_data+split_data] = z
                    else:
                        zz[i*split_data:i*split_data+split_data,i*split_data:i*split_data+split_data] = np.eye(split_data, dtype=float)
                h_corr.append(zz)
            K.set_value(self.model.optimizer.h_corr, np.array(h_corr))
            

        