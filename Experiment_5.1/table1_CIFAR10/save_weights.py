import keras
import keras.backend as K
import numpy as np
import os
import keras.backend as K


class save_weights(keras.callbacks.Callback):
    def __init__(self, fname, *kargs, **kwargs):
        super(save_weights, self).__init__(*kargs, **kwargs)
        self.fname = fname
        if not os.path.exists(self.fname):
            os.makedirs(self.fname)
    
    def on_epoch_begin(self, epoch, logs={}):
        if epoch>139:
            K.set_value(self.model.optimizer.add_noise, float(1.0))
            
    def on_epoch_end(self, epoch, logs=None):
        if epoch>139:
            ww = np.array(self.model.get_weights())
            index = []
            for i in range(len(ww)):
                if i%6!=0:
                    index.append(i)
            w = np.delete(ww,index ,axis=0)
            np.save(self.fname+'/w_sample_'+str(epoch-139)+'.npy',w[-4:-1]) 
            
        if epoch in [139]:
            self.model.save(self.fname+'/final_'+str(epoch)+'.h5')