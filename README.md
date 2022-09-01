**Code for paper Weight Expansion: A New Perspective on Dropout and Generalization**

**version:**  
python 3.7  
tensorflow 1.13.1-gpu  
keras 2.2.4

We define Noise_Layer(noise_a) in core.py line: 139 - 176  
Add Noise_Layer in /LOCAL/usr/anaconda3/lib/python3.7/site-packages/keras/layers/core.py

We define Noise_Layer(noise_a) in tensorflow_backend.py line: 3447 - 3466  
Add Noise_Layer in /LOCAL/usr/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py

We define Noise_Layer(noise_a) in nn_ops.py line: 3060 - 3148  
Add Noise_Layer in /LOCAL/usr/anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/nn_ops.py

We define SGD_test(noise_h) in optimizers.py line: 214 - 352  
Add SGD_test in /LOCAL/usr/anaconda3/lib/python3.7/site-packages/keras/optimizers.py

**Experiment_5.1:**
               
              training scripts and results are saved in Experiment_5.1/table1_CIFAR10

**Experiment_5.2:** 
               
              training scripts and results are saved in Experiment_5.2/VGG_CIFAR10

**Experiment_5.3:**
               
              training scripts and results are saved in Experiment_5.3/AlexNet_ImageNet32 or VGG16_imagenet32
