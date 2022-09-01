import numpy as np

def load_data():
    data = np.load('/LOCAL//iclr/datasets/Imagenet32_val_npz/val_data.npz')
    y_test = data['labels']-1
    x_test = data['data']

    x_train = np.array([])
    y_train = np.array([])
    
    data = np.load('/LOCAL//iclr/datasets/Imagenet32_train_npz/train_data_batch_1.npz')
    y_train = data['labels']-1
    x_train = data['data']
    
    for image_i in range(2,11):
        data = np.load('/LOCAL//iclr/datasets/Imagenet32_train_npz/train_data_batch_'+str(image_i)+'.npz')
        y_train = np.append(y_train, data['labels']-1)
        x_train = np.append(x_train, data['data'], axis=0)

    del data
    return (x_train, y_train), (x_test, y_test)