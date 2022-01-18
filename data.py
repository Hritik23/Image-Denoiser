import tensorflow as tf
import numpy as np

def load_data():
    (x_train, _), (x_test, _)=tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train=x_train/255
    x_test=x_test/255
    x_train=x_train.reshape(60000, 28,28,1)
    x_test=x_test.reshape(10000, 28,28,1)
    
    return (x_train,x_test)

def add_gaussian_noise(noise_factor, x_train, x_test):
    #adding noise to the data
    
    x_train_noisy=x_train+noise_factor*np.random.normal(loc=0, scale=1, size=x_train.shape)
    x_test_noisy=x_test+noise_factor*np.random.normal(loc=0, scale=1, size=x_test.shape)

    x_train_noisy=np.clip(x_train_noisy, 0,1)
    x_test_noisy=np.clip(x_test_noisy, 0,1)
    
    return (x_train_noisy, x_test_noisy)