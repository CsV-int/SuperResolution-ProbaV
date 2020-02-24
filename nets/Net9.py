 
#from utils.dataloader import load_best_images, load_mean_images, load_train_images, get_train_test_data, get_train_test_data_augumented
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.models import Model

from utils.network_helpers import BaseNet

class Net9(BaseNet):
    def __init__(self, depth = 16, input_im = 10):
        self.depth = depth
        self.input_im = input_im
        return

    def build_net(self):
        depth = self.depth
        input_img = tf.keras.layers.Input(shape=(128, 128, self.input_im)) 
        xs = tf.keras.layers.Lambda(BaseNet.normalize)(input_img)

        x_a = []
        for i in range(0, self.input_im, 2):
            x0 = tf.keras.layers.Conv2D(depth, (1, 1), padding='same')(xs[:,:,:,i:i+1])
            x0 = tf.keras.layers.Conv2D(depth , (1, 1),padding='same')(x0)
            x0 = tf.keras.layers.LeakyReLU(alpha = 0.2)(x0)
            x0 = tf.keras.layers.Conv2D(depth , (1, 1),padding='same')(x0)
            x0 = tf.keras.layers.LeakyReLU(alpha = 0.4)(x0)
            
            x1 = tf.keras.layers.Conv2D(depth , (1, 1), padding='same')(xs[:,:,:,i+1:i+2])
            x1 = tf.keras.layers.Conv2D(depth , (1, 1),padding='same')(x1)
            x1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(x1)
            x1 = tf.keras.layers.Conv2D(depth , (1, 1),padding='same')(x1)
            x1 = tf.keras.layers.LeakyReLU(alpha = 0.4)(x1)
            
            x = tf.keras.layers.Add()([x0,x1])
            
            x = tf.keras.layers.Conv2D(depth , (3, 3),padding='same')(x)
            x = tf.keras.layers.LeakyReLU(alpha = 0.2)(x)
            
            x_a.append(x)

        x_l = []
        for i in range(len(x_a)-1):
            print(np.shape(x_a[i:i+1][:]))
            x0 = tf.keras.layers.Add()(x_a[i:i+2])
            x0 = tf.keras.layers.Conv2D(depth, (1, 1), padding='same')(x0)
            
            x0 = tf.keras.layers.Conv2D(depth*9, (1, 1),padding='same')(x0)
            x0 = tf.keras.layers.Lambda(BaseNet.pixel_shuffle(3))(x0)

            x0 = tf.keras.layers.Conv2D(depth , (1, 1),padding='same')(x0)
            x0 = tf.keras.layers.LeakyReLU(alpha = 0.2)(x0)
            x_l.append(x0)
            
        x = tf.keras.layers.Add()(x_l)
        
        x = tf.keras.layers.Conv2D(depth, (2, 2), padding='same')(x)   
        x = tf.keras.layers.LeakyReLU(alpha = 0.2)(x)
        x = tf.keras.layers.Conv2D(depth, (1, 1), padding='same')(x)
        x = tf.keras.layers.Conv2D(depth, (1, 1), padding='same')(x)
        x = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(x)
        
        x = tf.keras.layers.Lambda(BaseNet.denormalize)(x)
        
        autoencoder = Model(input_img, x)
        
        return autoencoder 
