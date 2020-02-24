import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import Conv2D, UpSampling2D

from utils.network_helpers import BaseNet

class Net4(BaseNet):
    
    def __init__(self, depth = 16, input_im = 9):
        self.depth = depth
        self.input_im = input_im
        return
    
    def build_net(self):
        input_img = tf.keras.layers.Input(shape=(128, 128, self.input_im)) 
        xs = tf.keras.layers.Lambda(BaseNet.normalize)(input_img)
        x = Conv2D(self.depth*3, kernel_size=(3, 3), strides=(1, 1), padding='same')(xs)
        x = Activation('relu')(x)
        x = Conv2D(self.depth*2, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(self.depth, kernel_size=(3, 3), padding='same')(x)
        x = Activation('sigmoid')(x)
        x = UpSampling2D(size=(3, 3))(x)
        x = Conv2D(self.depth*3, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(self.depth*2, kernel_size=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(1, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.Lambda(BaseNet.denormalize)(x)

        return Model(input_img, x)