import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Lambda,Add,LeakyReLU, Dropout,MaxPooling2D

from utils.network_helpers import BaseNet

class Net6(BaseNet):
    
    def __init__(self, depth = 16, input_im = 9):
        self.depth = depth
        self.input_im = input_im
        return

    def build_net(self):
        depth = self.depth
        input_img = tf.keras.layers.Input(shape=(128,128,self.input_im))
        x = Conv2D(depth * 9,(5,5), padding = 'same')(input_img)
        x = Lambda(BaseNet.pixel_shuffle(3))(x)
        x = Conv2D(depth * 4,(7,7), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        
        x1 = x
        
        x = Conv2D(depth * 9,(5,5), padding = 'same')(x)
        x = Lambda(BaseNet.pixel_shuffle(3))(x)
        x = Conv2D(depth * 4,(7,7), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
      
        x = Conv2D(depth * 9,(5,5), padding = 'same')(x)
        x = Lambda(BaseNet.pixel_shuffle(3))(x)
        x = Conv2D(depth * 4,(7,7), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
                
        x = Conv2D(depth * 4,(7,7), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        
        x = Conv2D(depth * 4,(5,5), padding = 'same')(x)
        x = MaxPooling2D((3, 3), padding='same')(x)
        x = Conv2D(depth * 4,(7,7), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        
        x = Conv2D(depth * 4,(5,5), padding = 'same')(x)
        x = MaxPooling2D((3, 3), padding='same')(x)
        x = Conv2D(depth * 4,(7,7), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        
        x = Add()([x,x1])
        
        x = Conv2D(int(depth * 0.5),(3,3), padding = 'same')(x)
        x = Conv2D(1,(3,3), padding = 'same')(x)
        
        return Model(input_img, x)
        
        
        
        