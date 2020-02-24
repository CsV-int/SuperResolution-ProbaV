import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Lambda,Add,LeakyReLU, Dropout

from utils.network_helpers import BaseNet

class Net5(BaseNet):
    
    def __init__(self, depth = 16, input_im = 9):
        self.depth = depth
        self.input_im = input_im
        return
    
    def build_net(self):
        depth = self.depth
        input_img = tf.keras.layers.Input(shape=(128,128,self.input_im))
        
        xs = tf.keras.layers.Lambda(BaseNet.normalize)(input_img)

        
        x = Conv2D(depth * 9,(5,5), padding = 'same')(xs)
        x = Lambda(BaseNet.pixel_shuffle(3))(x)
        x = Conv2D(depth * 4,(7,7), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        
        s = Conv2D(depth * 9 * 4,(5,5), padding = 'same')(xs)
        s = Lambda(BaseNet.pixel_shuffle(3))(s)
        s = Conv2D(depth * 4,(5,5), padding = 'same')(s)
        x = Add()([s,x])
        
        x1 = x
        
        x = Conv2D(depth * 2,(5,5), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        
        s = Conv2D(depth * 9 * 2,(5,5), padding = 'same')(xs)
        s = Lambda(BaseNet.pixel_shuffle(3))(s)
        s = Conv2D(depth * 2,(5,5), padding = 'same')(s)
        x = Add()([s,x])
        
        x2 = x
        
        x = Conv2D(depth * 1,(3,3), padding = 'same')(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        
        s = Conv2D(depth * 9,(5,5), padding = 'same')(xs)
        s = Lambda(BaseNet.pixel_shuffle(3))(s)
        s = Conv2D(depth,(5,5), padding = 'same')(s)
        
        x1 = Conv2D(depth,(5,5), padding = 'same')(x1)
        x2 = Conv2D(depth,(5,5), padding = 'same')(x2)
        
        x = Add()([s,x,x1,x2])
        
        x = Conv2D(int(depth * 0.5),(3,3), padding = 'same')(x)
        x = Conv2D(1,(3,3), padding = 'same')(x)
        
        x = tf.keras.layers.Lambda(BaseNet.denormalize)(x)
        
        return Model(input_img, x)
        
        
        
        