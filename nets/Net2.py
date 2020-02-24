import tensorflow as tf
from tensorflow.keras.models import Model

from utils.network_helpers import BaseNet


class Net2(BaseNet):
    
    def __init__(self, depth = 16, input_im = 9, num_rez_block = 5):
        self.depth = depth
        self.input_im = input_im
        self.num_rez_block = num_rez_block  
        return
    
    def build_net(self):
        depth = self.depth
        input_img = tf.keras.layers.Input(shape=(128, 128, self.input_im)) 
        xs = tf.keras.layers.Lambda(BaseNet.normalize)(input_img)

        m = tf.keras.layers.Conv2D(depth*4, 3, padding='same')(xs)
        for j in range(self.num_rez_block):
            m = BaseNet.res_block(m, 3, depth*4) 
        m = tf.keras.layers.Conv2D(3 * 3 ** 2, 3, padding='same')(m)
        m = tf.keras.layers.Lambda(BaseNet.pixel_shuffle(3))(m)
    
        # skip branch
        s = tf.keras.layers.Conv2D(3 * 3 ** 2, 5, padding='same')(xs)
        s = tf.keras.layers.Lambda(BaseNet.pixel_shuffle(3))(s)
        
        x = tf.keras.layers.Add()([m, s])
        
        x = tf.keras.layers.Conv2D(depth, (3, 3), padding='same')(x)
        x = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(x)
        x = tf.keras.layers.Lambda(BaseNet.denormalize)(x)

        return Model(input_img, x)
