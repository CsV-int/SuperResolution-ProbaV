import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add

MEAN = 7433.6436/16384.
STD = 2353.0723/16384. 

class BaseNet:   
   
    def res_block(x_in, sz, num_filters):
        linear = 0.8
        x = Conv2D(num_filters, 1, padding='same', activation='relu')(x_in)
        x = Conv2D(int(num_filters * linear), 1, padding='same')(x)
        x = Conv2D(num_filters, sz, padding='same')(x)
        return Add()([x_in, x])
    
    def pixel_shuffle(scale):
        return lambda x: tf.nn.depth_to_space(x, scale)
    
    def normalize(x):
        return tf.where(tf.math.is_nan((x-MEAN)/STD), 0.0, (x-MEAN)/STD)
    
    def denormalize(x):
        return x * STD + MEAN  
