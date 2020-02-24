import tensorflow as tf
from tensorflow.keras.models import Model

from utils.network_helpers import BaseNet

class Net1(BaseNet):
    def __init__(self, depth = 16, input_im = 9, num_rez_block = 5):
        self.depth = depth
        self.input_im = input_im
        self.num_rez_block =  num_rez_block 
        return
    
    def build_net(self):
        depth = self.depth
        input_img = tf.keras.layers.Input(shape=(128, 128, self.input_im)) 
        xs = tf.keras.layers.Lambda(BaseNet.normalize)(input_img)

        x_a = []
        for i in range(0, self.input_im):
            m = tf.keras.layers.Conv2D(depth, 3, padding='same')(xs[:,:,:,i:i+1])

            for i in range(self.num_rez_block):
                m = BaseNet.res_block(m, 3, depth)
            
            x0 = tf.keras.layers.Conv2D(depth , (3, 3), padding='same')(m)
            x0 = tf.keras.layers.Conv2D(depth*4, (1, 1),padding='same')(x0)
            x0 = tf.keras.layers.Conv2D(depth*9, (1, 1),padding='same')(x0)
            x0 = tf.keras.layers.Lambda(BaseNet.pixel_shuffle(3))(x0)
            
            x_0 =x0
            
            x1 = tf.keras.layers.Conv2D(depth , (3, 3), padding='same')(m)
            x1 = tf.keras.layers.Conv2D(depth*4, (1, 1),padding='same')(x1)
            x1 = tf.keras.layers.Conv2D(depth*9, (1, 1),padding='same')(x1)
            x1 = tf.keras.layers.Lambda(BaseNet.pixel_shuffle(3))(x1)
            
            x0 = tf.keras.layers.add([x1,x0])            
            x0 = tf.keras.layers.Conv2D(depth*2 , (1, 1),padding='same')(x0)
            x0 = tf.keras.layers.LeakyReLU(alpha = 0.2)(x0)
            
            
            x1 = tf.keras.layers.add([x_0,x1])            
            x1 = tf.keras.layers.Conv2D(depth*2 , (1, 1),padding='same')(x1)
            x1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(x1)
            
            x3 = tf.keras.layers.add([x0,x1])
            
            x = tf.keras.layers.Conv2D(depth , (3, 3),padding='same')(x3)
            x = tf.keras.layers.LeakyReLU(alpha = 0.2)(x)
            x = tf.keras.layers.Conv2D(depth * 2, (3, 3),padding='same')(x)
            
            x = tf.keras.layers.add([x,x3])
            
            x_a.append(x)
                
        x_l = []
        for i in range(len(x_a)-1):
            x0 = tf.keras.layers.concatenate(x_a[i:i+2])
            x0 = tf.keras.layers.Conv2D(depth*4, (3, 3), padding='same')(x0)
            x0 = tf.keras.layers.Conv2D(depth*2 , (1, 1),padding='same')(x0)
            x0 = tf.keras.layers.LeakyReLU(alpha = 0.2)(x0)
            x_l.append(x0)
            
        x = tf.keras.layers.add(x_l)
        
        encoded = tf.keras.layers.Conv2D(depth*4, (2, 2), padding='same')(x)   
        x = tf.keras.layers.LeakyReLU(alpha = 0.2)(encoded)
        x = tf.keras.layers.Conv2D(depth*2, (1, 1), padding='same')(x)
        x = tf.keras.layers.Conv2D(depth, (5, 5), padding='same')(x)
        x = tf.keras.layers.Conv2D(depth, (3, 3), padding='same')(x)
        x = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(x)
        x = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation = 'sigmoid')(x)
        
        x = tf.keras.layers.Lambda(BaseNet.denormalize)(x)

        autoencoder = Model(input_img, x)
        
        return autoencoder