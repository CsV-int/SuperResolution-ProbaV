import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Conv3D,Lambda,Add,LeakyReLU, Dropout

from utils.network_helpers import BaseNet
class Net11(BaseNet):   
    def __init__(self, depth = 16, input_im = 9):
        self.depth = depth
        self.input_im = input_im
        return
    
    def build_net(self):
        depth = self.depth
        input_img = tf.keras.layers.Input(shape=(128,128,self.input_im))

        xs = Lambda(BaseNet.normalize)(input_img)

        x0 = Conv2D(depth ,(7,7), padding = 'same')(xs)
        x1 = Conv2D(depth ,(5,5), padding = 'same')(xs)
        x2 = Conv2D(depth ,(3,3), padding = 'same')(xs)
        x3 = Conv2D(depth ,(1,1), padding = 'same')(xs)
        
        m0 = BaseNet.res_block(x0, 3, depth)
        m1 = BaseNet.res_block(x1, 3, depth)
        m2 = BaseNet.res_block(x2, 3, depth)
        m3 = BaseNet.res_block(x3, 3, depth)
        
        x1 = Add()([x0, x1, x2, x3])
        x2 = Add()([m0, m1, m2, m3])
        
        x = Add()([x1,x2])
        
        x = Conv2D(depth *9,(3,3), padding = 'same')(x)
        
        x = Lambda(BaseNet.pixel_shuffle(3))(x)
        
        x = Conv2D(depth ,(3,3), padding = 'same')(x)
        x = Conv2D(depth ,(1,1), padding = 'same')(x)
        
        x = Conv2D(1, (1,1), padding = 'same')(x)
        
        x = Lambda(BaseNet.denormalize)(x)
        
        return Model(input_img, x)
    
class Net11v2(BaseNet):   
    def __init__(self, depth = 16, input_im = 9, num_rez_block = 5):
        self.depth = depth
        self.input_im = input_im
        self.num_rez_block = num_rez_block
        return
    
    def build_net(self):
        depth = self.depth
        input_img = tf.keras.layers.Input(shape=(128,128,self.input_im))
        
        xs = Lambda(BaseNet.normalize)(input_img)

        x0 = Conv2D(depth ,(7,7), padding = 'same')(xs)
        x1 = Conv2D(depth ,(5,5), padding = 'same')(xs)
        x2 = Conv2D(depth ,(3,3), padding = 'same')(xs)
        x3 = Conv2D(depth ,(1,1), padding = 'same')(xs)
        
        m0 = BaseNet.res_block(x0, 3, depth)
        m1 = BaseNet.res_block(x1, 3, depth)
        m2 = BaseNet.res_block(x2, 3, depth)
        m3 = BaseNet.res_block(x3, 3, depth)
        
        x1 = Add()([x0, x1, x2, x3])
        x2 = Add()([m0, m1, m2, m3])
        
        x = Add()([x1,x2])

        x = Conv2D(depth *9,(3,3), padding = 'same')(x)
        x = Lambda(BaseNet.pixel_shuffle(3))(x)
        
        for i in range(self.num_rez_block):
            x = BaseNet.res_block(x, 3, depth)
        
        x = Conv2D(depth ,(3,3), padding = 'same')(x)
        x = Conv2D(depth ,(1,1), padding = 'same')(x)
        
        x = Conv2D(1, (1,1), padding = 'same')(x)
        
        x = Lambda(BaseNet.denormalize)(x)
        
        return Model(input_img, x)
    
class Net11v3(BaseNet):   
    def __init__(self, depth = 16, input_im = 9, num_rez_block = 5):
        self.depth = depth
        self.input_im = input_im
        self.num_rez_block = num_rez_block 
        return
  
    def build_net(self):
        depth = self.depth
        input_img = tf.keras.layers.Input(shape=(128,128,self.input_im))
       
        xs = Lambda(BaseNet.normalize)(input_img)

        x0 = Conv2D(depth ,(7,7), padding = 'same')(xs)
        x1 = Conv2D(depth ,(5,5), padding = 'same')(xs)
        x2 = Conv2D(depth ,(3,3), padding = 'same')(xs)
        x3 = Conv2D(depth ,(1,1), padding = 'same')(xs)
        
        m0 = BaseNet.res_block(x0, 7, depth)
        m1 = BaseNet.res_block(x1, 5, depth)
        m2 = BaseNet.res_block(x2, 3, depth)
        m3 = BaseNet.res_block(x3, 1, depth)
        
        x1 = Add()([x0, x1, x2, x3])
        x2 = Add()([m0, m1, m2, m3])
        
        x = Add()([x1,x2])

        for i in range(self.num_rez_block):
            x = BaseNet.res_block(x, 3,depth)

        x = Add()([x,x1])
        
        x = Conv2D(depth *9,(3,3), padding = 'same')(x)
        x = Lambda(BaseNet.pixel_shuffle(3))(x)
                
        
        x = Conv2D(1, (1,1), padding = 'same')(x)
        
        x = Lambda(BaseNet.denormalize)(x)

        
        return Model(input_img, x)
    