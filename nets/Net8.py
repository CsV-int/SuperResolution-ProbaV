from tensorflow.keras.layers import Input, LeakyReLU, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Add, Average,Lambda, Reshape
from tensorflow.keras.models import Model

from utils.dataloader import DataHandler
from utils.network_helpers import BaseNet



class Net8(BaseNet):
    def __init__(self, depth = 16, input_im = 9):
        self.depth = depth
        self.input_im = input_im
        return
    
    def build_net(self):
        depth = self.depth
        h_depyh = self.depth / 2
        input_img = Input(shape=(128, 128, self.input_im)) 
        
        xs = Lambda(BaseNet.normalize)(input_img)

        x_l = []
        for i in range(self.input_im):
            x0 = Conv2D(depth , (3, 3), activation='relu', padding='same')(xs[:,:,:,i:i+1])
            x0 = UpSampling2D((2, 2), interpolation = 'bilinear')(x0)
            x0 = LeakyReLU(alpha = 0.2)(x0)
            x0 = Conv2D(depth , (3, 3),padding='same')(x0)
            x0 = UpSampling2D((3, 3), interpolation = 'bilinear')(x0)
            x0 = LeakyReLU(alpha = 0.2)(x0)
            x0 = Conv2D(depth , (3, 3),padding='same')(x0)
            x_l.append(x0)
            
        x = Add()(x_l)
        
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(depth, (3, 3), padding='same')(x)   
        #x = LeakyReLU(alpha = 0.2)(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = Conv2D(1, (3, 3), padding='same')(x)
        
        x = Lambda(BaseNet.denormalize)(x)
        
        autoencoder = Model(input_img, x)
        
        return autoencoder

