import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from utils.dataloader import DataHandler


#de inlocuit hr cu sr

class Losses():
    @staticmethod
    def sliding_win_loss(sr, hr):
        
        hr_, sr_ = Losses.apply_mask(hr,sr)
        border = 3
        cropped_sr=sr_[:,border:384-border,border:384-border]

        X=[]
        for i in range((2*border)+1):
            for j in range((2*border)+1):
                cropped_hr = hr_[:,i:i+(384-(2*border)),j:j+(384-(2*border))]
                X.append( Losses.cMSE(cropped_hr, cropped_sr) )
        X=tf.stack(X)
        #Take the minimum mse
        minim=tf.reduce_min(X,axis=0)
        mse=tf.reduce_mean(minim)
        return mse
                
    @staticmethod
    def cMSE(hr_masked, generated_masked):
        
        c = 1.0/tf.dtypes.cast(tf.math.reduce_sum(hr_masked[:,:,:,1:2]), tf.float32)
        bias = c * tf.math.reduce_sum(hr_masked[:,:,:,0:1] - generated_masked)
        generated_masked_ = generated_masked * hr_masked[:,:,:,1:2]
        loss = c * tf.math.reduce_sum(tf.square(hr_masked[:,:,:,0:1] - (generated_masked_ + bias)))
        return loss
    
    @staticmethod
    def cPSNR(sr, hr):

        return -10.0 * Losses.log10(Losses.cMSE(sr, hr)+1e-9)

    @staticmethod
    def cPSNR_MSE(mse):
       
        return -10.0 * Losses.log10(mse)
    
    @staticmethod
    def cPSNR_metric(sr, hr):
        
        hr_, sr_= Losses.apply_mask(sr, hr)
        return Losses.cPSNR_MSE(Losses.sliding_win_loss(sr_, hr_))
    
    @staticmethod
    def apply_mask(sr, hr):

        hr_ = hr#tf.where(tf.math.is_nan(hr), 0.0, hr)
        sr_ = sr#tf.where(tf.math.is_nan(hr), 0.0, sr)

        return hr_, sr_
    
    @staticmethod
    def log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
   
class NetworkHandler():
    
    def __init__(self, model, data_gen, lr, name, 
                 n_in = 9, 
                 max_batch_size = 5,
                 steps_per_epoch = 100, 
                 epochs = 1000,
                 ckpt_name = ""):
        
        self.net = model
        self.data = data_gen
        self.losses = Losses()
        self.lr = lr
        self.loss = 0
        self.n_in = n_in
        self.name = name
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.ev_loss = 0
        self.ev_metric = 0
        self.opt =[]
        self.ckpt_name = ckpt_name
        self.max_batch_size = max_batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
    def custom_loss(self, hr, sr):
        self.loss = self.losses.sliding_win_loss(hr,sr)
        return self.loss
    
    def fit(self):
        
        checkpoint_path = self.name + "/check_point/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                         verbose=1, 
                                                         save_weights_only=True)
        base_dir = self.name + "/board/"
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=base_dir + str(iter), 
                                                     histogram_freq=1, 
                                                     write_graph=True, 
                                                     write_images=False)

        datagen = ImageDataGenerator()
        if self.n_in== 10:
            LR, HR = self.data.get_augmented_train_pair()
        else:
            LR, HR = self.data.get_train_pair()
        
        if len(self.ckpt_name) > 0:
            path = self.name + "/check_point/" + self.ckpt_name  
            self.net.load_weights(path)    
        print(np.shape(LR))
        print(np.shape(HR))
        for i in range(1,self.max_batch_size):
            self.net.fit(datagen.flow( LR, HR, batch_size=i ),
                         steps_per_epoch = self.steps_per_epoch,
                         epochs=self.epochs, 
                         use_multiprocessing=False, 
                         callbacks=[cp_callback, tensorboard])
        
    def optimizer(self):
        
        self.opt = Adam(learning_rate=PiecewiseConstantDecay(boundaries=[40], 
                                                           values=[self.lr*10, self.lr]))
        
    def build(self):
        self.optimizer()
        self.net.compile(optimizer = self.opt,  loss =self.custom_loss, metrics=[self.losses.cPSNR_metric])
        self.net.summary()
        
    def sr_one(self, im_set, HR):
        return self.net.predict( im_set )
        
    def gen_raport(self):
        datagen = ImageDataGenerator()
        if(len(self.data.X_test) == 0):
            if self.n_in== 10:
                LR, HR = self.data.get_augmented_train_pair()
            else:
                LR, HR = self.data.get_train_pair()
        
        self.ev_loss , self.ev_metric = self.net.evaluate(datagen.flow(self.data.X_test, 
                                                                       self.data.Y_test, 
                                                                       batch_size=1 ))
        decoded_imgs = self.net.predict(self.data.X_test[:6])
        file = open(self.name+'/rez.txt', 'w')
        rez = "loss = " + str(self.ev_loss) + " metric = " + str(self.ev_metric)
        file.write(rez)
        file.close()
        self._gen_example_image(decoded_imgs)
        
        
    def _gen_example_image(self, decoded_imgs):
        
        import matplotlib.pyplot as plt
        n = 6
        for i in range(1,n):
            plt.imshow(self.data.Y_test[i,:,:,0].reshape(384, 384) * 255)
            plt.gray()
            plt.savefig(self.name+"/img_gt_"+str(i)+".png")
            plt.imshow(decoded_imgs[i].reshape(384, 384) *255)
            plt.gray()
            plt.savefig(self.name+"/img_pred_"+str(i)+".png")

    def load_cp(self, cp_name):
        
        if -1 == cp_name.find("/"):
            path = self.name + "/check_point/" + cp_name
        else:
            path = cp_name
        
        self.net.load_weights(path)
        
    def load_and_evaluate_net(self, check_point):
        self.load_cp(check_point)
        self.gen_raport()
    