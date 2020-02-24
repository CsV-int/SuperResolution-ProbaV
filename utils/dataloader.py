import numpy as np
import matplotlib.pyplot as plt
from skimage import io 
import glob
import pickle
from os import path
from sklearn.model_selection import train_test_split


class DataHandler(object):
    '''
        Parameters
        ----------
        train_dir - string , optional
            Path to data. 
        num_per_set - int , optional
            number of images considered as "goog". The default is 9.

        Returns
        -------
        None.

        '''
    def __init__(self,train_dir="../probav_data/train", num_per_set = 9):
        
        self.train_dir = train_dir
        self.num_per_set = num_per_set
        
        self.LR_images_train = []
        self.QM_Images_train = [] 
        self.HR_Images_train = []
        self.SM_Images_train = []
        
        self.LR_images_test = []
        self.QM_Images_test = []
        self.SM_Images_test = []
        
        self.LR_ret = []
        self.QM_ret = []

        self.LR_mean = []
        
        self.train = []
        self.test = []
        self.testSM = []
        
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        
        self.SM_train = []
        self.SM_test = []
        
    def _load_train_images(self):
        
        dir_list=glob.glob( self.train_dir+'/*/'+'*')
        dir_list.sort()
        self.LR_images_train = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/LR*.png'))] for dir_name in dir_list ])
        self.QM_Images_train = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/QM*.png'))] for dir_name in dir_list ])
        self.HR_Images_train = np.array([io.imread(glob.glob(dir_name+'/HR.png')[0]) for dir_name in dir_list ])
        self.SM_Images_train = np.array([io.imread(glob.glob(dir_name+'/SM.png')[0]) for dir_name in dir_list ])
        
    def _load_test_images(self):
        
        dir_list=glob.glob(self.train_dir+'/*/'+'*')
        dir_list.sort()
        self.LR_images_test = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/LR*.png'))] for dir_name in dir_list ])
        self.QM_Images_test = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/QM*.png'))] for dir_name in dir_list ])
        self.SM_Images_test = np.array([io.imread(glob.glob(dir_name+'/SM.png')[0]) for dir_name in dir_list ])

    def _select_best_images(self):
        
        QM_ret = []
        LR_ret = []
        
        for i in range(len(self.QM_Images_train)):
            nz = []
            QM_sorted = []
            LR_sorted = []
        
            for j in range(len(self.QM_Images_train[i])):
                nz.append(np.count_nonzero(self.QM_Images_train[i][j]==0))
            
            QM_sorted = [x for o,x in sorted(zip(nz,self.QM_Images_train[i]),key=lambda x: x[0])]
            QM_ret.append(QM_sorted[:self.num_per_set])
            LR_sorted = [x for o,x in sorted(zip(nz,self.LR_images_train[i]),key=lambda x: x[0])]
            LR_ret.append(LR_sorted[:self.num_per_set]) 
            
        self.LR_ret, self.QM_ret = LR_ret, QM_ret

    def _generate_mean_image(self):
    
        LR_mean = []
        for k in range(len(self.LR_images_train)):
            
            th = np.mean(self.LR_images_train[k])
            norm = np.max(self.LR_images_train[k])
            added = 0
            mean = self.LR_images_train[0][0].astype(float)
            mean[:,:] = 0.
            
            for i in range(len(self.LR_images_train[k])):
                for j in range(i+1,len(self.LR_images_train[k])):
            
                    if np.mean(self.LR_images_train[k][i] - self.LR_images_train[k][j]) < th*4:
                        mean += self.LR_images_train[k][i]/norm + self.LR_images_train[k][j]/norm
                        added += 2
            
            dt = (mean/added)*norm
            LR_mean.append(dt.astype(int))
                        
        self.LR_mean = LR_mean
        
    def _save_test_images(self):
       
        pickle_out = open("LR_images_t.pickle","wb")
        pickle.dump(self.LR_images_test, pickle_out)
        pickle_out.close()
        pickle_out = open("QM_Images_t.pickle","wb")
        pickle.dump(self.QM_Images_test, pickle_out)
        pickle_out.close()
        pickle_out = open("SM_Image_t.pickle","wb")
        pickle.dump(self.SM_Image_test, pickle_out)
        pickle_out.close()
        
    def _save_train_images(self):
       
        pickle_out = open("LR_images.pickle","wb")
        pickle.dump(self.LR_images_train, pickle_out)
        pickle_out.close()
        pickle_out = open("QM_Images.pickle","wb")
        pickle.dump(self.QM_Images_train, pickle_out)
        pickle_out.close()
        pickle_out = open("HR_Image.pickle","wb")
        pickle.dump(self.HR_Images_train, pickle_out)
        pickle_out.close()
        pickle_out = open("SM_Image.pickle","wb")
        pickle.dump(self.SM_Images_train, pickle_out)
        pickle_out.close()
    
    def _save_best_images(self):
       
        pickle_out = open("LR_ret.pickle","wb")
        pickle.dump(self.LR_ret, pickle_out)
        pickle_out.close()
        pickle_out = open("QM_ret.pickle","wb")
        pickle.dump(self.QM_ret, pickle_out)
        pickle_out.close()
            
    def _save_LR_mean(self):
        
        pickle_out = open("LR_mean.pickle","wb")
        pickle.dump(self.LR_mean, pickle_out)
        pickle_out.close()
        
    def load_test_images(self):
        
        if path.exists("LR_images_t.pickle") \
            and path.exists("QM_Images_t.pickle") \
            and path.exists("SM_Image_t.pickle"):
                pickle_in = open("LR_images_t.pickle","rb")
                self.LR_images_test = pickle.load(pickle_in)
                pickle_in = open("QM_Images_t.pickle","rb")
                self.QM_Images_test = pickle.load(pickle_in)
                pickle_in = open("SM_Image_t.pickle","rb")
                self.SM_Image_test = pickle.load(pickle_in)
                print("Loaded test files from pickle")
        else:
            self._load_test_images()
            self._save_test_images( )
            print("Loaded test files from images")
        
    def load_train_images(self):
        
        if path.exists("LR_images.pickle") \
            and path.exists("QM_Images.pickle") \
            and path.exists("HR_Image.pickle") \
            and path.exists("SM_Image.pickle"):
                pickle_in = open("LR_images.pickle","rb")
                self.LR_images_train = pickle.load(pickle_in)
                pickle_in = open("QM_Images.pickle","rb")
                self.QM_Images_train = pickle.load(pickle_in)
                pickle_in = open("HR_Image.pickle","rb")
                self.HR_Images_train = pickle.load(pickle_in)
                pickle_in = open("SM_Image.pickle","rb")
                self.SM_Images_train = pickle.load(pickle_in)
                print("Loaded train files from pickle")
        else:
            self._load_train_images()
            self._save_train_images()
            print("Loaded train files from file")

    def load_best_images(self):
        
        if path.exists("LR_ret.pickle") \
            and path.exists("QM_ret.pickle"):
                pickle_in = open("LR_ret.pickle","rb")
                self.LR_ret = pickle.load(pickle_in)
                pickle_in = open("QM_ret.pickle","rb")
                self.QM_ret = pickle.load(pickle_in)
                print("Loaded best files from pickle")
    
        else:
            self.load_train_images()
            self._select_best_images()
            self._save_best_images( )
            print("Loaded best files from file")
    
    def load_mean_images(self):
        
        if path.exists("LR_mean.pickle"):
                pickle_in = open("LR_mean.pickle","rb")
                self.LR_mean = pickle.load(pickle_in)
                print("Loaded mean files from pickle")
    
        else:
            self.load_best_images()
            self._generate_mean_image()
            self._save_LR_mean()
            print("Loaded mean files from file")
    
    def get_train_pair(self):
        '''
        Returns X_train(LR), Y_train(HR) to be fed directly in fit
        
            The data is in the [0,1] space
            
            augumented with the mean of the input images

        '''
        if len(self.train) != 0:
                
            _,_,_,s = np.shape(self.train)
            if s != self.num_per_set:
                self.train = []
                self.test = []
                self.testSM = []
            
                if len(self.LR_ret) == 0:
                    self.load_train_images()
                    self.load_best_images()
                mean_train = np.mean(self.LR_ret)
                mean_test = np.mean(self.HR_Images_train)
                std_train = np.std(self.LR_ret)
                std_test = np.std(self.HR_Images_train)  
   
                self.LR_ret = np.asarray(self.LR_ret) / 16384.
                self.HR_Images_train = np.asarray(self.HR_Images_train) / 16384.
            
                
                for i in range(len(self.LR_ret)):
                    for j in range(len(self.QM_ret[i])):
                        self.LR_ret[i][j][self.QM_ret[i][j] == 0] = 0
                    
                    self.train.append(np.dstack( self.LR_ret[i][:]))
                    self.test.append(np.dstack([self.HR_Images_train[i],self.SM_Images_train[i]/255]))
                    self.testSM.append(self.SM_Images_train[i])
                        
        else:
            if len(self.LR_ret) == 0:
                self.load_train_images()
                self.load_best_images()
            mean_train = np.mean(self.LR_ret)
            mean_test = np.mean(self.HR_Images_train)
            std_train = np.std(self.LR_ret)
            std_test = np.std(self.HR_Images_train)  

            self.LR_ret = np.asarray(self.LR_ret) / 16384.
            self.HR_Images_train = np.asarray(self.HR_Images_train) / 16384.
            
            for i in range(len(self.LR_ret)):
                for j in range(len(self.QM_ret[i])):
                    self.LR_ret[i][j][self.QM_ret[i][j] == 0] = 0
                
                self.train.append(np.dstack( self.LR_ret[i][:]))
                self.test.append(np.dstack([self.HR_Images_train[i],self.SM_Images_train[i]/255]))
                self.testSM.append(self.SM_Images_train[i])

        self.train =  np.asarray(self.train).astype('float64')
        self.test =  np.asarray(self.test).astype('float64')
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.train, self.test,
                                                                                 test_size=0.3, 
                                                                                 random_state=42)
         
        _, _, self.SM_train, self.SM_test = train_test_split(self.train, self.testSM, 
                                                                         test_size=0.3, 
                                                                         random_state=42)
        self.LR_images_train = []
        self.QM_Images_train = [] 
        self.HR_Images_train = []
        self.SM_Images_train = []
        
        self.LR_images_test = []
        self.QM_Images_test = []
        self.SM_Images_test = []
        
        self.LR_ret = []
        self.QM_ret = []

        self.LR_mean = []
        
        self.train = []
        self.test = []
        self.testSM = []
        
        return self.X_train, self.Y_train
    
    def get_augmented_train_pair(self):
        '''
        Returns X_train(LR), Y_train(HR) to be fed directly in fit
        
            The data is in the [0,1] space
            
            augumented with the mean of the input images

        '''
        import copy

        if len(self.train) != 0:
           
            _,_,_,s = np.shape(self.train)
            if s != self.num_per_set + 1:
                self.train = []
                self.test = []
                self.testSM = []
                
                if len(self.LR_ret) == 0:
                    self.load_train_images()
                    self.load_best_images()
                if len(self.LR_mean) == 0:
                    self.load_mean_images()
                    
                mean_train = np.mean(self.LR_ret)
                mean_test = np.mean(self.HR_Images_train)
                std_train = np.std(self.LR_ret)
                std_test = np.std(self.HR_Images_train)

                    
                LR_ret = copy.deepcopy(self.LR_ret)
        
                for i in range(len(self.LR_ret)):
                    LR_ret[i].append(self.LR_mean[i])
                    for j in range(len(self.QM_ret[i])):
                        LR_ret[i][j][self.QM_ret[i][j] == 0] = 0
                    self.train.append(np.dstack(LR_ret[i][:]))

                    self.test.append(np.dstack([self.HR_Images_train[i],self.SM_Images_train[i]/255]))
                    self.testSM.append(self.SM_Images_train[i])
                    
        else:
            if len(self.LR_ret) == 0:
                self.load_train_images()
                self.load_best_images()
            if len(self.LR_mean) == 0:
                self.load_mean_images()
                
            mean_train = np.mean(self.LR_ret)
            mean_test = np.mean(self.HR_Images_train)
            std_train = np.std(self.LR_ret)
            std_test = np.std(self.HR_Images_train)

            self.HR_Images_train = np.asarray(self.HR_Images_train) / 16384.
                
            LR_ret = copy.deepcopy(self.LR_ret)
    
            for i in range(len(self.LR_ret)):
                LR_ret[i].append(self.LR_mean[i])
                LR_ret[i] = np.asarray(LR_ret[i]) / 16384.
                for j in range(len(self.QM_ret[i])):
                    LR_ret[i][j][self.QM_ret[i][j] == 0] = 0
                self.train.append(np.dstack(LR_ret[i][:]))
                self.test.append(np.dstack([self.HR_Images_train[i],self.SM_Images_train[i]/255]))
                self.testSM.append(self.SM_Images_train[i])
                
        self.train =  np.asarray(self.train).astype('float64')
        self.test =  np.asarray(self.test).astype('float64')
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.train, self.test,
                                                                                 test_size=0.3, 
                                                                                 random_state=42)
         
        _, _, self.SM_train, self.SM_test = train_test_split(self.train, self.testSM, 
                                                                         test_size=0.3, 
                                                                         random_state=42)
        
        self.LR_images_train = []
        self.QM_Images_train = [] 
        self.HR_Images_train = []
        self.SM_Images_train = []
        
        self.LR_images_test = []
        self.QM_Images_test = []
        self.SM_Images_test = []
        
        self.LR_ret = []
        self.QM_ret = []

        self.LR_mean = []
        
        self.train = []
        self.test = []
        self.testSM = []
        
        return self.X_train, self.Y_train
    

    def get_train_subsamplea_pair(self,size = 64, stride = 16):

       _,_ = self.get_train_pair()
       a = self.X_train
       c = self.X_test 
       b = self.Y_train
       d = self.Y_test
       aa = []
       ab = []
       ac = []
       ad = []
       for i in range(0,128,stride):
           if i + size > 128:
               break
           m = a[:,i:i+size,i:i+size, :]
           mm = c[:,i:i+size,i:i+size, :]

           aa.append(m)
           ac.append(mm)
           
       for i in range(0,128 *3,stride *3):
           if i + size*3 > 128 * 3:
               break
           n = b[:,i:i+size*3,i:i+size*3]
           nn = d[:,i:i+size*3,i:i+size*3]

           ab.append(n)
           ad.append(nn)
           
       d1,d2,d3,d4,d5 = np.shape(aa)
       dd1,dd2,dd3,dd4,dd5 = np.shape(ac)
       
       aa = np.reshape(aa, (d1 * d2, size, size, 9) )
       ab = np.reshape(ab, (d1 * d2, size*3, size*3) )
       
       ac = np.reshape(ac, (dd1 * dd2, size, size, 9) )
       ad = np.reshape(ad, (dd1 * dd2, size*3, size*3) )
       self.X_train = aa
       self.X_test = ac
       self.Y_train = ab
       self.Y_test = ad
       
       return aa, ab
        