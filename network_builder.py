from network_handler import NetworkHandler
from utils.dataloader import DataHandler

class NetworkBuilder:   
    def __init__(self, name, 
                 depth = 8, 
                 num_input_im = 9, 
                 num_res_blocks = 10,
                 learning_rate = 1e-5,
                 max_batch_size = 5,
                 steps_per_epoch = 100,
                 epochs = 100):
        
        self.depth = depth
        self.num_input_im = num_input_im
        self.name = name
        self.num_res_blocks = num_res_blocks
        self.learning_rate =learning_rate
        self.max_batch_size = max_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        return
    
    def build_net(self):
        
        if self.name == "Net1": 
            from nets.Net1 import Net1
            m =  Net1(depth = self.depth, 
                      input_im =self.num_input_im, 
                      num_rez_block = self.num_res_blocks).build_net( )
        
        if self.name == "Net2": 
            from nets.Net2 import Net2
            m =  Net2().build_net( )
        
        if self.name == "Net3": 
            from nets.Net3 import Net3
            m =  Net3(depth = self.depth, 
                      input_im =self.num_input_im).build_net( )
        
        if self.name == "Net4":
            from nets.Net4 import Net4
            m =  Net4(depth = self.depth, 
                      input_im =self.num_input_im).build_net( )
        
        if self.name == "Net5":
            from nets.Net5 import Net5
            m =  Net5(depth = self.depth, 
                      input_im =self.num_input_im).build_net( )
        
        if self.name == "Net6":
            from nets.Net6 import Net6
            m =  Net6(depth = self.depth, 
                      input_im =self.num_input_im).build_net( )
        
        if self.name == "Net7":
            from nets.Net7 import Net7
            m =  Net7(depth = self.depth, 
                      input_im =self.num_input_im).build_net( )
        
        if self.name == "Net8":
            from nets.Net8 import Net8
            m =  Net8(depth = self.depth, 
                      input_im =self.num_input_im).build_net( )
        
        if self.name == "Net9":# set num_input_im to 10
            from nets.Net9 import Net9
            m =  Net9(depth = self.depth, 
                      input_im =self.num_input_im).build_net()
        
        if self.name == "Net10":
            from nets.Net10 import Net10
            m =  Net10(depth = self.depth, 
                      input_im =self.num_input_im).build_net()
            
        if self.name == "Net10v2":
            from nets.Net10 import Net10v2
            m =  Net10v2(depth = self.depth, 
                      input_im =self.num_input_im).build_net()
        
        if self.name == "Net11":
            from nets.Net11 import Net11
            m =  Net11(depth = self.depth, 
                      input_im =self.num_input_im).build_net()
        
        if self.name == "Net11v2":
            from nets.Net11 import Net11v2
            m =  Net11v2(depth = self.depth, 
                      input_im =self.num_input_im, 
                      num_rez_block= self.num_res_blocks).build_net()
        
        if self.name == "Net11v3":
            from nets.Net11 import Net11v3
            m =  Net11v3(depth = self.depth, 
                         input_im =self.num_input_im,
                         num_rez_block = self.num_res_blocks).build_net()
        if 'm' not in locals(): 
             raise Exception('Network name invalid!')
                             
        data_gen = DataHandler()
      
        nh = NetworkHandler(m, 
                            data_gen, 
                            self.learning_rate, 
                            self.name, 
                            self.num_input_im,
                            self.max_batch_size,
                            self.steps_per_epoch,
                            self.epochs)
        
        return nh