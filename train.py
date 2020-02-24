from network_builder import NetworkBuilder


nets_available = ["Net1","Net2","Net4","Net5","Net7","Net8","Net9","Net10","Net10v2","Net11","Net11v2","Net11v3"]

images_in = 10
for name in nets_available:
    if name == "Net9":
        
        bn = NetworkBuilder(name, depth=32,num_input_im = images_in, max_batch_size = 6, epochs = 500, steps_per_epoch = 50).build_net()
    else:
        bn = NetworkBuilder(name, depth=32,max_batch_size = 6, epochs = 500, steps_per_epoch = 5).build_net()
    bn.build()
    bn.fit()
    bn.gen_raport()
    del bn# = None
