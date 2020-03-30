
import numpy as np 


import os 
import time 
import model.modelwrapper as Model
import random 
import copy 
import json 



from GraphicCodeColection.mesh_sampling import generate_neighborhood

from GraphicCodeColection.loader import Loader 


argv = os.sys.argv


if argv[1] in ["test", "train", 'summary']:
    mode = argv[1]
else : 
    mode = "train"



############################################################################

# SETTINGS

############################################################################

#common sets
name="test_model_2020_3d-reiqwen"
name="submarine3"


#loader sets
noise_type = "plain"
data_path = "./processed_dataset/plain/bareteeth"
test_size = 10
loader = Loader(common_load_dir_name=data_path, 
                noise_type= noise_type,
                test_size=test_size)
ref = loader.get_reference()
neighbor, _, _ = generate_neighborhood(ref.v, ref.f)



print(mode, "what is mode ")

#Encoder Type : 
model_params = dict()
model_params['name'] = name
model_params['random_seed'] = 2
model_params['batch_size'] = 1
model_params['kernel_size'] = 3
model_params['use_latent'] = False 
model_params['latent_size'] = 32#64#128
model_params['num_epoch'] = 4
model_params['F'] = [16, 32]#, 64]#, 128] #####        input_shape      [ hidden_layer_output_shape1 ... ]
model_params['F'] = [16]#, 16,16]#, 64]#, 128] #####        input_shape      [ hidden_layer_output_shape1 ... ]

model_params['F_0'] = loader.get_train_shape()[-1]

model_params['adj'] = neighbor # 5023, 32
model_params['activation'] = "leakyrelu"

model_params['checkpoint_save_path'] =os.path.join('./checkpoints/', model_params['name'])
model_params['tensorboard_path'] = os.path.join('./summaries/',model_params['name'])


















############################################################################

# CONFIGURATION

############################################################################



model = Model.ModelWrapper(**model_params)





if mode == "train":

    datadict = loader.get_train_data()
    inputs = datadict['input']
    labels = datadict['labels']    
    # inputs = loader.get_data_normalize(inputs)
    # labels = loader.get_data_normalize(labels)
    model.train(inputs, labels)


elif mode == "test":
    # print("test mode")
    # datadict = loader.get_test_data()
    # inputs = datadict['input'][:test_size]
    # labels = datadict['labels'][:test_size]
    datadict = loader.get_train_data()
    inputs = datadict['input'][:test_size]
    labels = datadict['labels'][:test_size] 
    inputs = loader.get_data_normalize(inputs)
    labels = loader.get_data_normalize(labels)
    pred, loss = model.predict(inputs=inputs,labels=labels,  batch_size=1)
    print("loss : ", loss)

    loader.save_ply(pred, name="pred", path="./conv_ply/"+name)
    loader.save_ply(inputs, name = "test",path="./conv_ply/"+name)    
    loader.save_ply(labels, name = "orig",path="./conv_ply/"+name)    
    

    model.summary()
elif mode == "summary":
    datadict = loader.get_train_data()
    inputs = datadict['input'][:1]
    labels = datadict['labels'][:1] 
    pred, loss = model.predict(inputs=inputs,labels=labels,  batch_size=1)
    model.see_all_values()