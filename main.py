
import numpy as np 


import os 
import time 
import model.modelwrapper as Model
import random 
import copy 
import json 



from GraphicCodeColection.mesh_sampling import generate_transform_matrices, generate_neighborhood, available_psbody

from GraphicCodeColection.loader import Loader 


argv = os.sys.argv


if argv[1] in ["test", "train", 'summary']:
    mode = argv[1]
else : 
    mode = "train"



############################################################################

# SETTINGS and ARGUMENTS.

############################################################################

#common sets
name="test_model_2020_3sd-reiqwen"
name="mode_test_ASDASDf2asd222iles"


#loader sets
noise_type = "plain"
data_path = "./processed_dataset/plain/bareteeth"
test_size = 10
loader = Loader(common_load_dir_name=data_path, 
                noise_type= noise_type,
                test_size=test_size)
ref = loader.get_reference()
# neighbor, _, _ = generate_neighborhood(ref.v, ref.f)



print(mode, "what is mode ")

#Encoder Type : 
model_params = dict()
model_params['name'] = name
model_params['random_seed'] = 2
model_params['batch_size'] = 4
model_params['kernel_size'] = 8
model_params['use_latent'] = True 
model_params['latent_size'] = 128#64#128
model_params['num_epoch'] = 200
# model_params['F'] = [16, 32]#, 64, 96, 128] ### input_shape      [ hidden_layer_output_shape1 ... ]
model_params['F'] = [16, 32, 64, 96, 128] ### input_shape      [ hidden_layer_output_shape1 ... ]

model_params['F_0'] = loader.get_train_shape()[-1]

# model_params['activation'] = "leakyrelu"
# model_params['activation'] = "tanh"
model_params['activation'] = "relu"
model_params['face'] = ref.f
# model_params['kernel_initializer'] = "xavier_normal_initializer"
# model_params['kernel_initializer'] = "xavier_uniform_initializer"
model_params['kernel_initializer'] = "truncated_normal_initializer"
# model_params['kernel_initializer'] = "random_normal_initializer"
# model_params['kernel_initializer'] = "random_uniform_initializer"

# A and adj is essentially same. but The expression is different.

if available_psbody : 
    _,A,D,U,neighbor = generate_transform_matrices(ref.v,ref.f,[4]*len(model_params['F']))


    # A, D, U is Same Length.
    A = list(map(lambda x:x.astype('float32'), A))
    D = list(map(lambda x:x.astype('float32'), D))
    U = list(map(lambda x:x.astype('float32'), U))
    print("A", len(A))
    print("U", len(U))
    print("D", len(D))
    A=A[0]
    A.data = A.data/A.data
    A = [A]

    model_params['A'] = A # basically (5023, 5023) values is all 1,0 or 2.
    model_params['ds_D'] = D
    model_params['ds_U'] = U

else: 
    neighbor = generate_neighborhood(ref.v, ref.f)
model_params['adj'] = neighbor # neighbor is basically (5023, 32)



model_params['checkpoint_save_path'] =os.path.join('./checkpoints/', model_params['name'])
model_params['tensorboard_path'] = os.path.join('./summaries/',model_params['name'])




############################################################################

# CONFIGURATION

############################################################################



model = Model.ModelWrapper(**model_params)




############################################################################

# TRAIN & PREDICT & SUMMARY

############################################################################

if mode == "train":

    datadict = loader.get_train_data()
    inputs = datadict['input']
    labels = datadict['labels']    

    inputs = loader.get_data_normalize2(inputs)
    labels = loader.get_data_normalize2(labels)
    # print("inputs", inputs)
    # print("inputs", np.max(inputs), np.min(inputs))
    model.train(inputs, labels)


elif mode == "test":
    # print("test mode")
    # datadict = loader.get_test_data()
    # inputs = datadict['input'][:test_size]
    # labels = datadict['labels'][:test_size]
    datadict = loader.get_train_data()
    inputs = datadict['input'][:test_size]
    labels = datadict['labels'][:test_size] 
    print(np.max(inputs))
    # inputs = loader.get_data_normalize2(inputs)
    # labels = loader.get_data_normalize2(labels)

    print(np.max(inputs))
    # labels = loader.get_data_normalize(labels)
    pred, loss = model.predict(inputs=inputs,labels=labels,  batch_size=2)
    print("pred losses loss : ", loss)
    
    loader.save_ply(pred, name="pred", path="./conv_ply/"+name)
    loader.save_ply(inputs, name = "test",path="./conv_ply/"+name)    
    loader.save_ply(labels, name = "orig",path="./conv_ply/"+name)    
    
    # loader.save_ply(loader.get_data_denormalize2(pred), name="pred", path="./conv_ply/"+name)
    # loader.save_ply(loader.get_data_denormalize2(inputs), name = "test",path="./conv_ply/"+name)    
    # loader.save_ply(loader.get_data_denormalize2(labels), name = "orig",path="./conv_ply/"+name)    

    model.summary()
elif mode == "summary":
    datadict = loader.get_train_data()
    inputs = datadict['input'][:1]
    labels = datadict['labels'][:1] 
    inputs = loader.get_data_normalize(inputs)
    # labels = loader.get_data_normalize(labels)
    pred, loss = model.predict(inputs=inputs,labels=labels,  batch_size=1)
    model.see_all_values()