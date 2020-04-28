import tensorflow as tf 
from functools import singledispatch

import matplotlib.pyplot as plt




@singledispatch
def set_activation(self, opt):
    raise Exception("that instance was not supported. inherit tf.keras.Model or Layer")
    
@set_activation.register
def _(self : tf.keras.layers.Layer, opt : str):
    if type(opt) == str:
        opt = opt.lower()
        if  opt == "relu": 
            return tf.keras.layers.ReLU()
        elif opt == "leakyrelu": 
            # return tf.keras.activations.leakyrelu
            return tf.keras.layers.LeakyReLU()
        elif opt == "tanh":
            print("tanh")
            return tf.keras.layers.Activation("tanh")
    
        elif opt == None:
            return None

    else : 
        return opt

@set_activation.register
def _(self : tf.keras.Model, opt : str):
    if type(opt) == str:
        opt = opt.lower()
        if  opt == "relu": 
            return tf.keras.layers.ReLU()
        elif opt == "leakyrelu": 
            return tf.keras.layers.LeakyReLU()
        elif opt == "tanh":
            return tf.keras.layers.Activation("tanh")
    
    else : 
        return opt

def set_initializer(name):
    if name == "xavier_normal_initializer":
        return tf.keras.initializers.GlorotNormal()
    elif name == "xavier_uniform_initializer" : 
        return tf.keras.initializers.GlorotUniform()
    elif name == "truncated_normal_initializer":
        return tf.keras.initializers.TruncatedNormal()
    elif name == "random_normal_initializer":
        return tf.keras.initializers.RandomNormal()
    elif name == "random_uniform_initializer":
        return tf.keras.initializers.RandomUniform()
    else : 
        return tf.keras.initializers.TruncatedNormal()


def builder():
    pass

def draw_weight(model):
    import numpy as np 
    encoder = model.encoder
    decoder = model.decoder
    evar = encoder.exec_list
    dvar = decoder.exec_list
    cm = 'RdYlBu'
    
    for layer in evar+dvar:
        a=            len(layer.trainable_variables)
        fig, axs = plt.subplots()
        for var in layer.trainable_variables : 
            
            data = var.numpy()
            name = var.name
            ax = axs[i]
            ax.set_aspect('equal')
            pcm = ax.pcolormesh(data[i], cmap=cm)
            fig.colorbar(pcm, ax = ax)





    
    
    plt.show()




    fig, axs = plt.subplots(3)
    import numpy as np 
    data = dvar[0].numpy()
    test = data
    data = np.transpose(data, [1,0,2])
    print("is right", data[1] == test[:,1,:])
    cm = 'RdYlBu'
    print(evar[0].shape)
    for i in range(3):
        ax = axs[i]
        
        ax.set_aspect('equal')
        ax.set_xticks([i for i in range(16)])
        ax.set_yticks([i for i in range(3)])
        pcm = ax.pcolormesh(data[i], cmap=cm)
        
        
   
    plt.show()


