import tensorflow as tf 
from functools import singledispatch


@singledispatch
def set_activation(self, opt):
    raise Exception("that instance was not supported. inherit tf.keras.Model or Layer")
    
@set_activation.register
def _(self : tf.keras.layers.Layer, opt : str):
    if type(opt) == str:
        opt = opt.lower()
        if  opt == "relu": 
            return tf.keras.activations.relu
        elif opt == "leakyrelu": 
            # return tf.keras.activations.leakyrelu
            return tf.nn.leaky_relu
        elif opt == None:
            return None

    else : 
        return opt

@set_activation.register
def _(self : tf.keras.Model, opt : str):
    if type(opt) == str:
        opt = opt.lower()
        if  opt == "relu": 
            return tf.keras.layers.ReLU
        elif opt == "leakyrelu": 
            return tf.keras.layers.LeakyReLU

    else : 
        return opt

