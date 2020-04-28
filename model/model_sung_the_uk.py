import tensorflow as tf
import numpy as np 
from .logger import * #set_device, time_set
from .utils import set_activation, set_initializer

#for pooling
import math 
import heapq
import scipy.sparse as sp
import tensorflow_graphics.nn.layer.graph_convolution as tfg

"""
experimental model
"""

device = "GPU:1"

        #to sparse tensor
        
@tf.function
def loss(y_label, y_pred, args=None):
        total_loss = tf.keras.backend.mean(tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y_label-y_pred, 2.0), axis=-1)), axis=-1))
        return total_loss



@tf.function
def feature_normalize( inputs, alpha=0.9, epsilon=10e-8):
    #max_val, min_val := [batch size, 1, Fin]
    max_val = tf.keras.backend.max(inputs,keepdims=True)
    min_val = tf.keras.backend.min(inputs,keepdims=True)
    
    # if max_val == min_val, max and mean is inputs, +epssilon

    idx = tf.math.equal(max_val, min_val)
    max_val = tf.where(idx, max_val+epsilon, max_val)
    min_val = tf.where(idx, min_val-epsilon, min_val)
    inputs = 2 * alpha * (inputs-min_val)/(max_val-min_val) - alpha
    return inputs
    
class Model(tf.keras.Model):
    def __init__(self, 
                encoder_shape_list, 
                decoder_shape_list,
                adj,
                kernel_size,
                activation,
                use_latent,
                latent_size,
                face,
                ds_D,
                ds_U,
                A, 
                kernel_initializer,
                name="Model",
                trainable=True):
        super(Model, self).__init__(trainable=trainable, name = name)
        self.encoder_shape_list = encoder_shape_list
        self.decoder_shape_list = decoder_shape_list
        self.adj = adj
        self.kerenl_size = kernel_size
        self.activation_name= activation
        self.use_latent = use_latent
        self.latent_size = latent_size
        self.face = face
        self.ds_D =ds_D
        self.ds_U = ds_U
        self.A = A
        self.kernel_initializer = kernel_initializer       

    def build(self, input_shape): 
        batch_size, vertices_size, _ = input_shape
        self.encoder = Encoder(layer_info = self.encoder_shape_list, 
                                adj = self.adj, 
                                kernel_size= self.kerenl_size, 
                                activation= self.activation_name,
                                use_latent= self.use_latent,
                                latent_size = self.latent_size,
                                face=self.face,
                                ds_D = self.ds_D,
                                A =self.A, 
                                kernel_initializer=self.kernel_initializer,
                                name="encoder",
                                trainable=True
                                        )
        
        self.decoder = Decoder(layer_info = self.decoder_shape_list, 
                                adj = self.adj,
                                kernel_size= self.kerenl_size,
                                batch_size = batch_size,
                                vertex_size = self.A[-1].shape[0],
                                use_latent= self.use_latent,
                                activation = self.activation_name,
                                ds_U = self.ds_U,
                                A = self.A,
                                kernel_initializer=self.kernel_initializer,
                                name = "decoder", 
                                trainable=True)


    @tf.function
    def call(self, inputs): 
        outputs = self.encoder(inputs)
        if type(outputs) == tuple : 
            latent_z, z_mean, z_log_var = outputs
            outputs = self.decoder(latent_z)
            return outputs, latent_z, z_mean, z_log_var
        return self.decoder(outputs)


class Encoder(tf.keras.Model):
    def __init__(self, 
                layer_info, 
                adj,
                kernel_size=9,
                activation = 'relu',
                use_latent = True,
                latent_size = 8,
                face=None,
                ds_D=None,
                A = None,
                kernel_initializer = None,
                name = "encoder", 
                trainable=True):
        super(Encoder, self).__init__(trainable=trainable, name = name)
        self.conv1 = Conv1DBN(64, 20, 1, trainable)
        self.conv2 = Conv1DBN(64, 20, 1, trainable)
        self.conv3 = Conv1DBN(64, 20, 1, trainable)
        self.conv4 = Conv1DBN(64, 20, 1, trainable)
        self.conv5 = Conv1DBN(1, 20, 1, trainable)
        self.flatten = tf.keras.layers.Flatten()
        self.fn = tf.keras.layers.Dense(256)
        

    def build(self, input_shape):
        pass
                                                        
        
    @tf.function
    @set_device("GPU:0")        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fn(x)
        return x



class Decoder(tf.keras.Model):
    def __init__(self, 
                layer_info, 
                adj,
                kernel_size=9,
                batch_size = 16,
                vertex_size = 5023,
                use_latent = False,
                activation = tf.keras.activations.relu ,
                ds_U = None,
                A = None,
                kernel_initializer = None,
                name = "decoder", 
                trainable=True):
        super(Decoder, self).__init__(trainable=trainable, name = name)
        self.fn = tf.keras.layers.Dense(5023*1)
        self.conv1 = Conv1DBN(64, 20, 1, trainable)
        self.conv2 = Conv1DBN(64, 20, 1, trainable)
        self.conv3 = Conv1DBN(64, 20, 1, trainable)
        self.conv4 = Conv1DBN(64, 20, 1, trainable)
        self.conv5 = Conv1DBN(64, 20, 1, trainable)

        self.conv6 = Conv1DBN(64, 20, 1, trainable)

        self.conv7 = tf.keras.layers.Conv1D(3, 20, 1, padding='same', use_bias=False)
        

    def build(self, input_shape):
        pass

    def call(self, x):
        x = self.fn(x)
        x = tf.reshape(x, [-1, 5023, 1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)        
        x = self.conv7(x)

        return x


class Conv1DBN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, trainable):
        super(Conv1DBN, self).__init__(trainable=trainable)
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size, strides, padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = tf.nn.relu(x)

        return x
