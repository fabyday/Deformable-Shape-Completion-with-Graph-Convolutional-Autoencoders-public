import tensorflow as tf
import numpy as np 
from .logger import * #set_device, time_set
from .utils import * 

#for pooling
import math 
import heapq
import scipy.sparse as sp
import tensorflow_graphics.nn.layer.graph_convolution as tfg

"""
experimental model
"""

device = "GPU:1"

@tf.function
def loss(y_label, y_pred, args=None):
    return tf.keras.backend.mean(tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y_label-y_pred, 2.0), axis=-1)), axis=-1))
# @tf.function
# def loss(y_label, y_pred, args=None):
#         def log_normal_pdf(sample, mean, logvar, raxis=1): 
#             log2pi = tf.math.log(2. * np.pi)
#             return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        
#         total_loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y_label-y_pred, 2.0), axis=-1)), axis=-1)
#         if type(args) == list and len(args) == 2:            
#             latent_z = args[0]
#             z_mean = args[1]
#             z_log_var = args[2]

#             # kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var) 
#             kl_loss= tf.keras.backend.square(z_mean) + tf.keras.backend.exp(z_log_var)- z_log_var - 1
#             kl_loss = tf.keras.backend.sum(kl_loss, axis = - 1)
#             # logpz = log_normal_pdf(latent_z, 0., 0.)
#             # logpz_x = log_normal_pdf(latent_z, z_mean, z_log_var)
#             # kl_loss = logpz - logpz_x
            
#             # total_loss += (10e-8 * kl_loss)
#             total_loss += 0.5*( kl_loss)

#         total_loss = tf.keras.backend.mean(total_loss, axis = 0)
#         return total_loss



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
                name="Model",
                trainable=True):
        super(Model, self).__init__(trainable=trainable, name = name)
        self.encoder_shape_list = encoder_shape_list
        self.decoder_shape_list = decoder_shape_list
        self.adj = adj
        self.kerenl_size = kernel_size
        self.activation = activation
        self.use_latent = use_latent
        self.latent_size = latent_size
        self.face = face
        self.ds_D =ds_D
        self.ds_U = ds_U
        self.ds_U.reverse()
        self.A = A
        

    @set_device(device)
    def build(self, input_shape): 
        batch_size, vertices_size, _ = input_shape
        self.encoder = Encoder(layer_info = self.encoder_shape_list, 
                                adj = self.adj, 
                                kernel_size= self.kerenl_size, 
                                activation= self.activation,
                                use_latent= self.use_latent,
                                latent_size = self.latent_size,
                                face=self.face,
                                ds_D = self.ds_D,
                                A = self.A,
                                name="encoder",
                                trainable=True
                                        )
        self.decoder = Decoder(layer_info = self.decoder_shape_list, 
                                adj = self.adj,
                                kernel_size= self.kerenl_size,
                                batch_size = batch_size,
                                vertex_size = vertices_size,
                                use_latent= self.use_latent,
                                activation = self.activation,
                                ds_U = self.ds_U,
                                A = self.A, 
                                name = "decoder", 
                                trainable=True)


    # @tf.function
    # @set_device(device)
    # def call(self, inputs): 
    #     outputs = self.encoder(inputs)
    #     return self.decoder(outputs)

    @tf.function
    @set_device(device)
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
                name = "encoder", 
                trainable=True):
        super(Encoder, self).__init__(trainable=trainable, name = name)
        self.layer_info = layer_info
        self.adj = adj 
        self.latent_size = latent_size
        self.use_latent = use_latent
        self.activation = activation 
        self.activation_layer = set_activation(self, self.activation)# it return activation class. use activation build it. see also utils.set_activation
        self.kernel_size = kernel_size
        self.exec_list = []
        self.latent_list = []
        self.activation_list = []
        self.face=face
        self.ds_D = ds_D
        self.A = A

    @set_device(device)
    def build(self, input_shape):

        idx = 0 
        preset = dict()
        
        preset['translation_invariant'] = True
        preset['num_weight_matrices'] = self.kernel_size
        preset['num_output_channels'] = self.layer_info[idx+1]
        preset['initializer'] = tf.keras.initializers.GlorotNormal
        preset['name'] = "encode_conv_" + str(idx)
        
        #to sparse tensor
        print("==== test "*10, self.A[0].shape)
        self.sparse_A_tensor = [] 
        for A in self.A:
            coo = A.tocoo()
            print(coo)
            
            indices = np.mat([coo.row, coo.col]).transpose()
            print(indices)
            sparse_A_tensor = tf.sparse.SparseTensor(indices, coo.data, coo.shape)
            sparse_A_tensor = tf.sparse.expand_dims(sparse_A_tensor, axis=0)

            self.sparse_A_tensor.append( tf.sparse.concat( 0, [sparse_A_tensor for _ in range(input_shape[0])] ) )
        print(self.sparse_A_tensor[0].shape)
        
        layer = tfg.FeatureSteeredConvolutionKerasLayer(**preset)
        layer.is_layer = True
        self.exec_list.append(layer)
        act = tf.keras.layers.Activation(self.activation)
        act.is_layer = False
        self.exec_list.append(act)
        

        # self.exec_list.append(Pool(self.face))
        if self.use_latent : 
            # def avg_pool(k_size):
            #     def wrapper(inputs):
            #         return tf.nn.avg_pool(inputs, k_size)
            #     return wrapper

            # self.exec_list.append(avg_pool([1, ,1]))
            self.latent_list.append(LinearLayer(
                                    output_shape=self.latent_size,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.zeros,
                                    # activation=self.activation,
                                    activation=None,
                                    name="e_Dense_",
                                    trainable=True
                                    ))


            # self.latent_list.append(tf.keras.layers.Dense(self.latent_size))
            # self.latent_list.append(LinearLayer(
            #                         output_shape=self.latent_size,
            #                         kernel_initializer=tf.keras.initializers.GlorotNormal,
            #                         # use_bias=False,
            #                         use_bias = True,
            #                         bias_initializer=tf.keras.initializers.zeros,
            #                         # activation=self.activation,
            #                         activation=None,
            #                         name="e_Dense_",
            #                         trainable=True
            #                         ))



                                                        
        
           

    @tf.function
    @set_device(device)        
    def call(self, inputs):
        x = inputs
        i = 0
        for  layer in self.exec_list:
            if layer.is_layer == True : 
                # print(x, "x = is ")
                # print(self.sparse_A_tensor[0], "sparse A tensor")
                x = layer([x, self.sparse_A_tensor[0]])
                i += 1
            else : 
                x = layer(x)
        
        if self.use_latent : 
            x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1]*tf.shape(x)[-1]])
            x = self.latent_list[0](x)
            # tf.print(x[0])
            
        return x 

    # @set_device(device)
    # def latent_op(self, x):
    #     def sampling(args):
    #         z_mean, z_log_var = args
    #         batch_size = tf.shape(z_mean)[0]
    #         latent_dims = tf.shape(z_mean)[1]
    #         epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dims))
    #         return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

            
    #     vals =[]
    #     shape = tf.shape(x)
    #     x = tf.reshape(x, [shape[0], shape[1]*shape[-1]])
    #     for op in self.latent_list:
    #         vals.append(op(x))
    #     z_mean, z_log_var = vals[0], vals[1]
    #     return sampling(vals), z_mean, z_log_var
        
    # @tf.function
    # @set_device(device)        
    # def call(self, inputs):
    #     x = inputs
    #     # tf.print("input : \n{}\n".format(x))
    #     # tf.print("=======================")
    #     for i, layer in enumerate(self.exec_list):
    #         x = layer(x)
    #         # x = self.pool(x, i )

    #         # tf.print("layer name : {}\n".format(layer.name))

    #         # tf.print("x : \n{}\n".format(x))
    #     # tf.print("=======================")
    #     if self.use_latent : 
    #         # tf.print(x.shape)
    #         x, z_mean, z_log_var = self.latent_op(x)
    #         # tf.print("x", x[0])
    #         # tf.print("z_mean", z_mean[0])
    #         # tf.print("z_log_var", z_log_var[0])

    #         # tf.print("x \n{}\n, z_mean \n{}\n, z_log_var \n{}\n ".format(x, z_mean, z_log_var))
    #         return x, z_mean, z_log_var
    #     return x 
    
    @tf.function
    def pool(self, x, i):
        L = self.ds_D[i]
        Mp = L.shape[0]
        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sp.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.sparse.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse.reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        L = tf.cast(L, tf.float32)
        x = tf.sparse.sparse_dense_matmul(L, x) # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2,0,1]) # N x Mp x Fin
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
                name = "decoder", 
                trainable=True):
        super(Decoder, self).__init__(trainable=trainable, name = name)
        self.layer_info = layer_info
        self.adj = adj 
        self.activation = activation
        self.activation_layer = set_activation(self, self.activation)
        self.kernel_size = kernel_size
        self.vertex_size = vertex_size
        self.batch_size = batch_size
        self.use_latent = use_latent
        self.exec_list = []
        self.ds_U = ds_U
        self.A = A
        self.latent_list = []

    @set_device(device)
    def build(self, input_shape):
        if self.use_latent: 
            self.latent_list.append(LinearLayer(
                                    output_shape=self.layer_info[0]*self.adj[0].shape[0], 
                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.zeros,
                                    activation=None,
                                    # activation=self.activation,
                                    name="Dense_",
                                    trainable=True
                                    ))

        idx = 0
        preset = dict()
        preset['translation_invariant'] = True
        preset['num_weight_matrices'] = self.kernel_size
        preset['num_output_channels'] = self.layer_info[idx+1]
        preset['initializer'] = tf.keras.initializers.GlorotNormal
        preset['name'] = "decode_conv_" + str(idx)
        
        #to sparse tensor
        self.sparse_A_tensor = [] 
        for A in self.A:
            coo = A.tocoo()
            print(coo)
            indices = np.mat([coo.row, coo.col]).transpose()
            sparse_A_tensor = tf.sparse.SparseTensor(indices, coo.data, coo.shape)
            sparse_A_tensor = tf.sparse.expand_dims(sparse_A_tensor, axis=0)
            print(sparse_A_tensor.shape)
            self.sparse_A_tensor.append( tf.sparse.concat( 0, [sparse_A_tensor for _ in range(input_shape[0])] ) )
          
        print("what isj ialsfji ilsej lasiej ilsj ilesj ", 
            self.sparse_A_tensor[0].shape)
        layer = tfg.FeatureSteeredConvolutionKerasLayer(**preset)
        layer.is_layer = True
        self.exec_list.append(layer)
        act = tf.keras.layers.Activation(self.activation)
        act.is_layer = False
        self.exec_list.append(act)
        
        #     # self.exec_list.append(tf.keras.layers.BatchNormalization())
        #     self.exec_list.append(self.activation_layer())


        # for idx in range(len(self.layer_info)-2):
        idx = 0
        # self.exec_list.append(NLayer(input_shape=self.layer_info[idx],
        #                                 output_shape=self.layer_info[idx+1],
        #                                 adj=self.adj[0],
        #                                 kernel_size=self.kernel_size,
        #                                 activation=self.activation,
        #                                 # activation=None,
        #                                 name=self.name[0]+"Layer"+str(idx),
        #                                 trainable=self.trainable
        #                                 )
        #                         )
        # self.exec_list.append(NLayer(input_shape=self.layer_info[idx+1],
        #                                 output_shape=self.layer_info[idx+2],
        #                                 adj=self.adj[0],
        #                                 kernel_size=self.kernel_size,
        #                                 # activation=self.activation,
        #                                 activation=None,
        #                                 name=self.name[0]+"Layer"+str(idx+1),
        #                                 trainable=self.trainable
        #                                 )
        #                         )
        print("build complete Decoder")
        
    @tf.function
    @set_device(device)
    def call(self, inputs):
        x = inputs
        if self.use_latent : 
            x = self.latent_list[0](x)
            x = tf.reshape(x, [tf.shape(x)[0], self.vertex_size, -1])
        i = 0 
        for layer in self.exec_list:
            
            if layer.is_layer == True:
                x = layer([x, self.sparse_A_tensor[0]])
                i += 1
            else : 
                x = layer(x)


            
        return x 

    # @tf.function
    # @set_device(device)
    # def call(self, inputs):
    #     x = inputs
    #     # tf.print(x)
    #     # x=feature_normalize(x)
    #     first_event = True

        

    #     for layer in self.exec_list:

    #         x = layer(x)
            
    #         if first_event and self.use_latent: 
                
    #             first_event = False
    #             x = tf.reshape(x,[tf.shape(x)[0], self.vertex_size, -1])
    #             x=feature_normalize(x)


            
    #     return x 

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, 
                output_shape, 
                kernel_initializer = tf.keras.initializers.GlorotNormal,
                use_bias = False, 
                bias_initializer = tf.keras.initializers.zeros,
                activation=None,
                name = "Linear_Layer_",
                trainable=True):
        super(LinearLayer, self).__init__(trainable=trainable,name=name)
        self.output_channel = output_shape
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.activation_func = set_activation(self, self.activation)

    @set_device(device)
    def build(self, inputs_shape): 
        self.Fin = inputs_shape[-1]

        self.W = self.add_weight (name = "W",
                                     shape=[self.Fin, self.output_channel],
                                     initializer = self.kernel_initializer
                                     )
        if self.use_bias : 
            self.b = self.add_weight (name = "b",
                                     shape=[self.output_channel],
                                     initializer = self.bias_initializer
                                     )
    
    @tf.function
    @set_device(device)
    def call(self, inputs) : 
        # if len(inputs.shape) == 2 : 
        #     inputs=tf.expand_dims(inputs, axis=1)
        # tf.print(inputs.shape)
        result = tf.matmul(inputs, self.W)
        if self.use_bias : 
            result += self.b 
        return result if self.activation_func == None else self.activation_func(result)
        
