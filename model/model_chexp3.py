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
        
        total_loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y_label-y_pred, 2.0), axis=-1)), axis=-1)
      
        total_loss = tf.keras.backend.mean(total_loss )
        
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


    # @tf.function
    # @set_device(device)
    # def call(self, inputs): 
    #     outputs = self.encoder(inputs)
    #     return self.decoder(outputs)

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
        self.layer_info = layer_info
        self.adj = adj 
        self.latent_size = latent_size
        self.use_latent = use_latent
        self.activation_name = activation 
        # self.activation_layer = set_activation(self, self.activation_name)# it return activation class. use activation build it. see also utils.set_activation
        self.kernel_size = kernel_size
        self.exec_list = []
        self.latent_list = []
        self.face=face
        self.ds_D = ds_D
        self.A = A
        self.sparse_A_tensor = [] 
        self.kernel_initializer = kernel_initializer


    def build(self, input_shape):
                                                                                                
        self.exec_list.append(ConvNet(output_channel = 16, 
                                        kernel_initializer = set_initializer(self.kernel_initializer),
                                        activation=None, 
                                        neighbor=self.adj[0], 
                                        name = "true", 
                                        trainable=True))
        self.exec_list.append(set_activation(self, self.activation_name))
        self.exec_list.append(ConvNet(output_channel = 16, 
                                        kernel_initializer = set_initializer(self.kernel_initializer),
                                        activation=None, 
                                        neighbor=self.adj[0], 
                                        name = "true", 
                                        trainable=True))
        self.exec_list.append(set_activation(self, self.activation_name))                                        
                                       

        if self.use_latent : 
            # pass
            self.exec_list.append(lambda x : tf.reshape(x, [-1, 1, tf.shape(x)[1]*tf.shape(x)[2]]))
            self.latent_list.append(LinearLayer(
                                    # output_shape=self.latent_size,
                                    output_shape=32,
                                    kernel_initializer=set_initializer(self.kernel_initializer),
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.TruncatedNormal(),
                                    activation=None,
                                    use_batchnorm=False,
                                    name="sig",
                                    trainable=True
                                    ))
            # self.latent_list.append(LinearLayer(
            #                         output_shape=self.latent_size,
            #                         kernel_initializer=set_initializer(self.kernel_initializer),
            #                         use_bias = True,
            #                         bias_initializer=tf.keras.initializers.TruncatedNormal(),
            #                         activation=None,
            #                         use_batchnorm=False,
            #                         name="mu",
            #                         trainable=True
            #                         ))



                                                        
        
    @tf.function
    @set_device("GPU:0")        
    def call(self, inputs):
        x = inputs
        # tf.print("input : \n{}\n".format(x))
        # tf.print("=======================")

        for layer in (self.exec_list):
    
            x = layer(x)
            # print(layer)
        # print("enc", x.shape)

        if self.use_latent : 
            # x, z_mean, z_log_var = self.latent_op(x)
            # print("enc x shape", x.shape)
            # return x, z_mean, z_log_var
            return self.latent_list[0](x)
        return x 
    
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
                kernel_initializer = None,
                name = "decoder", 
                trainable=True):
        super(Decoder, self).__init__(trainable=trainable, name = name)
        self.layer_info = layer_info
        self.adj = adj 
        self.activation_name= activation
        self.kernel_size = kernel_size
        self.vertex_size = vertex_size
        self.batch_size = batch_size
        self.use_latent = use_latent
        self.exec_list = []
        self.ds_U = ds_U
        self.A = A
        self.sparse_A_tensor = []
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        for A in self.A:
            coo = A.tocoo()
            # print(coo)
            indices = np.mat([coo.row, coo.col]).transpose()
            sparse_A_tensor = tf.sparse.SparseTensor(indices, coo.data, coo.shape)
            sparse_A_tensor = tf.sparse.expand_dims(sparse_A_tensor, axis=0)
            # print(sparse_A_tensor.shape)
            self.sparse_A_tensor.append( tf.sparse.concat( 0, [sparse_A_tensor for _ in range(input_shape[0])] ) )
                  
        if self.use_latent: 

            self.exec_list.append((LinearLayer(
                                    output_shape=16*self.vertex_size, 
                                    # output_shape=128, 
                                    kernel_initializer=set_initializer(self.kernel_initializer),                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.TruncatedNormal(),
                                    # activation=self.activation_name,
                                    # use_batchnorm=True,
                                    activation=None,
                                    use_batchnorm=False,
                                    name="Linear_Input",
                                    trainable=True
                                    )))
            
            # self.exec_list.append((tf.keras.layers.BatchNormalization()))
            self.exec_list.append((lambda x : tf.reshape(x, [tf.shape(x)[0], self.vertex_size, -1])))
            
            
            

        self.exec_list.append(ConvNet(output_channel = 16, 
                                        kernel_initializer = set_initializer(self.kernel_initializer),
                                        activation=None, 
                                        neighbor=self.adj[0], 
                                        name = "true", 
                                        trainable=True))
        self.exec_list.append(set_activation(self, self.activation_name))

        self.exec_list.append(ConvNet(output_channel = 16, 
                                        kernel_initializer = set_initializer(self.kernel_initializer),
                                        activation=None, 
                                        neighbor=self.adj[0], 
                                        name = "true", 
                                        trainable=True))
        self.exec_list.append(set_activation(self, self.activation_name))                                        
        self.exec_list.append(ConvNet(output_channel = 16, 
                                        kernel_initializer = set_initializer(self.kernel_initializer),
                                        activation=None, 
                                        neighbor=self.adj[0], 
                                        name = "true", 
                                        trainable=True))
        self.exec_list.append(set_activation(self, self.activation_name))                                        
        self.exec_list.append(ConvNet(output_channel = 3, 
                                        kernel_initializer = set_initializer(self.kernel_initializer),
                                        activation=None, 
                                        neighbor=self.adj[0], 
                                        name = "true", 
                                        trainable=True))                                                                                

        print("build complete Decoder")
        
    @tf.function
    @set_device("GPU:1")
    def call(self, inputs):
        x = inputs

        for layer in (self.exec_list):
            print("x.shape", x.shape)
            x = layer(x)
            # print(layer.name,x.shape)

            
        return x 


class ConvNet(tf.keras.layers.Layer):
    def __init__(self, output_channel, kernel_initializer, activation, neighbor, name, trainable=True):
        super(ConvNet, self).__init__(trainable=trainable, name = name)
        self.neighbor = neighbor
        self.output_channel = output_channel 
        self.kernel_initializer = kernel_initializer
        self.act = activation

    def build(self, input_shape):
        #weight for point
        _, self.vertice, self.Fin = input_shape
        self.Wx = self.add_weight (name = "Wx",
                                     shape=[self.Fin, self.output_channel],
                                     initializer =  self.kernel_initializer
                                     )
        #weight for neighbor
        self.Wn = self.add_weight (name = "Wn",
                                     shape=[self.Fin, self.output_channel],
                                     initializer = self.kernel_initializer
                                     )

        self.b = self.add_weight (name = "b",
                                     shape=[self.output_channel],
                                     initializer =  tf.keras.initializers.zeros()
                                     )


    def point_calc(self, x):
        return tf.map_fn(lambda x  : tf.matmul(x, self.Wx), x)
        

    def neighbor_calc(self, x):

        def compute_nb_feature(inputs):
            return tf.gather(inputs, self.neighbor)
        
        padding_feature = tf.zeros([tf.shape(x)[0], 1, self.Fin], tf.float32)
        padded_input = tf.concat([padding_feature, x], 1)

        nb_feature = tf.reduce_sum(tf.map_fn(compute_nb_feature, padded_input), axis=2)/tf.cast(tf.shape(self.neighbor)[-1], tf.float32)
        nb_feature = tf.map_fn(lambda x : tf.matmul(x, self.Wn), nb_feature)

        return nb_feature


    def call(self, x):
        return self.point_calc(x) + self.neighbor_calc(x) + self.b
    




class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, 
                output_shape, 
                kernel_initializer = tf.keras.initializers.GlorotNormal,
                use_bias = False, 
                bias_initializer = tf.keras.initializers.zeros(),
                activation=None,
                use_batchnorm=False,
                name = "Linear_Layer_",
                trainable=True):
        super(LinearLayer, self).__init__(trainable=trainable,name=name)
        self.output_channel = output_shape
        self.activation_name= activation
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.activation_func = set_activation(self, self.activation_name)
        
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm : 
            self.batch_norm = tf.keras.layers.BatchNormalization()


    def build(self, inputs_shape): 


        print(inputs_shape)
        assert len(inputs_shape) == 3, \
            "input_shape length must be 3-dimension. it is {}".format(len(inputs_shape))
        self.Fin = inputs_shape[-1]

        print(self.Fin, self.output_channel)
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
    def call(self, inputs) : 
        
        assert len(inputs.shape) == 3, "input_shape length must be 3-dimension. it is {}".format(len(inputs))

        result = tf.map_fn(lambda x: tf.matmul(x, self.W), inputs)
        print("result shape", result.shape)
        if self.use_bias : 
            result += self.b 
        assert len(result.shape) == 3, "return value length must be 3-dimension. it is {}".format(len(inputs))
        
        return result 
        



