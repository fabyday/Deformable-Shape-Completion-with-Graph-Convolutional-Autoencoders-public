"""
    Network Model Implementation

        function loss : it is calculate loss using vae.
        class Model : model class is wrapper class for Encoder, Decoder. it contains Encoder, Decoder.
            class Encoder : it contains NLayer, LinearLayer.
            class Decoder : it contains NLayer, LinearLayer.
                class NLayer : FeastNet Imeplementaion class.
                class LinearLayer : it's custom Dense class.


"""

import tensorflow as tf
import numpy as np 
from .logger import set_device, timeset #set_device, timeset
from .utils import set_activation # it's legacy.

#for pooling.
import math 
import heapq
import scipy.sparse as sp



"""
experimental model.
"""

device = "GPU:1" # set GPU.

# not VAE just L2 Loss Function.
# @tf.function 
# def loss(y_label, y_pred, args=None):
#     return tf.keras.backend.mean(tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y_label-y_pred, 2.0), axis=-1)), axis=-1))

@tf.function
def loss(y_label, y_pred, args=None):
        def log_normal_pdf(sample, mean, logvar, raxis=1): 
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        
        total_loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y_label-y_pred, 2.0), axis=-1)), axis=-1)
        if type(args) == list and len(args) == 2:            
            latent_z = args[0]
            z_mean = args[1]
            z_log_var = args[2]

            # kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var) 
            kl_loss= tf.keras.backend.square(z_mean) + tf.keras.backend.exp(z_log_var)- z_log_var - 1
            kl_loss = tf.keras.backend.sum(kl_loss, axis = - 1)
            # logpz = log_normal_pdf(latent_z, 0., 0.)
            # logpz_x = log_normal_pdf(latent_z, z_mean, z_log_var)
            # kl_loss = logpz - logpz_x
            
            # total_loss += (10e-8 * kl_loss)
            total_loss += 0.5*( kl_loss)

        total_loss = tf.keras.backend.mean(total_loss, axis = 0)
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
        

    @set_device(device)
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
                name = "encoder", 
                trainable=True):
        super(Encoder, self).__init__(trainable=trainable, name = name)
        self.layer_info = layer_info
        self.adj = adj 
        self.latent_size = latent_size
        self.use_latent = use_latent
        self.activation_name = activation 
        self.activation_layer = set_activation(self, self.activation_name)# it return activation class. use activation build it. see also utils.set_activation
        self.kernel_size = kernel_size
        self.exec_list = []
        self.latent_list = []
        self.face=face
        self.ds_D = ds_D

    @set_device(device)
    def build(self, input_shape):

        # self.exec_list.append(LinearLayer(  
        #                                 output_shape = self.layer_info[1],
        #                                 activation=None,
        #                                 # activation=self.activation,
        #                                 name = "C_LAYER",
        #                                 trainable=True
        #                                 ))
        # # self.exec_list.append(tf.keras.layers.BatchNormalization())
        # self.exec_list.append(self.activation_name_layer())

        idx = 0 
        self.exec_list.append(LinearLayer(
                                    output_shape=self.layer_info[idx+1],
                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.zeros,
                                    # activation=self.activation_name,

                                    activation=None,
                                    use_batchnorm=True,
                                    name="e_Dense_1",
                                    trainable=True
                                    ))
        # self.exec_list.append( NLayer(input_shape=self.layer_info[idx],
        #                                 output_shape=self.layer_info[idx+1],
        #                                 adj=self.adj[0],
        #                                 kernel_size=self.kernel_size,
        #                                 activation=self.activation_name,
        #                                 # activation= None,
        #                                 name=self.name[0]+"_Layer_"+str(idx),
        #                                 trainable=self.trainable
        #                                 )
        #                         )

        # self.exec_list.append(testlayer('inputlin', self.layer_info[idx+1], self.activation) )
        print("Test")
        print("self.", self.adj[1].shape)
        self.exec_list.append( NLayer(input_shape=self.layer_info[idx+1],
                                        output_shape=self.layer_info[idx+2],
                                        adj=self.adj[0],
                                        kernel_size=self.kernel_size,
                                        activation=self.activation_name,
                                        # activation= None,
                                        use_batchnorm=True,
                                        name=self.name[0]+"_Layer_"+str(idx+1),
                                        trainable=self.trainable
                                        )
                                )
        self.exec_list.append( NLayer(input_shape=self.layer_info[idx+2],
                                        output_shape=self.layer_info[idx+3],
                                        adj=self.adj[0],
                                        kernel_size=self.kernel_size,
                                        activation=self.activation_name,
                                        # activation= None,
                                        use_batchnorm=True,
                                        name=self.name[0]+"_Layer_"+str(idx+2),
                                        trainable=self.trainable
                                        )
                                )
        self.exec_list.append( NLayer(input_shape=self.layer_info[idx+3],
                                        output_shape=self.layer_info[idx+4],
                                        adj=self.adj[0],
                                        kernel_size=self.kernel_size,
                                        # activation=self.activation_name,
                                        activation= None,
                                        use_batchnorm=False,
                                        name=self.name[0]+"_Layer_"+str(idx+3),
                                        trainable=self.trainable
                                        )
                                )                                                                
        
        # # self.exec_list.append(self.activation_layer())
        # self.exec_list.append(self.activation_layer)
        self.exec_list.append(Pool(name="pool", pooling_type="mean", axis=1))        
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
                                    # activation=self.activation_name,
                                    activation=None,
                                    name="e_Dense_1",
                                    trainable=True
                                    ))
            # self.latent_list.append(tf.keras.layers.Dense(self.latent_size))
            self.latent_list.append(LinearLayer(
                                    output_shape=self.latent_size,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.zeros,
                                    # activation=self.activation_name,
                                    activation=None,
                                    name="e_Dense_2",
                                    trainable=True
                                    ))



                                                        
        
           

    # @tf.function
    # @set_device(device)        
    # def call(self, inputs):
    #     x = inputs
    #     for i, layer in enumerate(self.exec_list):
    #         x = layer(x)
    #         # x = self.pool(x,i)
    #     if self.use_latent : 
    #         x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1]*tf.shape(x)[-1]])
    #         x = self.latent_list[0](x)
    #         # tf.print(x[0])
            
    #     return x 

    @set_device(device)
    def latent_op(self, x):
        def sampling(args):
            z_mean, z_log_var = args
            # batch_size = tf.shape(z_mean)[0]
            
            # latent_dims = tf.shape(z_mean)[1]
            # epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dims))
            epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))

            return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

            
        vals =[]
        # shape = tf.shape(x)
        # x = tf.reshape(x, [shape[0], shape[1]*shape[-1]])
        for op in self.latent_list:
            vals.append(op(x))
        z_mean, z_log_var = vals[0], vals[1]
        print("z_mean", z_mean.shape)
        return sampling(vals), z_mean, z_log_var
        
    @tf.function
    @set_device(device)        
    def call(self, inputs):
        x = inputs
        # tf.print("input : \n{}\n".format(x))
        # tf.print("=======================")
        for i, layer in enumerate(self.exec_list):
            x = layer(x)
        print("enc", x.shape)

        if self.use_latent : 

            x, z_mean, z_log_var = self.latent_op(x)
            print("enc x shape", x.shape)
            return x, z_mean, z_log_var
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
                name = "decoder", 
                trainable=True):
        super(Decoder, self).__init__(trainable=trainable, name = name)
        self.layer_info = layer_info
        self.adj = adj 
        self.activation_name= activation
        self.activation_layer = set_activation(self, self.activation_name)
        self.kernel_size = kernel_size
        self.vertex_size = vertex_size
        self.batch_size = batch_size
        self.use_latent = use_latent
        self.exec_list = []
        self.latent_list =[]
        self.ds_U = ds_U

    @set_device(device)
    def build(self, input_shape):
        if self.use_latent: 
            print("self vertice size of all", self.vertex_size)
            self.latent_list.append(LinearLayer(
                                    output_shape=self.layer_info[0]*self.vertex_size, 
                                    # output_shape=128, 
                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.zeros,
                                    # activation=None,
                                    activation=self.activation_name,
                                    use_batchnorm=True,
                                    name="Dense_",
                                    trainable=True
                                    ))

        # self.exec_list.append(MeanPool(name="unpool"))
            # self.exec_list.append(tf.keras.layers.Dense(self.layer_info[0]*self.vertex_size))
        
        #     # self.exec_list.append(tf.keras.layers.BatchNormalization())
        #     self.exec_list.append(self.activation_layer())


        # for idx in range(len(self.layer_info)-2):
        idx = 0
        self.exec_list.append(NLayer(input_shape=self.layer_info[idx],
                                        output_shape=self.layer_info[idx+1],
                                        adj=self.adj[0],
                                        kernel_size=self.kernel_size,
                                        activation=self.activation_name,
                                        # activation=None,
                                        use_batchnorm=True,
                                        name=self.name[0]+"Layer"+str(idx),
                                        trainable=self.trainable
                                        )
                                )
        # self.exec_list.append(testlayer('inputlin', self.layer_info[idx+1], None) )
        self.exec_list.append(NLayer(input_shape=self.layer_info[idx+1],
                                        output_shape=self.layer_info[idx+2],
                                        adj=self.adj[0],
                                        kernel_size=self.kernel_size,
                                        activation=self.activation_name,
                                        # activation=None,
                                        use_batchnorm=True,
                                        name=self.name[0]+"Layer"+str(idx+1),
                                        trainable=self.trainable
                                        )
                                )
        self.exec_list.append(NLayer(input_shape=self.layer_info[idx+2],
                                                output_shape=self.layer_info[idx+3],
                                                adj=self.adj[0],
                                                kernel_size=self.kernel_size,
                                                activation=self.activation_name,
                                                # activation=None,
                                                use_batchnorm=True,
                                                name=self.name[0]+"Layer"+str(idx+2),
                                                trainable=self.trainable
                                                )
                                        )
        # self.exec_list.append(NLayer(input_shape=self.layer_info[idx+3],
        #                                         output_shape=self.layer_info[idx+4],
        #                                         adj=self.adj[0],
        #                                         kernel_size=self.kernel_size,
        #                                         # activation=self.activation_name,
        #                                         activation=None,
        #                                         name=self.name[0]+"Layer"+str(idx+3),
        #                                         trainable=self.trainable
        #                                         )
        #                                 )
        self.exec_list.append(LinearLayer(
                                    # output_shape=self.layer_info[0]*self.vertex_size, 
                                    output_shape= self.layer_info[idx+4], 
                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.zeros,
                                    activation=None,
                                    use_batchnorm=False,
                                    # activation=self.activation,
                                    name="Dense_",
                                    trainable=True
                                    ))
                
        print("build complete Decoder")
        
    @tf.function
    @set_device(device)
    def call(self, inputs):
        x = inputs
        print("dec1", x.shape)

        if self.use_latent : 
            x=self.latent_list[0](x)
            x = tf.reshape(x, [tf.shape(x)[0], self.vertex_size, -1])

        for layer in self.exec_list:
            x = layer(x)
            
        return x 
    def pool(self, x, i):
        L = self.ds_U[i]
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




class NLayer(tf.keras.layers.Layer):
    def __init__(self, 
                input_shape,
                output_shape, 
                adj = None,
                kernel_size = 9,
                activation=tf.keras.activations.relu,
                use_batchnorm = False,
                name = "Layer_",
                trainable=True):
        super(NLayer, self).__init__(trainable=trainable,name=name)
        self.input_chanel = input_shape
        # self.output_shape = output_shape
        self.output_chanel = output_shape

        self.adj = adj
        self.kernel_size = kernel_size
        self.activation_name= activation
        self.activation_func = set_activation(self, self.activation_name)
        
        self.use_batchnorm = use_batchnorm
        self.batch_norm = tf.keras.layers.BatchNormalization()
    @set_device(device)                              
    def build(self, input_shape):
        batch_size, vertex_size, coord_size = input_shape

        # if self.use_batchnorm :
        #     self.batch_norm = tf.keras.layers.BatchNormalization()


        self.adj = tf.constant(self.adj, dtype=tf.int32)
        self.W = self.add_weight (name = "W",
                                     shape=[coord_size, self.kernel_size,self.output_chanel],
                                     initializer = tf.keras.initializers.GlorotNormal
                                     )
        #for testing
        # self.W_x = self.add_weight (name = "W_x",
        #                              shape=[coord_size, self.kernel_size,self.output_chanel],
        #                              initializer = tf.keras.initializers.GlorotNormal
        #                              )


        self.b = self.add_weight (name = "b",
                                     shape=[self.output_chanel],
                                     initializer = tf.keras.initializers.zeros
                                     )


        self.u = self.add_weight (name = "u",
                                     shape=[coord_size, self.kernel_size],
                                     initializer = tf.keras.initializers.GlorotNormal
                                     )
        
                                     

        self.c = self.add_weight (name = "c",
                                     shape=[self.kernel_size],
                                     initializer = tf.keras.initializers.zeros
                                     )


    @tf.function       
    @set_device(device)                              
    def neighbor(self, x):
        
        batch_size, _, coord_size = x.shape.as_list()
        

        
        padding_feature = tf.zeros([batch_size, 1, coord_size], tf.float32)
        
        padded_input = tf.concat([padding_feature, x], 1)
        def compute_nb_feature(input_f):
            return tf.gather(input_f, self.adj)
        
        # expected ...
        # if call by self.calc_diff func
        # [ batch_size, vertice_size, neighbor_num, kernel_size ]
        #                       or 
        # if call by self.call func
        # [batch_size, vertice_size, neighbor_num, output_size*kernel_size]
        total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
        return total_nb_feature
    @tf.function
    @set_device(device)                              
    def calc_diff(self, x) : 
        """
            invariant mapping. u(X_point - X_neighbor)
        """
        
        # ux, uv := [batch_size, vertice_size, kernel_size]
        ux = tf.map_fn(lambda x: tf.matmul(x, self.u), x)
        # vx = tf.map_fn(lambda x: tf.matmul(self.v, x), x)
        # vx = tf.map_fn(lambda x: tf.matmul(x, self.u), x)

        
        # patch := [ batch_size, vertice_size, neighbor_num, kernel_size ]
        # patches = self.neighbor(vx)
        patches = self.neighbor(ux)
        # ux := [batch_size,vertice_size, 1, kernel_size]
        ux = tf.expand_dims(ux, 2)
        # patches = u(X_point + X_neighbor) + c 
        # patches = tf.add(ux, patches)
        patches = tf.add(ux, -patches)
        
        patches = tf.add(patches, self.c)
        
        
        patches = tf.nn.softmax(patches)
        return patches

    def feature_normalize(self, inputs, alpha=0.9, epsilon=10e-8):
        #max_val, min_val := [batch size, 1, Fin]
        max_val = tf.keras.backend.max(inputs, axis=1, keepdims=True)
        min_val = tf.keras.backend.min(inputs, axis=1, keepdims=True)
        
        # if max_val == min_val, max and mean is inputs, +epssilon

        idx = tf.math.equal(max_val, min_val)
        max_val = tf.where(idx, max_val+epsilon, max_val)
        min_val = tf.where(idx, min_val-epsilon, min_val)
        inputs = 2 * alpha * (inputs-min_val)/(max_val-min_val) - alpha
        

        

        return inputs

    @tf.function
    @set_device(device)                              
    def call(self, inputs):
        batch_size, vertice_size, coord_size = inputs.shape.as_list()
        neighbor_num = self.adj.shape[-1]
        adj_size = tf.math.count_nonzero(self.adj, -1) # 5023, 1
        non_zeros = tf.math.not_equal(adj_size, 0)
        adj_size = tf.cast(adj_size, tf.float32)

        #5023, 1
        adj_size = tf.where(non_zeros, tf.math.reciprocal(adj_size), tf.zeros_like(adj_size))
        adj_size = tf.reshape(adj_size, [1, vertice_size, 1, 1])
        x = inputs
        
        w = tf.reshape(self.W, [coord_size, self.kernel_size*self.output_chanel])
        wx = tf.map_fn(lambda x : tf.matmul(x, w), x) #batch_size, vertice_size, self.kerenl_size*self.output_shape
        
        # [batch_size, vertice_size, neighbor_num, kerenl_size*output_size]
        patches = self.neighbor(wx)
        q = self.calc_diff(x)

        patches = tf.reshape(patches, [batch_size, vertice_size, neighbor_num, self.kernel_size, self.output_chanel])
        # patches = tf.transpose(patches, [4, 0 ,1, 2, 3])
        # [batch_size, vertice_size, neighbor_num, kerenl_size, 1]
        q = tf.expand_dims(q, -1)
        patches = tf.multiply(q, patches)
        # patches = tf.transpose(patches, [1, 2, 3, 4, 0])

        #[batch_size, vertice_size, kernel_size, output_size]
        patches = tf.reduce_sum(patches, axis = 2)
        #[vertice_size, 1, 1]
        # print("tesT", adj_size.shape)
        patches = tf.multiply(adj_size, patches) 

        patches = tf.reduce_sum(patches, axis = 2)
        patches = patches + self.b 
        
        # #test
        # w_x = tf.reshape(self.W_x, [coord_size, self.kernel_size*self.output_chanel])
        # point_x = tf.map_fn(lambda x : tf.matmul(x, w_x), x)
        # # batch_size, vertice_size, self.kerenl_size, self.output_shape
        # x_point_patches = tf.reshape(point_x, [batch_size, vertice_size, self.kernel_size, self.output_chanel]) 
        # x_point_patches = tf.reduce_sum(x_point_patches, axis=2)
        # patches += x_point_patches

        result = patches

        if self.use_batchnorm : 
            result = self.batch_norm(result)
                
        if self.activation_func != None :
            result = self.activation_func(patches)
            

        return result 

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, 
                output_shape, 
                kernel_initializer = tf.keras.initializers.GlorotNormal,
                use_bias = False, 
                bias_initializer = tf.keras.initializers.zeros,
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
        self.batch_norm = tf.keras.layers.BatchNormalization()

    @set_device(device)
    def build(self, inputs_shape): 

        # if self.use_batchnorm : 
        #     self.batch_norm = tf.keras.layers.BatchNormalization()

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
    @set_device(device)
    def call(self, inputs) : 
        # if len(inputs.shape) == 2 : 
        #     inputs=tf.expand_dims(inputs, axis=1)
        # tf.print(inputs.shape)
        print(inputs.shape)
        assert len(inputs.shape) == 3, "input_shape length must be 3-dimension. it is {}".format(len(inputs))

        result = tf.map_fn(lambda x: tf.matmul(x, self.W), inputs)
        if self.use_bias : 
            result += self.b 
        assert len(result.shape) == 3, "return value length must be 3-dimension. it is {}".format(len(inputs))
        
        if self.use_batchnorm : 
            result = self.batch_norm(result)

        if self.activation_func != None:
             result = self.activation_func(result)

        return result 
        

class Pool(tf.keras.layers.Layer):
    """
        pooooooooooooooooooooooooooool
        provide mean, max, min pooling.
    """
    
    __setter_dict = {
                    "mean" : tf.keras.backend.mean,
                    "max" : tf.keras.backend.max,
                    "min" : tf.keras.backend.min
                    }

    def __init__(self, name, pooling_type, axis=1, keep_dims=True):
        """
            name : layer's name
            pooling type : choose string in {max, mean, min}. it find pooling type that you want. 

        """
        super(Pool, self).__init__(name="pool_")
        self.pooling_type = pooling_type
        self.axis = axis
        self.keep_dims = keep_dims
        self.pool_func = self.select_pool(pooling_type)

    def build(self, inputs_shape):
        """
            build do nothing.
        """
        pass
    
    def select_pool(self, pooling_type):
        """
        
            pooling_type : choose string in {max, mean, min}. it find pooling type that you want. 

        """
        
        def wrapper(function, *args, **kwrags):
            def pool_func(inputs):
                if function == None : 
                    return inputs
                return function(inputs, *args, **kwrags)
            return pool_func

        choosed_function = None
        if type(pooling_type) != str : 
            raise Exception("pooling type is not string")

        if pooling_type in Pool.__setter_dict:
            choosed_function =  Pool.__setter_dict[pooling_type]
        else : 
            raise Exception("uncorrect pooling typename")

        return wrapper(choosed_function, self.axis, self.keep_dims)



    def call(self, inputs):
        result = self.pool_func(inputs)
        print("pooling", result.shape)
        return result

class MeanPool(tf.keras.layers.Layer):
    """
        nameing mistake. it's unpooling.
        only provide average unpooling.
    """
    def __init__(self, name):
        """
            name : layer's name
            pooling type : choose string in {max, mean, min}. it find pooling type that you want. 

        """
        super(MeanPool, self).__init__(name="pool_mean")
       

    def build(self, inputs_shape):
        """
            build do nothing.
        """
        pass
    
  
    def call(self, inputs):
        # batch_size, 1, 128
        # inputs= tf.expand_dims(inputs, axis=1)
        # assert len(inputs.shape) != 3 , "length is not 3."
        result = tf.tile(inputs, [1,5023,1])
        # assert result.shape != [4, 5023, 128], "error"
        print("mean pool", result.shape)
        return result


