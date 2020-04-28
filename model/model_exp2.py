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
        def log_normal_pdf(sample, mean, logvar, raxis=1): 
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        
        total_loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y_label-y_pred, 2.0), axis=-1)), axis=-1)
        # total_loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y_label-y_pred, 2.0), axis=-1)))
        # total_loss =tf.keras.losses.binary_crossentropy( y_pred, y_label )
        # total_loss = tf.keras.backend.mean(total_loss, axis=-1)
        # print("total loss shape is ", total_loss, tf.keras.backend.sum(total_loss[0]), tf.keras.backend.mean(total_loss[0]))
        if type(args) == list and len(args) == 3:            
            
            latent_z = args[0]
            z_mean = args[1]
            z_log_var = args[2]

            assert len(latent_z.shape) == 3, "latent_z.shape is not 3d"
            assert len(z_mean.shape) == 3, "latent_z.shape is not 3d"
            assert len(z_log_var.shape) == 3, "latent_z.shape is not 3d"

            z_mean = tf.squeeze(z_mean)
            z_log_var = tf.squeeze(z_log_var)
            latent_z = tf.squeeze(latent_z)

    
            kl_loss=  tf.keras.backend.mean(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
            # kl_loss = tf.keras.backend.sum(kl_loss, axis = - 1)

            # kl_loss = tf.keras.losses.KLD(y_label, y_pred)
            # print(total_loss, kl_loss)
            # kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
            # kl_loss = tf.keras.backend.mean(kl_loss, axis=-1)
            kl_loss *= +10e-8
            # print("total loss shape is ", kl_loss)
            total_loss += kl_loss
        total_loss = tf.keras.backend.mean(total_loss )
            # kl_loss = tf.reduce_mean(tf.reduce_sum(1 + 2 * z_log_var - tf.square(z_mean) - tf.square(tf.exp(z_log_var)), 1))
            # total_loss -= 10e-6*( kl_loss)
        # total_loss = tf.keras.backend.mean(total_loss, axis = 0)
        
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

        for A in self.A:
            coo = A.tocoo()
            # print(coo)
            indices = np.mat([coo.row, coo.col]).transpose()
            sparse_A_tensor = tf.sparse.SparseTensor(indices, coo.data, coo.shape)
            sparse_A_tensor = tf.sparse.expand_dims(sparse_A_tensor, axis=0)
            # print(sparse_A_tensor.shape)
            self.sparse_A_tensor.append( tf.sparse.concat( 0, [sparse_A_tensor for _ in range(input_shape[0])] ) )
          

        
        
        self.exec_list.append((LinearLayer(
                                    output_shape=self.layer_info[1],
                                    kernel_initializer=set_initializer(self.kernel_initializer),
                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.TruncatedNormal(),
                                    # activation=self.activation_name,
                                    # use_batchnorm=True,
                                    activation=None,
                                    use_batchnorm=False,
                                    name="InputLayer",
                                    trainable=True
                                    ),False))
        self.exec_list.append((tf.keras.layers.BatchNormalization(), False))
        self.exec_list.append((set_activation(self, self.activation_name), False))

        for idx, output_size in enumerate(self.layer_info[2:]): 
                
            preset = dict()
            preset['translation_invariant'] = True
            preset['num_weight_matrices'] = self.kernel_size
            preset['num_output_channels'] = output_size
            preset['initializer'] = set_initializer(self.kernel_initializer)
            preset['name'] = "enc_" + str(idx)
            
            layer = tfg.FeatureSteeredConvolutionKerasLayer(**preset)
            
            self.exec_list.append((layer, True))
            if idx != len(self.layer_info[2:]) - 1 :
                self.exec_list.append((tf.keras.layers.BatchNormalization(), False))
            self.exec_list.append((set_activation(self, self.activation_name), False))

                                                                                                
        

        if self.use_latent : 
            self.exec_list.append((Pool(name="pool", pooling_type="mean", axis=1), False))        

            self.latent_list.append(LinearLayer(
                                    output_shape=self.latent_size,
                                    kernel_initializer=set_initializer(self.kernel_initializer),
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.TruncatedNormal(),
                                    activation=None,
                                    use_batchnorm=False,
                                    name="sig",
                                    trainable=True
                                    ))
            self.latent_list.append(LinearLayer(
                                    output_shape=self.latent_size,
                                    kernel_initializer=set_initializer(self.kernel_initializer),
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.TruncatedNormal(),
                                    activation=None,
                                    use_batchnorm=False,
                                    name="mu",
                                    trainable=True
                                    ))



                                                        
        
           

    def latent_op(self, x):
        def sampling(args):
            z_mean, z_log_var = argsims))
            epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))

            return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

            
        vals =[]
        # shape = tf.shape(x)
        # x = tf.reshape(x, [shape[0], shape[1]*shape[-1]])
        for op in self.latent_list:
            vals.append(op(x))
        z_mean, z_log_var = vals[0], vals[1]
        # print("z_mean", z_mean.shape)
        return sampling(vals), z_mean, z_log_var
        
    @tf.function
    @set_device("GPU:0")        
    def call(self, inputs):
        x = inputs
        # tf.print("input : \n{}\n".format(x))
        # tf.print("=======================")

        for layer, need_neighbor in (self.exec_list):
            if need_neighbor : 
                x = layer([x, self.sparse_A_tensor[0]])
            else : 
                x = layer(x)
            # print(layer)
        # print("enc", x.shape)

        if self.use_latent : 
            x, z_mean, z_log_var = self.latent_op(x)
            # print("enc x shape", x.shape)
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
                                    # output_shape=self.layer_info[0]*self.vertex_size, 
                                    output_shape=128, 
                                    kernel_initializer=set_initializer(self.kernel_initializer),                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.TruncatedNormal(),
                                    # activation=self.activation_name,
                                    # use_batchnorm=True,
                                    activation=None,
                                    use_batchnorm=False,
                                    name="Linear_Input",
                                    trainable=True
                                    ), False))
            self.exec_list.append((lambda x : tf.reshape(x, [tf.shape(x)[0], self.vertex_size, -1]),False))
            self.exec_list.append((tf.keras.layers.BatchNormalization(), False))
            self.exec_list.append((set_activation(self, self.activation_name), False))
            
            self.exec_list.append((tf.keras.layers.Reshape([input_shape[0], self.vertex_size, -1]), False))
            self.exec_list.append((lambda x : tf.reshape(x, [tf.shape(x)[0], self.vertex_size, -1]),False))

            self.exec_list.append((MeanPool(name="unpool"),False))
        for idx, output_size in enumerate(self.layer_info[1:-1]): 

                
            preset = dict()
            preset['translation_invariant'] = True
            preset['num_weight_matrices'] = self.kernel_size
            preset['num_output_channels'] = output_size
            preset['initializer'] = set_initializer(self.kernel_initializer)

            preset['name'] = "dec_" + str(idx)
            
            layer = tfg.FeatureSteeredConvolutionKerasLayer(**preset)
            
            self.exec_list.append((layer, True))
            if idx != len(self.layer_info[1:-1]) - 1 :
                self.exec_list.append((tf.keras.layers.BatchNormalization(), False))
            self.exec_list.append((set_activation(self, self.activation_name), False))

                                        

        self.exec_list.append((LinearLayer(

                                    output_shape= self.layer_info[-1], 
                                    kernel_initializer=set_initializer(self.kernel_initializer),
                                    # use_bias=False,
                                    use_bias = True,
                                    bias_initializer=tf.keras.initializers.TruncatedNormal(),
                                    activation=None,
                                    use_batchnorm=False,
                                    # activation=self.activation,
                                    name="Lenear_Output",
                                    trainable=True
                                    ), False))
                
        print("build complete Decoder")
        
    @tf.function
    @set_device("GPU:1")
    def call(self, inputs):
        x = inputs

        for layer, need_neighbor in (self.exec_list):
            if need_neighbor : 
                x = layer([x, self.sparse_A_tensor[0]])
            else : 
                x = layer(x)
            # print(layer.name,x.shape)

            
        return x 




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
        # if len(inputs.shape) == 2 : 
        #     inputs=tf.expand_dims(inputs, axis=1)
        # tf.print(inputs.shape)
        # print(inputs.shape)
        assert len(inputs.shape) == 3, "input_shape length must be 3-dimension. it is {}".format(len(inputs))

        result = tf.map_fn(lambda x: tf.matmul(x, self.W), inputs)
        if self.use_bias : 
            result += self.b 
        assert len(result.shape) == 3, "return value length must be 3-dimension. it is {}".format(len(inputs))
        
        if self.use_batchnorm : 
            result = self.batch_norm(result, training=self.trainable)

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
        # print("pooling", result.shape)
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
        # print("mean pool", result.shape)
        return result


