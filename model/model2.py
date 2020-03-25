import tensorflow as tf
import numpy as np 
from .logger import * #set_device, time_set
from .utils import * 

"""
 no prob latent.
"""

device = "GPU:1"

@tf.function
def loss(label_y, pred_y, args):
    return tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(label_y-pred_y, 2.0), axis=-1)), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, 
                encoder_shape_list, 
                decoder_shape_list,
                adj,
                kernel_size,
                activation,
                use_latent,
                latent_size,
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
        
        
    @set_device(device)
    def build(self, input_shape): 
        batch_size, vertices_size, _ = input_shape
        self.encoder = Encoder(layer_info = self.encoder_shape_list, 
                                adj = self.adj, 
                                kernel_size= self.kerenl_size, 
                                activation= self.activation,
                                use_latent= self.use_latent,
                                latent_size = self.latent_size,
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
                                name = "decoder", 
                                trainable=True)


    @tf.function
    @set_device(device)
    def call(self, inputs): 
        outputs = self.encoder(inputs)
        return self.decoder(outputs)



class Encoder(tf.keras.Model):
    def __init__(self, 
                layer_info, 
                adj,
                kernel_size=9,
                activation = 'relu',
                use_latent = True,
                latent_size = 8,
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

    @set_device(device)
    def build(self, input_shape):

        self.exec_list.append(LinearLayer(  
                                        output_shape = self.layer_info[1],
                                        activation=None,
                                        # activation=self.activation,
                                        name = "C_LAYER",
                                        trainable=True
                                        ))
        # self.exec_list.append(tf.keras.layers.BatchNormalization())
        self.exec_list.append(self.activation_layer())

        # for idx in range(1, len(self.layer_info)-1):
        for idx in range(len(self.layer_info)-1):
            if idx == len(self.layer_info)-2 :
                activation = None 
            else : 
                activation = self.activation

            self.exec_list.append( NLayer(input_shape=self.layer_info[idx],
                                        output_shape=self.layer_info[idx+1],
                                        adj=self.adj,
                                        kernel_size=self.kernel_size,
                                        # activation=self.activation,
                                        activation= activation, 
                                        name=self.name[0]+"Layer"+str(idx),
                                        trainable=self.trainable
                                        )
                                )
            if not activation : # if activation None. Add ACTIVATION AND NORMALIZATION.
                # self.exec_list.append(tf.keras.layers.BatchNormalization())
                self.exec_list.append(self.activation_layer())

        
        
        if self.use_latent : 
            self.latent_list.append(tf.keras.layers.Dense(self.latent_size,
                                                        # activation=self.activation,
                                                        activation=None,
                                                        # use_bias = True,
                                                        kernel_initializer=tf.keras.initializers.GlorotNormal,
                                                        # bias_initializer=tf.keras.initializers.zeros
                                                        ))
           

    @tf.function
    @set_device(device)        
    def call(self, inputs):
        x = inputs
        # tf.print("input : \n{}\n".format(x))
        # tf.print("=======================")
        for layer in self.exec_list:
            x = layer(x)
            # tf.print("layer name : {}\n".format(layer.name))

            # tf.print("x : \n{}\n".format(x))
        # tf.print("=======================")
        if self.use_latent : 
            x = self.latent_list[0](x)
            # tf.print("x \n{}\n, z_mean \n{}\n, z_log_var \n{}\n ".format(x, z_mean, z_log_var))
            
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

    @set_device(device)
    def build(self, input_shape):
        if self.use_latent: 
            self.exec_list.append(tf.keras.layers.Dense(self.layer_info[0]*self.vertex_size, 
                                    # activation=self.activation,
                                    activation=None,
                                    # use_bias = True,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal
                                    # bias_initializer=tf.keras.initializers.zeros
                            
                                    ))
            # self.exec_list.append(tf.keras.layers.BatchNormalization())
            self.exec_list.append(self.activation_layer())


        # for idx in range(len(self.layer_info)-2):
        for idx in range(len(self.layer_info)-1):
            # if idx == len(self.layer_info)-3 : 
            if idx == len(self.layer_info)-3 : 
                activation = None 
            else : 
                activation = self.activation

            self.exec_list.append(NLayer(input_shape=self.layer_info[idx],
                                        output_shape=self.layer_info[idx+1],
                                        adj=self.adj,
                                        kernel_size=self.kernel_size,
                                        # activation=self.activation,
                                        activation=activation,
                                        name=self.name[0]+"Layer"+str(idx),
                                        trainable=self.trainable
                                        )
                                )
            if not activation : # if activation None. Add ACTIVATION AND NORMALIZATION.
                # self.exec_list.append(tf.keras.layers.BatchNormalization())
                self.exec_list.append(self.activation_layer())
            
        self.exec_list.append(LinearLayer(
                                output_shape = self.layer_info[-1],
                                activation=None,
                                name = "output",
                                trainable=True
                                ))
        
    @tf.function
    @set_device(device)
    def call(self, inputs):
        x = inputs
        first_event = True

        # tf.print("="*10)

        for layer in self.exec_list:
            x = layer(x)
            if first_event and self.use_latent: 
                first_event = False
                x = tf.reshape(x,[self.batch_size, self.vertex_size, -1])
            
            # tf.print("name layer : ", layer.name)
            # for i in layer.get_weights():
                # tf.print("weights : ", i)


        # tf.print("x : ", x)
        # tf.print("**"*10)

        return x 



class CLLayer(tf.keras.layers.Layer):
    def __init__(self, 
                input_shape,
                output_shape, 
                activation=None,
                name = "Layer_",
                trainable=True):
        super(CLLayer, self).__init__(trainable=trainable,name=name)
        self.input_channel = input_shape
        self.output_channel = output_shape
        self.activation = activation
        self.activation_func = set_activation(self, self.activation)

    @set_device(device)
    def build(self, inputs_shape): 
        _, self.vertice_size, self.Fin = inputs_shape

        self.W = self.add_weight (name = "W",
                                     shape=[self.vertice_size, self.Fin, self.output_channel],
                                     initializer = tf.keras.initializers.GlorotNormal
                                     )
    @tf.function
    @set_device(device)
    def call(self, inputs) : 
        # [batch_size, vertices_size, 1 , Fin]
        x = tf.reshape(inputs, [-1, self.vertice_size, 1, self.Fin])
        result = tf.einsum('baij, ajk->baik', x, self.W)
        result = result[..., 0, :]
        return result if self.activation_func == None else self.activation_func(result)

class NLayer(tf.keras.layers.Layer):
    def __init__(self, 
                input_shape,
                output_shape, 
                adj = None,
                kernel_size = 9,
                activation=tf.keras.activations.relu,
                name = "Layer_",
                trainable=True):
        super(NLayer, self).__init__(trainable=trainable,name=name)
        self.input_chanel = input_shape
        # self.output_shape = output_shape
        self.output_chanel = output_shape

        self.adj = adj
        self.kernel_size = kernel_size
        self.activation = activation
        self.activation_func = set_activation(self, self.activation)
    
    @set_device(device)                              
    def build(self, input_shape):
        batch_size, vertex_size, coord_size = input_shape

        self.adj = tf.constant(self.adj, dtype=tf.int32)
        self.W = self.add_weight (name = "W",
                                     shape=[coord_size, self.kernel_size,self.output_chanel],
                                     initializer = tf.keras.initializers.GlorotNormal
                                     )
                                     
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
        vx = tf.map_fn(lambda x: tf.matmul(x, self.u), x)

        
        # patch := [ batch_size, vertice_size, neighbor_num, kernel_size ]
        patches = self.neighbor(vx)
        # ux := [batch_size,vertice_size, 1, kernel_size]
        ux = tf.expand_dims(ux, 2)
        # patches = u(X_point + X_neighbor) + c 
        patches = tf.add(ux, patches)
        
        patches = tf.add(patches, self.c)
        
        
        patches = tf.nn.softmax(patches)
        return patches

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
        
        result = self.activation_func(patches) if self.activation_func != None else patches       
        
        return result 

        



class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, 
                output_shape, 
                activation=None,
                name = "Linear_Layer_",
                trainable=True):
        super(LinearLayer, self).__init__(trainable=trainable,name=name)
        self.output_channel = output_shape
        self.activation = activation
        self.activation_func = set_activation(self, self.activation)

    @set_device(device)
    def build(self, inputs_shape): 
        _, self.vertice_size, self.Fin = inputs_shape

        self.W = self.add_weight (name = "W",
                                     shape=[self.Fin, self.output_channel],
                                     initializer = tf.keras.initializers.GlorotNormal
                                     )
    @tf.function
    @set_device(device)
    def call(self, inputs) : 
        # [batch_size, vertices_size, 1 , Fin]
        return tf.map_fn(lambda x: tf.matmul(x, self.W), inputs)
