import tensorflow as tf 
import numpy as np 
#                                    ___,,___
#                                 ,d8888888888b,_
#                             _,d889'        8888b,
#                         _,d8888'          8888888b,
#                     _,d8889'           888888888888b,_
#                 _,d8889'             888888889'688888, /b
#             _,d8889'               88888889'     `6888d 6,_
#          ,d88886'              _d888889'           ,8d  b888b,  d\
#        ,d889'888,             d8889'               8d   9888888Y  )
#      ,d889'   `88,          ,d88'                 d8    `,88aa88 9
#     d889'      `88,        ,88'                   `8b     )88a88'
#    d88'         `88       ,88                   88 `8b,_ d888888
#   d89            88,      88                  d888b  `88`_  8888
#   88             88b      88                 d888888 8: (6`) 88')
#   88             8888b,   88                d888aaa8888, `   'Y'
#   88b          ,888888888888                 `d88aa `88888b ,d8
#   `88b       ,88886 `88888888                 d88a  d8a88` `8/
#    `q8b    ,88'`888  `888'"`88          d8b  d8888,` 88/ 9)_6
#      88  ,88"   `88  88p    `88        d88888888888bd8( Z~/
#      88b 8p      88 68'      `88      88888888' `688889`
#      `88 8        `8 8,       `88    888 `8888,   `qp'
#        8 8,        `q 8b       `88  88"    `888b
#        q8 8b        "888        `8888'
#         "888                     `q88b
#                                   "888'
# BISON BISON BISON BISON

class FeastNet(tf.keras.layers.Layer):


    """
        ReImplemented FeastNet. its calculation Keras Layer based on 
        "Deformable Shape Completion with Graph Convolutional Autoencoders"
        
        I think tensorflow_graphics libary works wrong.
        original feast net use too many transpose. so I minimize transpose.
        
        
        CAUTION ! : Only support Translation Invariant. if you need translation Variant. implement yourself.
        TODO __init__ is_invariant argument is deprecated. but it's just interface for old code.
        TODO no bias exists. it is not necessary now. if you want, add weight in build and call function. :)
    """
    def __init__(self, output_channel,adj,  kernel_size=8, is_invariant=True, initializer= tf.keras.initializers.GlorotNormal,name="Layer", trainable=True):
        """
            output_channel : output shape
            adj : FeastNet Class assume that all meshes use same references face list. 
                  it means neighbor list is same for all mesh. it's shape must be [ vertex_size, maximum_neighbor_length ]
                  when calculate neighbor list. vertex number is start at 1. not 0. because this code understand 0 is NULL.
            kernel_size : kernel size is filter size.
            initializer : tf.keras.initializer class callable instance.
            name : name of layer.
            trainable : boolean trainable.

        """
        super(FeastNet, self).__init__(trainable, name)
        self.out_channel = output_channel
        self.adj  = adj
        self.kernel_size = kernel_size
        self.initializer = initializer
        

    def build(self, input_shape) : 
        """
            
        """
        self.batch_size, self.vertex_size, self.input_channel = input_shape

        #transform numpy array to tensor
        self.adj = tf.constant(self.adj, dtype=tf.int32)

        #add weight matrice :)
        self.u = self.add_weight(name="u", shape=[self.input_channel, self.kernel_size], initializer=self.initializer)
        self.W = self.add_weight(name="W", shape=[self.input_channel, self.kernel_size*self.out_channel], initializer=self.initializer) 


        # self.W_bias = self.add_weight(name="W_bias", shape=[self.out_channel], initializer=self.initializer)
        # self.u_bias = self.add_weight(name="u_bias", shape=[self.kernel_size], initializer=self.initializer)



    def get_patches(self, x):
        assert len(x.shape) == 3, "function get_patches get wrong shape inputs."

        def compute_nb_feature(input_f):
            return tf.gather(input_f, self.adj)
        batch_size = tf.shape(x)[0]
        _, vertex_size, input_channel = x.shape
        pad_input = tf.zeros([batch_size, 1, input_channel], dtype=tf.float32)
        x = tf.concat([pad_input, x], 1)


        #patches := [batch_size, vertex_size, neighbor_num, feature_channel=input_channel]
        patches = tf.map_fn(compute_nb_feature, x)

        assert len(patches.shape) == 4, "function get_patches get wrong output shape. it is not 4 dims"
        return patches

    def get_weight_invariant(self, x, patches):
        """
            (Xi-Xj)u 
        """
        def remove_disjoint_data(data):
            
            # return tf.where(tf.math.not_equal(tf.expand_dims(self.adj, -1), 0), data, tf.zeros_like(patches))
            
            result = tf.where(
                                tf.math.not_equal(tf.expand_dims(self.adj, axis=-1), 0), 
                                data,
                                tf.zeros_like(data)
                                )
            
            return result

        assert len(x.shape) == 3, "function get_weight_invariant get wrong shape in x."
        assert len(patches.shape) == 4, "function get_weight_invariant get wrong shape in patches."
        batch_size = tf.shape(x)[0]
        _, vertex_size, input_channel = x.shape
        _, _, neighbor_num, _ = patches.shape

        # x = tf.reshape(x, [batch_size, vertex_size, 1, input_channel])
        x = tf.reshape(x, [batch_size, vertex_size, 1, input_channel])
        # result := [ batch_size, vertex_size, neighbor_num, feature_channel=input_channel]
        result = tf.subtract(x, patches)
        
        #  remove neighbor value is 0
        result = tf.map_fn(remove_disjoint_data, result)

        result = tf.reshape(result, [batch_size, vertex_size*neighbor_num, input_channel])
        # result := [ batch_size, vertex_size, neighbor_num, kernel_size]
        result = tf.map_fn(lambda x : tf.matmul(x, self.u), result)
        result = result# + self.u_bias
        result = tf.nn.softmax(result)
        # result := [ batch_size, vertex_size, neighbor_num, kernel_size, 1]
        result = tf.reshape(result, [batch_size, vertex_size, neighbor_num, self.kernel_size, 1])
        
        #postcondition
        assert len(result.shape) == 5, "shape is not correct. :("

        return result 
        
    def call(self, x) : 

        #check precondition. all precondition check that shape is coooooooooooooorect. ahahahahah..
        assert len(x.shape) == 3, "function call get wrong shape"
        batch_size = tf.shape(x)[0]
        _, vertex_size, input_channel = x.shape
        assert vertex_size == self.vertex_size, "vertex_size is wrong."
        assert input_channel == self.input_channel, "vertex_size is wrong."

        # adj_size == 1/{N}
        adj_size = tf.math.count_nonzero(self.adj, -1)
        #deal with unconnected points: replace NaN with 0
        non_zeros = tf.math.not_equal(adj_size, 0)
        adj_size = tf.cast(adj_size, tf.float32)
        adj_size = tf.where(non_zeros,tf.math.reciprocal(adj_size),tf.zeros_like(adj_size))
        adj_size = tf.reshape(adj_size, [1, vertex_size, 1, 1])

        #ref_patches := [ batch_size, vertex_size, neighbor_num, feature_channel==input_channel]
        ref_patches = self.get_patches(x)
        _, _, neighbor_num, _ = ref_patches.shape
        
        #calc W * X{i,j}
        patches_main_feature = tf.reshape(ref_patches, [batch_size, self.vertex_size*neighbor_num, input_channel])
        patches_main_feature = tf.map_fn(lambda x : tf.matmul(x, self.W), patches_main_feature)
        patches_main_feature = tf.reshape(patches_main_feature, [batch_size, self.vertex_size, neighbor_num, self.kernel_size ,self.out_channel])


        #calc q(X{i}, X{j})
        # q := [batch_size, vertex_size, neighbor_num, kernel_size, 1]
        q = self.get_weight_invariant(x, ref_patches)

        #temporary just use patches_main_feature
        result = tf.math.multiply(patches_main_feature, q)


        #result := [batch_size, vertex_size, kernel_size, out_channel]
        result = tf.reduce_sum(result, axis=2)
        #multiply 1/{N}
        result = tf.math.multiply(result, adj_size)
        #result := [batch_size, vertex_size, out_channel]
        result = tf.reduce_sum(result, axis=2)

        assert len(result.shape) == 3, "result shape is wrong."
        return result# + self.W_bias



        
        

        





