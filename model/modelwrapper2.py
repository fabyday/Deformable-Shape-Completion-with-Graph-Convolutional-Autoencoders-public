import tensorflow as tf
from . import model_exp as model 
# import tensorflow_graphics as tfg
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# tf.debugging.set_log_device_placement(True)
import numpy as np 
from . import feastnet as fn

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

# NOTIFICATION
# IT IS PREDICT CORRECT WHEN EPOC WAS 22 AT LEAST. LOSS MUST BE LESS THAN 10E-7.
#
#
#
#
#




session = tf.compat.v1.Session(config=config)
class CVAE(tf.keras.Model):
  def __init__(self,batch_size, adj, initializer=tf.keras.initializers.GlorotNormal):
    super(CVAE, self).__init__()
    self.latent_dim = 64
    self.adj = adj
    self.kernel = 8
    self.initializer = initializer
    # self.adj = tf.constant(self.adj, dtype=tf.int32)

    # self.adj = np.expand_dims(self.adj, 0)
    # self.adj = np.concatenate([self.adj for _ in range(batch_size)], axis=0)
    with tf.device("/GPU:0"):
      self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(5023,3)),
          tf.keras.layers.Dense(16, name="encoder_input", use_bias=False, kernel_initializer=self.initializer() ),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),

          fn.FeastNet( 32, adj = self.adj, kernel_size=self.kernel, is_invariant=False, initializer=self.initializer(), name="dGconv1"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),

          fn.FeastNet( 64, adj = self.adj, kernel_size=self.kernel, is_invariant=False, initializer=self.initializer(), name="dGconv2"),
          # tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),


          # fn.FeastNet( 96, adj = self.adj, kernel_size=self.kernel, is_invariant=False,  name="dGconv3"),
          # # tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.ReLU(),

          # fn.FeastNet( 128, adj = self.adj, kernel_size=self.kernel, is_invariant=False,  name="dGconvlast"),
          # # tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.ReLU(),          
          # fn.FeastNet( 16, adj = self.adj, kernel_size=self.kernel, is_invariant=False,  name="dGconv3"),
          # # tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.ReLU(),          



          # fn.FeastNet( 96, adj = self.adj, kernel_size=self.kernel, is_invariant=False,  name="dGconv3"),
          # tf.keras.layers.ReLU(),

          # tf.keras.layers.Flatten(),
          tf.keras.layers.Lambda(lambda x : tf.math.reduce_mean(x, 1)),
          # # # # No activation
          tf.keras.layers.Dense(self.latent_dim+self.latent_dim ,name="encoder_output", use_bias=False, kernel_initializer=self.initializer())#+ self.latent_dim ,name="encoder_output", use_bias=False)# ),
      ]
    )
    with tf.device("/GPU:0"):
      self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
          tf.keras.layers.Dense(units=5023* self.latent_dim ,name="decoder_input", use_bias=False, kernel_initializer=self.initializer()),#*self.latent_dim),
          tf.keras.layers.Reshape(target_shape=(5023, self.latent_dim)),# self.latent_dim)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),
          
          # fn.FeastNet( 128, adj = self.adj, kernel_size=self.kernel,  is_invariant=False,  name="dGconv4"),
          # tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.ReLU(),
          
          
          # fn.FeastNet( 96, adj = self.adj, kernel_size=self.kernel,  is_invariant=False,  name="dGconv5"),
          # tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.ReLU(),

          fn.FeastNet( 64,adj = self.adj, kernel_size=self.kernel,  is_invariant=False,  initializer=self.initializer(), name="dGconv6"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),

          fn.FeastNet( 32, adj = self.adj, kernel_size=self.kernel, is_invariant=False,  initializer=self.initializer(), name="dGconv7"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),

          fn.FeastNet(16, adj = self.adj, kernel_size=self.kernel,  is_invariant=False,  initializer=self.initializer(), name="dGconv8"),
          tf.keras.layers.ReLU(),

          tf.keras.layers.Dense(3, name="decoder_output", use_bias=False, kernel_initializer=self.initializer())
        ]
          

        
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar
    # return self.inference_net(x)

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=tf.shape(mean))
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)


    return logits
  
  def call(self, inputs):

    with tf.device("/GPU:0"):

      mean, var = self.encode(inputs)
      z = self.reparameterize(mean, var)
      # z = self.encode(inputs)
      


      # z = self.encode(inputs)
    # print(z.shape, "what is z shape")
    kl_loss = - 0.5 * tf.reduce_mean(
                                var - tf.square(mean) - tf.exp(var) + 1, -1)                      
    # kl_loss *= 10e-8
    # tf.print("="*10, kl_loss)
    # tf.print("kl_loss :\n", kl_loss)
    # tf.print("mean :\n", mean)
    # tf.print("var : \n", var)
    # tf.print("z : \n", z)
    # tf.print("*="*5)
    self.add_loss(kl_loss)
    with tf.device("/GPU:0"):
      out =  self.decode(z)
      
      # kl_loss = - 0.5 * tf.reduce_sum(1 + var - tf.square(mean) - tf.exp(var), -1)                                
      # kl_loss = kl_loss
      
      # self.add_loss(kl_loss)
      # self.add_loss(10e+4)
    # tf.print("whait is vae losses", self.losses)
    return out


def get_vae(adj):
  
  vae = CVAE(batch_size = 2,adj = adj)
  with tf.device("/GPU:1"):
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    vae.compile(optimizer, loss =tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

  # vae.compile(optimizer, metrics=['accuracy'])
  return vae

import os
import glob
import datetime

def fit(vae, x, epochs, name):

  checkpoint_path = os.path.join("./checkpoints", name, "cp-{epoch:04d}.ckpt")
  checkpoint_dir = os.path.dirname(checkpoint_path)
    

  
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  logdir=os.path.join("summaries", name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  if not os.path.exists(os.path.dirname(logdir)):
    os.makedirs(os.path.dirname(logdir))


  latest = tf.train.latest_checkpoint(checkpoint_dir)

  if latest : 
    print("load weight...")
    vae.load_weights(latest)

  print(vae.losses, "what is vae losses")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
  with tf.device("/GPU:1"):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True,save_freq=5)
    vae.fit(x, x, epochs=epochs, batch_size = 2, callbacks=[cp_callback, tensorboard_callback])


def pred(vae, x, name) : 
  
  checkpoint_path = os.path.join("./checkpoints", name, "cp-{epoch:04d}.ckpt")
  checkpoint_dir = os.path.dirname(checkpoint_path)

  latest = tf.train.latest_checkpoint(checkpoint_dir)

  vae.load_weights(latest)
  return vae.predict(x, batch_size = 2)

def summary(vae):
  vae.inference_net.summary()
  vae.generative_net.summary()
  vae.summary()






class FeastLayer(tf.keras.layers.Layer):

  def __init__(self, 
          input_shape,
          output_shape, 
          adj = None,
          kernel_size = 9,
          activation=tf.keras.activations.relu,
          is_invariant = False,
          name = "Layer_",
          trainable=True):
    super(FeastLayer, self).__init__(trainable=trainable,name=name)
    self.output_chanel = output_shape

    self.adj = adj
    self.kernel_size = kernel_size
    self.activation_name= activation
    self.is_invariant = is_invariant


  def build(self, input_shape):
    batch_size, self.vertex_size, self.coord_size = input_shape

    M = self.kernel_size
    out_channels = self.output_chanel
    in_channels = self.coord_size
    
    self.adj = tf.constant(self.adj, dtype=tf.int32)
    self.W = self.add_weight(name = "W", shape=[M, out_channels, in_channels], )
    self.b = self.add_weight(name = "b", shape=[out_channels])
    self.u = self.add_weight(name = "u", shape=[M, in_channels])
    if not self.is_invariant : 
      self.v = self.add_weight(name = "v", shape=[M, in_channels])
    self.c = self.add_weight(name = "c", shape=[M])

  def call(self, x):
    #preprocess for input
    batch_size = tf.shape(x)[0]
    input_size, in_channels = self.vertex_size, self.coord_size
    adj = self.adj
    K = tf.shape(self.adj)[-1]
    W = self.W
    u = self.u 
    c = self.c
    b = self.b
    if not self.is_invariant : 
      v = self.v


    #preprocess for output
    M = self.kernel_size
    out_channels = self.output_chanel



    
    # Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
    adj_size = tf.math.count_nonzero(adj, 2)
    #deal with unconnected points: replace NaN with 0
    non_zeros = tf.math.not_equal(adj_size, 0)
    adj_size = tf.cast(adj_size, tf.float32)
    adj_size = tf.where(non_zeros,tf.math.reciprocal(adj_size),tf.zeros_like(adj_size))
    # [batch_size, input_size, 1, 1]
    adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
    #[1, input_size, 1, 1]
    # [batch_size, input_size, K, M]
    if not self.is_invariant : 
      q = self.get_weight_assigments(x, adj, u,v, c)
    else :
      q = self.get_weight_assigments_translation_invariance(x, adj, u, c)

    # [batch_size, in_channels, input_size]
    x = tf.transpose(x, [0, 2, 1])
    W = tf.reshape(W, [M*out_channels, in_channels])
    # Multiple w and x -> [batch_size, M*out_channels, input_size]
    wx = tf.map_fn(lambda x: tf.matmul(W, x), x)
    # Reshape and transpose wx into [batch_size, input_size, M*out_channels]
    wx = tf.transpose(wx, [0, 2, 1])
    # Get patches from wx - [batch_size, input_size, K(neighbours-here input_size), M*out_channels]
    patches = self.get_patches(wx, adj)
    # [batch_size, input_size, K, M]
    #q = get_weight_assigments_translation_invariance(x, adj, u, c)
    # Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
    patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
    # [out, batch_size, input_size, K, M]
    patches = tf.transpose(patches, [4, 0, 1, 2, 3])
    patches = tf.multiply(q, patches)
    print(patches.shape, q.shape)
    patches = tf.transpose(patches, [1, 2, 3, 4, 0])
    # Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
    patches = tf.reduce_sum(patches, axis=2)
    patches = tf.multiply(adj_size, patches)
    # Add add elements for all m
    patches = tf.reduce_sum(patches, axis=2)
    # [batch_size, input_size, out]
    patches = patches + b
    return patches

  def get_weight_assigments(self, x, adj, u, v, c):

    # [batch_size, M, N]
    x = tf.transpose(x, [0, 2, 1])

    ux = tf.map_fn(lambda x: tf.matmul(u, x), x)
    vx = tf.map_fn(lambda x: tf.matmul(v, x), x)
    # [batch_size, N, M]
    vx = tf.transpose(vx, [0, 2, 1])
    # [batch_size, N, K, M]
    patches = self.get_patches(vx, adj)
    
    ux = tf.transpose(ux, [0,2,1])
    ux = tf.expand_dims(ux, 2)
    patches = tf.where(tf.math.not_equal(tf.expand_dims(self.adj, -1), 0), tf.add(ux, patches), tf.zeros_like(patches))

    # [K, batch_size, M, N]
    patches = tf.transpose(patches, [2, 0, 3, 1])
    # [K, batch_size, M, N]
    # patches = tf.add(ux, patches)
    # [K, batch_size, N, M]
    patches = tf.transpose(patches, [0, 1, 3, 2])
    patches = tf.add(patches, c)
    # [batch_size, N, K, M]
    patches = tf.transpose(patches, [1, 2, 0, 3])
    patches = tf.nn.softmax(patches)
    return patches

  def get_weight_assigments_translation_invariance(self, x, adj, u, c):
    #preprocess for weight and shape
    batch_size = tf.shape(x)[0]
    num_points = self.vertex_size
    in_channels = self.coord_size
    K = tf.shape(adj)[-1]
    M = self.kernel_size


    # [batch, N, K, ch]
    patches = self.get_patches(x, adj)
    # [batch, N, ch, 1]
    # x = tf.reshape(x, [-1, num_points, in_channels, 1])
    x = tf.reshape(x, [-1, num_points, 1, in_channels])
    patches = tf.where(tf.math.not_equal(tf.expand_dims(self.adj, -1), 0), tf.subtract(x, patches), tf.zeros_like(patches))
    
    print("shape is what is shape,", x.shape, patches.shape)
    
    # [batch, N, ch, K]
    patches = tf.transpose(patches, [0, 1, 3, 2])


    

    # [batch, N, ch, K]
    # patches = tf.subtract(x, patches)
    # [batch, ch, N, K]
    patches = tf.transpose(patches, [0, 2, 1, 3])
    # [batch, ch, N*K]
    x_patches = tf.reshape(patches, [-1, in_channels, num_points*K])
    # batch, M, N*K
    patches = tf.map_fn(lambda x: tf.matmul(u, x) , x_patches)
    # batch, M, N, K
    patches = tf.reshape(patches, [-1, M, num_points, K])
    # [batch, K, N, M]
    patches = tf.transpose(patches, [0, 3, 2, 1])
    # [batch, K, N, M]
    patches = tf.add(patches, c)
    # batch, N, K, M
    patches = tf.transpose(patches, [0, 2, 1, 3])
    patches = tf.nn.softmax(patches)
    return patches

  def get_slices(self, x, adj):
    batch_size = tf.shape(x)[0]

    num_points = self.vertex_size
    in_channels =x.shape [-1]

    print("whait is in channels", in_channels)
    K = tf.shape(adj)[-1]
    zeros = tf.zeros([batch_size, 1, in_channels], dtype=tf.float32)
    x = tf.concat([zeros, x], 1)
    x = tf.reshape(x, [batch_size*(num_points+1), in_channels])
    adj = tf.reshape(adj, [batch_size*num_points*K])
    adj_flat = self.tile_repeat(batch_size, num_points*K)
    adj_flat = adj_flat*(num_points+1)
    adj_flat = adj_flat + adj
    adj_flat = tf.reshape(adj_flat, [batch_size*num_points, K])
    slices = tf.gather(x, adj_flat)
    slices = tf.reshape(slices, [batch_size, num_points, K, in_channels])
    return slices

  def get_patches(self, x, adj):
    patches = self.get_slices(x, adj)
    return patches

  def tile_repeat(self, n, repTime):
    '''
    create something like 111..122..2333..33 ..... n..nn 
    one particular number appears repTime consecutively.
    This is for flattening the indices.
    '''
    idx = tf.range(n)
    idx = tf.reshape(idx, [-1, 1])    # Convert to a n x 1 matrix.
    idx = tf.tile(idx, [1, repTime])  # Create multiple columns, each column has one number repeats repTime 
    y = tf.reshape(idx, [-1])
    return y


