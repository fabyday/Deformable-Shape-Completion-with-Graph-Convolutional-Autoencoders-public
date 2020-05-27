import tensorflow as tf
from . import model_exp as model 
import tensorflow_graphics as tfg
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np 
import scipy.sparse as sp
# tf.debugging.set_log_device_placement(True)
#if you want to run eagerly. use tf.config.experimental_run_functions_eagerly(True)
# tf.config.experimental_run_functions_eagerly(True)
session = tf.compat.v1.Session(config=config)
class CVAE(tf.keras.Model):
  def __init__(self, adj, batch_size):
    super(CVAE, self).__init__()
    self.latent_dim = 32
    # self.adj = adj
    self.kernel = 7
    
    self.adj  = sp.csr_matrix(adj)
    coo = self.adj.tocoo()
    # print(coo)
    indices = np.mat([coo.row, coo.col]).transpose()
    sparse_A_tensor = tf.sparse.SparseTensor(indices, coo.data, coo.shape)
    sparse_A_tensor = tf.sparse.expand_dims(sparse_A_tensor, axis=0)
    
    # print(sparse_A_tensor.shape)
    self.adj = ( tf.sparse.concat( 0, [sparse_A_tensor for _ in range(batch_size)] ) )
  
    e_inputs = tf.keras.layers.Input((5023,3), batch_size)
    e1 = tf.keras.layers.Dense(16, name="linear_input")(e_inputs)
    e2 = tf.keras.layers.BatchNormalization()(e1)
    e3 = tf.keras.layers.ReLU()(e2)

    e4 = tfg.nn.layer.graph_convolution.FeatureSteeredConvolutionKerasLayer(True, 8, 32)(e3, self.adj)
    e5 = tf.keras.layers.BatchNormalization()(e4)
    e6 = tf.keras.layers.ReLU()(e5)
    
    e7 = tf.keras.layers.Lambda(lambda x : tf.math.reduce_mean(x, 1))(e6)
    # # # No activation
    e8 = tf.keras.layers.Dense(self.latent_dim+ self.latent_dim)(e7)
    self.inference_net = tf.keras.models.Model(inputs=e_inputs, outputs=e8)


    d_inputs = tf.keras.layers.Input((self.latent_dim,), batch_size)
    d2 = tf.keras.layers.Dense(units=5023* self.latent_dim )(d_inputs)
    # model.Uppool("up_pool"),
    d3 = tf.keras.layers.Reshape(target_shape=(5023, self.latent_dim))(d2)
    # tf.keras.layers.Reshape(target_shape=(1, self.latent_dim)),# self.latent_dim)),
    d4 = tf.keras.layers.BatchNormalization()(d3)
    d5 = tf.keras.layers.ReLU()(d4)
            
    # model.NLayer(None, 96, name="dGconv1", adj = self.adj, kernel_size=self.kernel),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.ReLU(),
    
    # model.NLayer(None, 64, name="dGconv2", adj = self.adj, kernel_size=self.kernel),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.ReLU(),

    d6 = tfg.nn.layer.graph_convolution.FeatureSteeredConvolutionKerasLayer(True, 8, 32)(d5, self.adj)
    d7 = tf.keras.layers.BatchNormalization()(d6)
    d8 = tf.keras.layers.ReLU()(d7)

    d9 = tfg.nn.layer.graph_convolution.FeatureSteeredConvolutionKerasLayer(True, 8, 16)(d8, self.adj)
    d10 = tf.keras.layers.ReLU()(d9)

    d11 = tf.keras.layers.Dense(3, name="linear_output")(d10)
    self.generative_net = tf.keras.models.Model(inputs=d_inputs, outputs=d11)





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
    # print(z.shape, "what is z shape")
    kl_loss = - 0.5 * tf.reduce_mean(
                                var - tf.square(mean) - tf.exp(var) + 1, -1)
    kl_loss = kl_loss
    self.add_loss(kl_loss)
    with tf.device("/GPU:1"):

      return self.decode(z)




def get_vae(adj,batch_size):
  vae = CVAE(adj,batch_size)
  optimizer = tf.keras.optimizers.Adam(1e-4)
  
  vae.compile(optimizer, loss =tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
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


  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

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