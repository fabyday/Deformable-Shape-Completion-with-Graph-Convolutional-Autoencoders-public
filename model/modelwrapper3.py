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

def reparameterize( mean_logvar):
  mean, logvar = mean_logvar
  eps = tf.random.normal(shape=tf.shape(mean))
  return eps * tf.exp(logvar * .5) + mean



def get_vae(adj):
  
  
  latent_dim = 64
  adj = adj
  kernel = 8
  initializer = tf.keras.initializers.GlorotNormal
  # self.adj = tf.constant(self.adj, dtype=tf.int32)

  # self.adj = np.expand_dims(self.adj, 0)
  # self.adj = np.concatenate([self.adj for _ in range(batch_size)], axis=0)
  with tf.device("/GPU:0"):

    encoder_input = tf.keras.Input(shape=(5023,3))
    x = tf.keras.layers.Dense(16, name="encoder_input", use_bias=False, kernel_initializer=initializer() )(encoder_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = fn.FeastNet( 32, adj = adj, kernel_size=kernel, is_invariant=False, initializer=initializer(), name="dGconv1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = fn.FeastNet( 64, adj = adj, kernel_size=kernel, is_invariant=False, initializer=initializer(), name="dGconv2")(x)
    # tf.keras.layers.BatchNormalization(),
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Lambda(lambda x : tf.math.reduce_mean(x, 1), name="pooling")(x)

    # # # # No activation
    z_mean = tf.keras.layers.Dense(latent_dim ,name="z_mean", use_bias=False, kernel_initializer=initializer())(x)
    z_log_var = tf.keras.layers.Dense(latent_dim ,name="z_log_var", use_bias=False, kernel_initializer=initializer())(x)
    
    z = tf.keras.layers.Lambda(reparameterize, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])


    inference_net = tf.keras.Model(encoder_input, [z_mean, z_log_var, z])
    inference_net.summary()

      
      


    input_decoder = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(units=5023* latent_dim ,name="decoder_input", use_bias=False, kernel_initializer=initializer())(input_decoder)
    x = tf.keras.layers.Reshape(target_shape=(5023, latent_dim))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = fn.FeastNet( 64,adj = adj, kernel_size=kernel,  is_invariant=False,  initializer=initializer(), name="dGconv6")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = fn.FeastNet( 32, adj = adj, kernel_size=kernel, is_invariant=False,  initializer=initializer(), name="dGconv7")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = fn.FeastNet(16, adj = adj, kernel_size=kernel,  is_invariant=False,  initializer=initializer(), name="dGconv8")(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dense(3, name="decoder_output", use_bias=False, kernel_initializer=initializer())(x)
      
    generative_net =  tf.keras.Model(input_decoder, x)
    generative_net.summary()


    output = generative_net(inference_net(encoder_input)[2])
    vae = tf.keras.Model(encoder_input, output)
    print(output.shape)
      
    # tf.print("whait is vae losses", self.losses)
    total_loss = tf.keras.losses.MSE(encoder_input, output)
    # total_loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(encoder_input-output, 2.0), axis=-1)), axis=-1)
    kl_loss = - 0.5* 10e-8 * tf.reduce_sum(
                                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, -1)
    kl_loss = tf.reshape(kl_loss, [-1, 1])
    vae_loss = tf.keras.backend.mean(kl_loss + total_loss)
    print(vae_loss.shape)
    vae.add_loss(vae_loss)

    
                              
    print(total_loss, "total loss is whats")
   

    
    # optimizer = tf.keras.optimizers.Adam(1e-4)
  with tf.device("/GPU:1"):

    vae.compile(optimizer="adam",metrics=['accuracy'])
    vae.summary()
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
  cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True,save_freq=5)

  with tf.device("/GPU:1"):
    vae.fit(x, epochs=epochs, batch_size = 2, callbacks=[cp_callback, tensorboard_callback])


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



