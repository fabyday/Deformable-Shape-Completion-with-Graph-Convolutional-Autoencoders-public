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
    self.acc = tf.keras.metrics.Accuracy()
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
    # kl_loss = - 0.5 * tf.reduce_sum(
    #                             var - tf.square(mean) - tf.exp(var) + 1, -1)              
                    
    # kl_loss *= 10e-8
    # tf.print("="*10, kl_loss)
    # tf.print("kl_loss :\n", kl_loss)
    # tf.print("mean :\n", mean)
    # tf.print("var : \n", var)
    # tf.print("z : \n", z)
    # tf.print("*="*5)
    # self.add_loss(kl_loss)
    # self.add_loss(10e-8*kl_loss)

    with tf.device("/GPU:0"):
      out =  self.decode(z)
      self.acc.reset_states()
      # kl_loss = - 0.5 * tf.reduce_sum(1 + var - tf.square(mean) - tf.exp(var), -1)                                
      # kl_loss = kl_loss
      
      # self.add_loss(kl_loss)
    # tf.print("whait is vae losses", self.losses)
    # total_loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(inputs-out, 2.0), axis=-1)), axis=-1)
    # print(total_loss, "total loss is whats")
      
      # total_loss = tf.keras.losses.MSE(inputs, out)
      total_loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(out-inputs, 2.0), axis=-1)), axis=-1)
      kl_loss = - 0.5 * tf.reduce_sum(
                                var - tf.square(mean) - tf.exp(var) + 1, -1)
      kl_loss = 10e-8 * kl_loss
      kl_loss = tf.reshape(kl_loss, [-1, 1])
      vae_loss = tf.keras.backend.mean(kl_loss + total_loss, -1)
      vae_loss = tf.keras.backend.mean(kl_loss + total_loss)
      self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
      self.add_metric(total_loss, name='mse_loss', aggregation='mean')
      self.acc.update_state(out, inputs)
      # self.acc.update_state(out, inputs)
      # self.add_metric(self.acc.update_state(out, inputs).result().numpy(), name="acc",  aggregation='mean')
      self.add_loss(vae_loss)
    return out


def get_vae(adj):
  
  vae = CVAE(batch_size = 2,adj = adj)
  with tf.device("/GPU:1"):
    optimizer = tf.keras.optimizers.Adam(10e-4)
    # optimizer = tf.keras.optimizers.Adam(1e-4)

    # vae.compile(optimizer, loss =tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    vae.compile(optimizer, metrics=['accuracy'])
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
    # vae.fit(x, x, epochs=epochs, batch_size = 2, callbacks=[cp_callback, tensorboard_callback])
    vae.fit( x, epochs=epochs, batch_size = 2, callbacks=[cp_callback, tensorboard_callback])



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

def phase2_test(vae, x_inputs, x_label, name, batch_size=1):

      
  checkpoint_path = os.path.join("./checkpoints", name, "cp-{epoch:04d}.ckpt")
  checkpoint_dir = os.path.dirname(checkpoint_path)

  latest = tf.train.latest_checkpoint(checkpoint_dir)
  print("load weight... : ", latest)

  vae.load_weights(latest)
  size=len(x_inputs)
  iterations = 200
  decoder = vae.generative_net
  optimizer = tf.keras.optimizers.SGD(0.6)
  data_list = [0]*size
  print("data size : ", size)
  print("batch_size : ", batch_size)
  latent = tf.Variable(tf.random.normal([batch_size, 64]), dtype=tf.float32, trainable=True)

  bools = np.not_equal(np.sum(x_inputs, axis=-1), 3.)
  x_label = np.expand_dims(x_label, 0)
  x_label = np.concatenate([x_label for _ in range(batch_size)], axis=0)
  print(x_label.shape, "what is x_labels")

  for begin in range(0, size, batch_size):
    end = begin + batch_size
    end = min([size, end])
    x_label = x_label.astype(np.float32)
    print("input shape : ", x_inputs[begin:end].shape)
    result = inner_train(x_inputs[begin:end], x_label, bools[begin:end],latent, vae, optimizer, iterations)
    
    data_list[begin:end] = result.numpy()[:]

  return np.array(data_list)


# @tf.function
def inner_train(x_inputs, x_label, bools,latent, vae, optimizer, iterations = 10000) : 

  mean, var = vae.encode(x_label)
  # mean, var = vae.encode(x_inputs)

  latent.assign(vae.reparameterize(mean, var))
  # latent.assign(tf.random.normal([1, 64]))
  with tf.device("/GPU:1"):
    x_inputs = tf.boolean_mask(x_inputs, bools)
    # for i in range(iterations):
    loss = 100
    while loss > 0.0008  : 

      with tf.GradientTape() as tape:


        result = vae.decode(latent)
        result = tf.boolean_mask(result, bools) 
        # result = tf.where(tf.expand_dims(bools, -1),tf.ones_like(result), result)     
        # loss = tf.reduce_mean(tf.math.reduce_sum( result-x_inputs, axis=-1), axis=-1)
        loss = tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(result-x_inputs, 2.0), axis=-1)), axis=-1)
        loss = tf.reduce_mean(loss)
      
        print("iter : "," loss : ", loss,end="\r")
        gradient = tape.gradient(loss, latent)
        # print(gradient,"\n", latent)
        # optimizer.apply_gradients(zip([gradient], [latent]))
        optimizer.apply_gradients(zip([gradient],[latent]))
  return vae.decode(latent)


