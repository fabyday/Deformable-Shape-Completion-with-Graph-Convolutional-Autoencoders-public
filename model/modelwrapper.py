# ModelWrapper Module  wraps small models. 
# ModelWrapper give us many advantages.
# 1. change model quickly. (module must be implement loss function and Model class).
# 2. make new model based on trained model. ( like a phase1 model -> phase 2 model)














import tensorflow as tf
import numpy as np 
from .logger import * #set_device, time_set
import os 
from .model_exp import Model, loss
# from .model_exp import Model, loss
import copy 
from .utils import *

# global setting



#if you want to run eagerly. use tf.config.experimental_run_functions_eagerly(True)
# tf.config.experimental_run_functions_eagerly(True)
Model = Model
model_loss = loss

class ModelWrapper:
    def __init__(self, 
                name,
                random_seed, 
                batch_size, 
                kernel_size,
                num_epoch,
                use_latent,
                latent_size,
                F_0, 
                F,
                adj,
                activation,
                face,
                ds_D,
                ds_U,
                A,
                checkpoint_save_path,
                tensorboard_path,
                ):

        self.name = name
        self.random_seed =random_seed
        self.batch_size = batch_size
        self.kerenl_size = kernel_size
        self.use_latent = use_latent
        self.latent_size = latent_size
        self.F = F
        self.F_0 = F_0
        self.adj = adj
        self.ds_D = ds_D
        self.ds_U = ds_U
        self.ds_U.reverse()
        self.A = A
        self.activation = activation
        self.checkpoint_save_path = checkpoint_save_path
        self.tensorboard_path = tensorboard_path 
        self.num_epochs = num_epoch
        self.face=face
        

        self.optimizer = None
        self.loss_func = None 
        self.train_summary_writer = None

        #configure
        self._basic_preconfigure()
        self._custom_preconfigure()
        self._build()
        self._basic_postconfigure()
        self._custom_postconfigure()

    # ==========CONFIGURE==================
    def _basic_preconfigure(self):
        #SEED setting.
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)


        # GPU MEMORY GROWTH SET.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # tf.debugging.set_log_device_placement(True)


        session = tf.compat.v1.Session(config=config)

    def _custom_preconfigure(self): 
        self.optimizer  = tf.keras.optimizers.Adam()
        self.loss_func = model_loss
        
        self.encoder_shape_list = [self.F_0] + self.F # [Fin] + [Filter size list]
        
        self.decoder_shape_list = copy.deepcopy(self.encoder_shape_list)
        self.decoder_shape_list.reverse() # decoder shape list := REVERSED [filter size list] + [Fin]

    @set_device("GPU:1")
    def _build(self):
        #Model is only one.

        self.model = Model(encoder_shape_list = self.encoder_shape_list,
                            decoder_shape_list = self.decoder_shape_list, 
                            adj = self.adj,
                            kernel_size = self.kerenl_size,
                            activation = self.activation,
                            use_latent = self.use_latent,
                            latent_size = self.latent_size,
                            face = self.face,
                            ds_D = self.ds_D,
                            ds_U = self.ds_U,
                            A = self.A,
                            name = "Model",
                            trainable = True
                            )

        

    def _basic_postconfigure(self):
        self.ckpt, self.manager = self.load_ckpt_manager() #load from checkpoint
        self.train_summary_writer = self.get_create_summary_writer() #get summary writer
    

    def _custom_postconfigure(self):
        pass
    # -------------------------------------------------------------------------------------
    # =============LOSS & train_batch CONFIGURE ===========================================

    @tf.function
    def train_batch(self, inputs, labels):
        with tf.GradientTape() as tape:
            result = self.model(inputs)
            if type(result) == tuple : 
                result, latent_z, z_mean, z_log_var = result
                loss = self.loss_func(y_label = labels, y_pred = result, args=[latent_z, z_mean, z_log_var])
            else:
                loss = self.loss_func(y_label = result, y_pred = labels)
            gradient = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return loss


    # -------------------------------------------------------------------------------------
    # ==================TRAIN & PREDICTION=================================================
    
    def train(self, inputs, labels): 
        if len(inputs.shape) != 3 :
            raise Exception("demension isn't matched. [batch_size, vertice_size, point_size")

        size = inputs.shape[0]
        self.restore_checkpoint()

        #save model graph (TODO but it's not work now. can't trace graph.)
        tf.summary.trace_on(graph=True)
        self.train_batch(inputs= inputs[:self.batch_size], labels=labels[:self.batch_size])

        with self.train_summary_writer.as_default():
            tf.summary.trace_export(name="phase1_graph", step=0)
        
        save_path = self.manager.latest_checkpoint
        

        for epoch in range(self.num_epochs):

            losses = 0
            for step, begin in enumerate(range(0, size, self.batch_size)):

                self.ckpt.step.assign_add(1)
                end = begin + self.batch_size
                end = min([size, end])

                loss = self.train_batch(inputs=inputs[begin:end], labels=labels[begin:end])    
                losses += loss 
                if step % 5 == 0 : 
                    print_process(epoch, self.num_epochs, (step+1)*self.batch_size, size, loss, save_path)

                if int(self.ckpt.step) % 100 == 0 :
                    save_path = self.manager.save()
            
            # with self.train_summary_writer.as_default():
            #     tf.summary.scalar("loss", losses/(step+1), step=epoch)

        return loss    


    def predict(self, inputs, labels, batch_size): 
        """
            phase 1 Prediction.
        """
        self.restore_checkpoint()
        
        size=len(inputs)
        pred=[0]*size
        losses =0

        #pre-processing
        begin=0
        end = begin + batch_size
        end = min([size, end])
        self.model.trainable = False

        for step, begin in enumerate(range(0, size, batch_size)):
            end = begin + batch_size
            end = min([size, end])
            output = self.model(inputs[begin:end])
            
            if type(output) == tuple : 
                output ,_ , _, _ = output
            
            loss = self.loss_func(output, labels[begin:end])
            pred[begin:end] = output.numpy()[:]
            losses+=loss

        
        pred=np.array(pred)

        
        return (pred, (losses/batch_size/(step+1)).numpy())


    def predict2(self,inputs, labels, batch_size=1, iterations = 100):
        """
        phase 2 prediction. it just use decoder. 
        """
        self.restore_checkpoint()
        
        size=len(inputs)

        
        
        #pre-inferences.
        self.model.trainable = False 
        encoder = self.model.encoder
        decoder = self.model.decoder
        latent_z = tf.Variable(trianable=True, shape=[self.batch_size, self.latent_size], name = "trainable_latent_z")


        for step, begin in enumerate(range(0, size, batch_size)):
            end = begin + batch_size
            end = min([size, end])
            output = encoder(inputs[begin:end])
            latent_z.assign(output)

            for _ in range(iterations):
                with tf.GradientTape() as tape:
                    y_pred = decoder(latent_z)
                    loss = self.loss_func(y_pred, labels[begin:end])                    
                    gradient = tape.gradient(loss, latent_z)
                    self.optimizer.apply_gradients(zip(gradient, latent_z))
            

        
    # -------------------------------------------------------------------------------------
    # =======================================SAVE==========================================
    


    def load_ckpt_manager(self):
        if not os.path.exists(self.checkpoint_save_path+"/"):
            os.makedirs(self.checkpoint_save_path)
        ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer=self.optimizer, net=self.model)
        manager = tf.train.CheckpointManager(ckpt, self.checkpoint_save_path, max_to_keep=10)
        return ckpt, manager
    
    def restore_checkpoint(self):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint : 
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else : 
            print("Initializing from scratch")

    def get_create_summary_writer(self):
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)         
        summary_writer = tf.summary.create_file_writer(self.tensorboard_path)
        print("tensorboard : , ", self.tensorboard_path)
        return summary_writer
    
    
    # -------------------------------------------------------------------------------------
    # ======================================Etc.===========================================
    def summary(self):
        self.model.summary()
        self.model.encoder.summary()
        self.model.decoder.summary()

    def see_all_values(self):
        encoder = self.model.encoder
        decoder = self.model.decoder 

        print_summary_detail(encoder)
        print_summary_detail(decoder)

        draw_weight(self.model)