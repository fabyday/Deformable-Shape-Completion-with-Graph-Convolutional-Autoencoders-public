import tensorflow as tf 
import GraphicCodeColection.loader as L
import numpy as np 
noise_type = "plain"
data_path = "./processed_dataset/plain/bareteeth"
test_size = 10
loader = L.Loader(common_load_dir_name=data_path, 
                noise_type= noise_type,
                test_size=test_size)
datadict = loader.get_train_data()
inputs = datadict['input']
labels = datadict['labels']    
inputs = loader.get_data_normalize(inputs)
class layer(tf.keras.layers.Layer):
    def __init__(self, name, output_chanel, activation):
        super(layer, self).__init__(name=name)
        self.output_chanel = output_chanel
        self.activation = activation
        self.act_layer = None
    
    def build(self, input_shape):
        batch_size, vertex_size, coord_size = input_shape

        self.W = self.add_weight(name = "W",
                                     shape=[coord_size, self.output_chanel],
                                     initializer = tf.keras.initializers.GlorotNormal
                                     )
        self.act_layer = tf.keras.layers.Activation(self.activation)

    def call(self, inputs):
        result = tf.map_fn(lambda x : tf.matmul(x, self.W), inputs)
        return result
        # if self.act_layer == None : 
        #     return result
        # return self.act_layer(result)

latent_dim = 32

x = tf.keras.Input((5023, 3), 2)
e1=layer('in', 16, 'relu')(x)
e2= layer('in2', 32, None)(e1)
e3 = tf.keras.layers.Flatten()(e2)
print("e3",e3.shape)
fc1=tf.keras.layers.Dense(latent_dim)(e3)
print("fc1",fc1.shape)

fc2= tf.keras.layers.Dense(5023*32)(fc1)
print("fc2",fc2.shape)
re = tf.keras.layers.Reshape([5023,32])(fc2)
print("re" ,re.shape)
d1=layer('out', 16, 'relu')(re)
print("d1",d1.shape)
d2=layer('out2', 3, None)(d1)
print("outd2",d2.shape)
encoder = tf.keras.Model(inputs=x, outputs=d2)




optimizer  = tf.keras.optimizers.Adam()

    
    

    
    

def loss(x, y ):
    return tf.reduce_mean(tf.math.sqrt(tf.math.reduce_sum( tf.math.pow(y-x, 2.0), axis=-1)), axis=-1)

def train(x, y ):
    with tf.GradientTape() as tape:
        result = encoder(x)
        loss_val = loss(result, y)
        gradients = tape.gradient(loss_val, encoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
        return result

def train_all(x, y ):
    batch_size =2 
    size = len(x)
    data = [0]*size
    for step, begin in enumerate(range(0, size, batch_size)):
        end = begin + batch_size
        end = min([size, end])
        print(step)
        result = train(x[begin:end], y[begin:end])
        data[begin:end]=result[:]
    print("convert")
    return np.array(data[-20:])

data = train_all(inputs, labels)
print("test over")
print(data[0].shape)

print("save pred")
loader.save_ply(encoder(inputs[90:100]).numpy(), name="pred", path="./conv_ply/testmoduleo")
print("save label")
loader.save_ply(labels[90:100], name="label", path="./conv_ply/testmoduleo")



        

 