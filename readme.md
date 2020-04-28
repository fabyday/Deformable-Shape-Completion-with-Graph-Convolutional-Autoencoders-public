# Deformable Shape Completion with Graph Convolutional Autoencoders-public
##### it is based on Deformable Shape Completion with Graph Convolutional Autoencoders paper's implementation.
this is incomplete yet.



## Model Arch
model_exp2.py is actual model file.
#### called by 


##### main.py -> modelwrapper.py -> model_exp2.py (it include model and loss )



<pre>
Input = (batch_size = ?, vertice_size = 5023, Feature_input = 3)
    class model
        - encoder
            * LinearLayer(output=[ ?, 5023, 16 ])
            * batchNorm [ ](output=[ ?, 5023, 16 ])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 16 ])
    
            * GCONV1(output=[ ?, 5023, 32 ])
            * batchNorm [ ](output=[ ?, 5023, 32 ])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 32 ])
    
            * GCONV2(output=[ ?, 5023, 64 ])
            * batchNorm [ ](output=[ ?, 5023, 64 ])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 64 ])
    
            * GCONV3(output=[ ?, 5023, 96 ])
            * batchNorm [ ](output=[ ?, 5023, 96 ])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 96 ]) 
    
            * GCONV3(output=[ ?, 5023, 128])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 128 ])
            * Pooling[](output=[?, 1, 128])   

            * LinearLayer(output=[ ?, 1 , 128 ]) -> For z_log_var
            * LinearLayer(output=[ ?, 1 , 128 ]) -> For z_log_var   
        
        - decoder
            * LinearLayer(output=[ ?, 5023*128 ]) 
            * batchNorm [ ](output=[ ?, 5023, 96 ])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 128 ])
            * Reshape(output=[ ? 5023, 128 ])
          
            * GCONV1(output=[ ?, 5023, 128])
            * batchNorm [ ](output=[ ?, 5023, 128 ])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 96 ])            
          
            * GCONV2(output=[ ?, 5023, 96 ])
            * batchNorm [ ](output=[ ?, 5023, 96 ])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 96 ])
          
            * GCONV3(output=[ ?, 5023, 64 ])
            * batchNorm [ ](output=[ ?, 5023, 64 ])
            * activation[relu | tanh | leakyrelu](output=[ ?, 5023, 64 ])                        
          
            * GCONV4(output=[ ?, 5023, 32 ])
            * batchNorm [ ](output=[ ?, 5023, 32 ])
            * activation[ relu | tanh | leakyrelu ](output=[ ?, 5023, 32 ]) 
          
            * GCONV5(output=[ ?, 5023, 16 ])
            * batchNorm [ ](output=[ ?, 5023, 16 ])
            * activation[ relu | tanh | leakyrelu ](output=[ ?, 5023, 16 ])
          
            * LinearLayer(output=[ ?, 5023, 3 ])
</pre>



## Configurations

numpy   
tensorflow-gpu 2.0 or 2.1   
libigl for python.   
mesh lib(optional) : https://github.com/MPI-IS/mesh


just using requirements.txt








## data preprocessing.

###### 1. first download CoMA dataset files.(it's need login.)
###### 2. extract files. and remove readme in CoMA dataset. 
###### 3. make one more folder(I actually named ./unrpocessed_dataset), and put renamed CoMa folder( rename plain ) on.
###### 4. run fileconverter.py. before using fileconverter, and write your own data root path(for me. ./unprocessed_dataset) _target_input_dir variable in fileconverter.

## run model

<pre>
python main.py [ train | test | summary ]
</pre>





