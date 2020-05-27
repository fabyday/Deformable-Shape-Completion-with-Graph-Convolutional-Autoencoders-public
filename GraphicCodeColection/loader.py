import numpy as np 
import time 
import os
import copy
from .mesh import Mesh
import copy



from .fileprocess  import * 

from .serialize import * 

class Loader(object):
    """
        Class Loader :
            Custom Loader Class manage train/test dataset.

            TODO issue
            hole noise data is difficult to restore data to initial data.
            Instead, use _nn postfix.
            ex ) self.X_noise_train_nn


    """
    #private static Variables
    __reference_mesh_file = 'data/template.obj'
    __original_dirname = "plain"
    __data_path = "./processed_dataset"
    __save_path ='./conv_ply'

    __plain_dirname  = "plain"
    __small_hole_noise_dirname = "small_hole_noise"
    __big_hole_noise_dirname = "big_hole_noise"
    __sparse_noise_dirname = "sparse_noise"

    __load_dir_cache =   {
                        __plain_dirname : "plain",
                        __small_hole_noise_dirname : "smallhole",
                        __big_hole_noise_dirname : "bighole",
                        __sparse_noise_dirname : "sparse"
                    }

    def __announce(self):
        """
            function display informations.
            but it display only minimal information.
        """
        print("==Custom Loader Announcement==")
        print("data common dir : {}".format(Loader.__data_path))
        print("load data name : {}".format(self.plain_dir))
        print("load data noise type : {}".format(self.current_noisetype))
        print("==============================\n")

    # def __init__(self, directory, noise_type, nz):
    def __init__(self, 
                common_load_dir_name, 
                noise_type="plain",
                test_size = 100,
                valid_size = 0
                ):
        """
            directory : bareteeth, any name of dataset directory names. if it get full paths string, it extract last path name.
            noise_type = it determin input or noise input dataset. to do that. select string keyword in "bighole" or "smallhole" or "sparse" or plain.
        """

        



        split_dir = os.path.split(common_load_dir_name)
        if split_dir[-1] == '' : 
            common_dir = os.path.split(split_dir[:-1])[-1]
        else :
            common_dir = split_dir[-1]
        
        self.current_noisetype = noise_type
        self.valid_size = valid_size
        self.test_size = test_size
        self.plain_dir = common_load_dir_name

        self.plain_dir = self.plain_dir.lower()
        self.output_dir = self.plain_dir.replace(Loader.__plain_dirname, Loader.__load_dir_cache[self.current_noisetype])

        #load raw data. if there is no preprocessed raw data. excute fileconverter.
        self.label_facedata = DataProcessor(data_path=os.path.join(self.plain_dir), 
                                                reference_face_file=Loader.__reference_mesh_file, 
                                                testset_size=self.test_size)
        
        #input_data type : [noise, sparse_noise, hole]
        noise_dir = common_load_dir_name.lower()
        noise_dir = common_load_dir_name.replace(Loader.__plain_dirname, Loader.__load_dir_cache[self.current_noisetype])
        self.input_facedata = DataProcessor(data_path=os.path.join(noise_dir), 
                                                reference_face_file=Loader.__reference_mesh_file, 
                                                testset_size=test_size)

        self.references = self.input_facedata.ref_mesh
        #train data set. 
        self.train_input_vertice = self.input_facedata.vertices_train.astype("float32")
        self.train_label_vertice = self.label_facedata.vertices_train.astype("float32")
        

        #test data set
        self.test_input_vertice = self.input_facedata.vertices_test.astype("float32")        
        self.test_label_vertice = self.label_facedata.vertices_test.astype("float32")
        
        self.facedata = self.input_facedata.face.astype("int64")


        assert len(self.test_input_vertice) + len(self.train_input_vertice) == len(self.test_label_vertice) + len(self.train_label_vertice) , "funci error"
        
        self.__announce()
    
    def get_data_normalize(self, data):
        # return (data - self.label_facedata.mean) / self.label_facedata.std

        max_val = self.label_facedata.max_data
        min_val = self.label_facedata.min_data
        print("type :", data.dtype)
        
        print("max val min val", max_val, min_val)

        # if max_val == min_val, max and mean is inputs, +epssilon
        alpha = 0.9
        epsilon = 10e-8
        idx = max_val==min_val
        print(idx)
        max_val[idx] += epsilon
        min_val[idx] -= epsilon
        inputs = 2 * alpha * (data-min_val)/(max_val-min_val) - alpha
        print("inputs max min", np.min(inputs), np.max(inputs))
        print("type :", inputs.dtype)
        return inputs
        
    def get_data_normalize2(self, data):
        return (data - self.label_facedata.mean)# / self.label_facedata.std

    def get_data_denormalize2(self, data):
        # return data * self.label_facedata.std + self.label_facedata.mean
        return data  + self.label_facedata.mean
        
    def get_train_data(self):
        """
            return dict type datasets.
            key is [labels, input, val_input, val_labels]


        """
        return {"labels" : self.train_label_vertice, "input" : self.train_input_vertice}
    
    def get_test_data(self):
        """
            return dict() type test dataset.
            key is [input, labels]
        """
        return {"input" : self.test_input_vertice, "labels" : self.test_label_vertice}


    def get_train_shape(self):
        """
            return plain train dataset shape. it is normalized
        """
        shape = self.train_label_vertice.shape
        assert shape[0] == self.train_input_vertice.shape[0]
        return shape

    def remove_holemask(self,data, data_type):
        # shape = data.shape[0]
        # if data_type == "train":
        #     data[self.idx_train[shape]]  = 1.
        # elif data_type == "test":
        #     data[self.idx_test[shape]]  = 1.
        pass
    
    def get_face(self):
        return self.facedata

    def get_reference(self):
        return self.references

    def save_ply(self, data, name='no_name', path=None, mean=None, std=None):
        """
            input 
                data : 3-Dimension dataset or 2-Dimension. it is optimized at 3d mesh.
                name : save target name
                path (optional): if path is given. use path. if not, function use default __save_path="conv_ply". 
                mean (optional) : it is using for standarzations. if None it use internal faceData Ojbect mean.
                std (optional) : it is using for standarzations. if None it use internal faceData Ojbect std.
        """
        if path == None:
            path = Loader.__save_path
            
        if not os.path.exists(path):
            os.makedirs(path)
        


        if len(data.shape) == 3 :
            for i in range(len(data)):
                choice = data[i]
                # mesh = Mesh(v=choice, f=self.facedata)
                
                save_name = path+'/'+str(name)+'_ply_'+str(i)+'.ply'
                serialize.save_to_mesh(save_name, choice, self.facedata)
                #make_obj.save_obj(mesh, save_name)
                # mesh.write_mesh_file(save_name)
                



