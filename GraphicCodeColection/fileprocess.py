import os 
import glob 

from . import serialize
import numpy as np 
from . import noise
import tqdm
from . import mesh
class DirectoryManager():
    def __init__(self, dir_path, target_ext, recursive = True):
        if type(target_ext) != type([]):
            target_ext=[target_ext]
        self.dir_path = dir_path
        self.target_ext = target_ext
        self.recursive = recursive
        self.path = glob.glob(os.path.join(self.dir_path, "**"), recursive=True)
    
    def __call__(self, func, *args):
        target_ext = tuple(self.target_ext)
        print(target_ext)
        for obj in self.path:
            if obj.endswith(target_ext):
                func(obj, self.dir_path, *args)            



class DataProcessor():
    __default_name = "data.npy"
    def __init__(self, data_path,
                reference_face_file, 
                testset_size, 
                filename = None, 
                valset_size=100):
        """
            reference_face_file is obj.
        """
        
        self.testset_size = testset_size
        self.valset_size = valset_size
        self.data_path = data_path
        self.reference_face_file = reference_face_file
        self.filename = filename
        self._load() 
        self.std, self.mean = self._store_normalize_data() #for normalized



    def _load(self):
        """
            load face data and vertices data.

        """
        if self.filename != None : 
            load_name = self.filename
        else : 
            load_name = DataProcessor.__default_name
        
        raw_vertex_data = serialize.load_from_numpy(os.path.join(self.data_path, load_name))
        print(raw_vertex_data.shape)
        ref_v, raw_face_data = serialize.load_from_mesh(self.reference_face_file)
        self.ref_mesh = mesh.Mesh(ref_v, raw_face_data)
        self.face = raw_face_data
        self.vertices = raw_vertex_data
        self.std = np.std(self.vertices)
        self.mean = np.mean(self.vertices)
        #split data.
        self._split()
    
    def _split(self):
        self.vertices_train = self.vertices[:-self.testset_size]
        self.vertices_test = self.vertices[-self.testset_size:]
        self.vertices_val = self.vertices_train[-self.valset_size:]
        self.vertices_train = self.vertices_train[:-self.valset_size]


    def _store_normalize_data(self):
        return self.std, self.mean


    

class DataPreprocessor():
    def __init__(self, input_dir_list, output_dir_list):
        """
            input_dir_list : 
            output_dir_list :
            reference_mesh_file_name : 
        """
        if type(input_dir_list) != type(list()):
            input_dir_list = [input_dir_list]
        if type(output_dir_list) != type(list()):
            input_dir_list = [output_dir_list]

        assert len(input_dir_list) == len(output_dir_list), \
                "input_dir_list {:3} and output_dir_list {:3} is different size.\
                please make sure it to be same.".format(len(input_dir_list), len(output_dir_list))
        
        self.input_dir_list = input_dir_list
        self.output_dir_list = output_dir_list

        self.pre_extension = [".obj", ".ply"]

    
    def convert_mesh2numpy_expwise(self):
        """
            ideal dir structure
            parent ---
                     - plain
                     - noise
                     - hole
            input_dir_list = [plain, noise, hole] 

            
        """
        test_exps = ['bareteeth','cheeks_in','eyebrow','high_smile','lips_back','lips_up','mouth_down',
                    'mouth_extreme','mouth_middle','mouth_open','mouth_side','mouth_up']

        for cur_dir, cur_output_dir in zip(self.input_dir_list, self.output_dir_list):
            if not os.path.exists(cur_dir) : 
                raise Exception("{} is not exist. check files name string.".format(cur_dir))
            print("cur_dir : {}, cur_output_dir : {}".format(cur_dir, cur_output_dir))
            v_list = []
            for exp in test_exps : 
                cur_dir_child = glob.glob(os.path.join(cur_dir, "*"))
                for child in cur_dir_child:
                    find_path = os.path.join(child, exp)

                    truncated_v_list = self._mesh2numpy(find_path)
                    v_list.append(truncated_v_list)
                
                v_data = np.concatenate(v_list)
                if not os.path.exists(os.path.join(cur_output_dir, exp)):
                    os.makedirs(os.path.join(cur_output_dir, exp)) #recursively make directories.
                serialize.save_to_numpy(os.path.join(cur_output_dir, exp, "data"), v_data)



    def _mesh2numpy(self, path):
        """
            convert file type(.ply or .obj)
        """
        if not os.path.exists(path):
            raise Exception("path is not exist.")
        cur_dir_child = glob.glob(os.path.join(path, "*"))
        

        v_list = []
        for child_file in cur_dir_child : 
            if child_file.endswith( tuple(self.pre_extension) ) :
                v, _ = serialize.load_from_mesh(child_file)
                v_list.append(v)

        return np.array(v_list)


        
        

class NoiseProcessor():
    __noise_type = {"sparse" : 'noise', "hole" : 'hole'}
    def __init__(self, input_dir, output_dir, noise_type, template_face = None, living_vertex_info=None):
        """
            input_dir : [input_dir]. it is source dest.
        """

        self.pre_extension = [".obj", ".ply"]

        if type(input_dir) != type(list()):
            input_dir = [input_dir]
        if type(output_dir) != type(list()):
            output_dir = [output_dir]
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.template_face = template_face
        self.living_vertex_info = living_vertex_info
        
        if noise_type not in NoiseProcessor.__noise_type:
            noise_type = 'sparse'
        self.noise_type = NoiseProcessor.__noise_type[noise_type]
        
    def save_noise_dataset(self):
        self._add_noise(self.input_dir, self.output_dir, self.noise_type)

    def _add_noise(self,input_dir, output_dir, noise_type):
        for cur_dir, cur_output_dir in zip(input_dir, output_dir):
            if not os.path.exists(cur_dir) : 
                raise Exception("{} is not exist. check files name string.".format(cur_dir))
            cur_dir = cur_dir.replace("/", "\\")
            cur_output_dir = cur_output_dir.replace('/', '\\')
            for child in (glob.glob(os.path.join(cur_dir, "**"), recursive=True)):
                if child.endswith( tuple(self.pre_extension) ) : 
                    child=os.path.abspath(os.path.normcase(child))
                    child.replace('/', "\\")
                    child=child.lower()
                    print(child)
                    cur_dir=os.path.abspath(os.path.normcase(cur_dir))
                    save_path = child.replace(cur_dir, cur_output_dir)
                    save_dir = os.path.split(save_path)[0]

                    v, f = self.apply_noise(child, noise_type)


                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    serialize.save_to_mesh(save_path, v, f)

    def apply_noise(self, filename, noise_type):
        v,f = serialize.load_from_mesh(filename)
        living_index_info = noise.less_bummpy_noise(path = './hole_info.txt')
        v = noise.add_noise(v,f, living_vertex_info=living_index_info, noise_type=noise_type)
        return v, f
        
        
    

