import igl
import numpy as np 
import os

############################################
# serialization for mesh or numpy files.
############################################

def load_from_mesh(file_path):
    """
        return (vertex, face)
    """
    
    return igl.read_triangle_mesh(file_path)



def save_to_mesh(file_path, v, f):
    """
        
    """
    igl.write_triangle_mesh(file_path, v, f)





def save_to_numpy(file_path, data):
    np.save(file_path, data)

def load_from_numpy(file_path) : 
    return np.load(file_path, allow_pickle=True)

