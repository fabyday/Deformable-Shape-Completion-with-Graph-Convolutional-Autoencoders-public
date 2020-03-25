import numpy as np 

import scipy

from . import serialize
from . import search
class Mesh():
    def __init__(self,
                 v=None,
                 f=None,
                 segm=None,
                 filename=None,
                 ppfilename=None,
                 lmrkfilename=None,
                 basename=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 landmarks=None):
        self.v = v
        self.f = f 
        self.filename = filename

    
    
    #####################serialization#######################################
    def write_mesh_file(self, filename=None):
        if filename == None:
            filename = "no_name"
        serialize.save_to_mesh(filename, self.v, self.f)
    

    def read_mesh_file(self, filename=None):
        if filename == None :
            raise Exception("file name is None")
        self.filename = filename
        serialize.load_from_mesh(filename)

    def compute_aabb_tree(self):
        return search.AabbTree(self)