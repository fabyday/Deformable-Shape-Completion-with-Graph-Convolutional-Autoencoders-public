
import numpy as np

class AabbTree(object):
    """Encapsulates an AABB (Axis Aligned Bounding Box) Tree"""
    def __init__(self, m):
        #TODO s
        self.v = m.v
        self.f = m.f
        # self.cpp_handle = spatialsearch.aabbtree_compute(m.v.astype(np.float64).copy(order='C'), m.f.astype(np.uint32).copy(order='C'))

    def nearest(self, v_samples, nearest_part=False):
        "nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)"
        # f_idxs, f_part, v = spatialsearch.aabbtree_nearest(self.cpp_handle, np.array(v_samples, dtype=np.float64, order='C'))
        # return (f_idxs, f_part, v) if nearest_part else (f_idxs, v)
        pass
    # def nearest_alongnormal(self, points, normals):
    #     distances, f_idxs, v = spatialsearch.aabbtree_nearest_alongnormal(self.cpp_handle,
    #                                                                       points.astype(np.float64),
    #                                                                       normals.astype(np.float64))
    #     return (distances, f_idxs, v)
