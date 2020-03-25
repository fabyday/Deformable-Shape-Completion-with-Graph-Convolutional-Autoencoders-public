import igl 


def downsampling(vertex, face, max_m):
    return igl.qslim(vertex, face, max_m)