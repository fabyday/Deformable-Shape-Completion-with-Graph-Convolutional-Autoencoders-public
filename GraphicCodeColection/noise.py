import numpy as np


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def bummpy_noise(orig_v, facedata, noise_ratio = 0.015, living_vertex_info = None):
    
    #mesh fresh info
    orig_data = orig_v.copy()
    facial_list = facedata

    data_info = len(orig_data.shape)
    noise_data = []
    if data_info >= 3:
        for v in orig_data:
            noise_data.append(add_noise_with_normal(v, facial_list, noise_ratio, living_vertex_info))
    elif data_info ==2 :
        noise_data = add_noise_with_normal(orig_data, facial_list, noise_ratio, living_vertex_info)
    else:
        print("Error")

    return np.array(noise_data)

#return 2 Demension data.
def add_noise_with_normal(vertexs, facial_list, noise_ratio, living_vertex_info = None):
    
    normal_list, v_avg_normal = get_normal(vertexs, facial_list)
    #v_avg_normal = vertex_avg_normal(vertexs, facial_list, normal_list)
    
    
    # add noise
    #uniform = np.random.uniform(0., 0.5, size=(vertexs.shape[0],1))
    #uniform = np.random.binomial(1, 0.25, size = (vertexs.shape[0],1))
    uniform = np.random.normal(scale=0.25, size=(vertexs.shape[0],1))
    if living_vertex_info is not None:
        living_vertex_matrix = np.zeros((vertexs.shape[0],1), dtype = np.float32)
        for i in living_vertex_info:
            np.random.shuffle(i)
            living_vertex_matrix[i[0], 0] = 1.0        
        uniform = uniform*living_vertex_matrix


    noise_data = vertexs+ noise_ratio*v_avg_normal*uniform


    """
    #print data in pyplot. compare orig_data to data that added normal

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print('norm : ', normal_list)
    print('avg_norm : ', v_avg_normal)

    # for name , v in {'v' : vertexs, 'norm' : normal_list, 'avg_norm' : v_avg_normal}.items() : 
    # ppap = vertexs+0.01*v_avg_normal
    ppap = noise_data
    v= vertexs.swapaxes(0, 1)
    ppap = ppap.swapaxes(0,1)

    print('v: ', v)
    print('ppap', ppap)
        # if name == 'v':
    ax.scatter(v[0], v[1], v[2], c='r', marker = 'o')
    print("Reeeeeeeeed!")
    ax.scatter(ppap[0], ppap[1], ppap[2], c='b', marker = 'o')
        # elif name == 'avg_norm':
        #     ax.scatter(v[0], v[1], v[2], c='g', marker = 'o')
        #     print(v)
        #     print("green!")
        # else:
        #     ax.scatter(v[0], v[1], v[2], c='b', marker = 'o')
        #     print("bbbbbbbbbbbbbb!")
        #     print(v)
    
    ax.set_xlabel('X')
    ax.set_xlim3d([-0.5, 0.5])
    ax.set_ylabel('Y')
    ax.set_ylim3d([-0.5, 0.5])
    ax.set_zlabel('Z')
    ax.set_zlim3d([-0.5, 0.5])
    ax.set_title('3D Test')
    plt.show()
    """
    return noise_data



#find normal about vertex,face info 
#return normal list
def get_normal(vertex_list,facial_list):
    face_normal_list = np.zeros_like(facial_list, dtype=np.float32)
    vertex_normal_list = np.zeros_like(vertex_list, dtype=np.float32)#5023, 3
    for i, f in enumerate(facial_list):
        vet_value = vertex_list[f]
        vec1 = vet_value[0] - vet_value[1]
        vec2 = vet_value[0] - vet_value[2]
        face_normal_list[i] = np.cross(vec1, vec2)
        vertex_normal_list[f] = face_normal_list[i]
        
        
    return (face_normal_list, vertex_normal_list/ np.linalg.norm(vertex_normal_list, axis=1).reshape(5023,1))



#calc average for normal
def vertex_avg_normal(vertex, facial_list, normal_list):
    avg_normal = np.zeros(vertex.shape, dtype=np.float32) #5023, 3
    v_index = np.zeros(vertex.shape[0]) # 5023

    for i, f in enumerate(facial_list):
        for v in f:
            v_index[v] += 1
            avg_normal[v] += normal_list[i]

    #make average.
    avg_normal /= v_index[..., np.newaxis]

    #make vector unit_vector
    avg_normal /= np.linalg.norm(avg_normal, axis=1).reshape(5023,1)
    

    return avg_normal



def save_noise(filename, noised_data):
    np.save(filename, noised_data)

def less_bummpy_noise(path = './hole_info.txt'):
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False
    
    with open(path) as f:
        read_data = f.read()
        data_list = read_data.split()
        data_list = [ x for x in data_list if is_int(x)]
        data_list = [int(x) for i, x in enumerate(data_list) if i%4 != 0]
        data_list = np.array(data_list)

        return data_list.reshape(data_list.shape[0]//3,3)


        

def hole_noise(data, facedata= None, hole_info_path = "./hole_info.txt"):
    data = data.copy()
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False

    with open(hole_info_path) as f:
        read_data = f.read()
        data_list = read_data.split()
        data_list = [ x for x in data_list if is_int(x)]
        data_list = [int(x) for i, x in enumerate(data_list) if i%4 != 0]
        print(data_list)
        print(data[:,0].shape)
        for i in data_list:
            data[:,i] = np.spacing(np.array([0],dtype='float32'))
    return data
    

def add_noise(X_data, face_data, living_vertex_info = None, noise_type = 'noise'):
    #solve normalizing
    X_noise_data = X_data
    if noise_type == 'noise':
        X_noise_data = bummpy_noise(X_noise_data, face_data, living_vertex_info = living_vertex_info)
    elif noise_type == 'hole':
        X_noise_data = hole_noise(X_noise_data)
    #normalizing.
    X_noise_data = (X_noise_data)
    return X_noise_data
