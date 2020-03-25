
import numpy as np

def generate_neighborhood(vertex, face):
    """
    Create by RJH
    Generate Sample's 1-ring neighborhood


    return neighborhood list.
    """

    print("what the heolgjiwejigwjlgeij ", face.shape, vertex.shape)
    neighbour = [ [] for _ in range(len(vertex)) ]
    
    # for f_data_list in face:
    #     for offset in f_data_list:
    #         for input_data in f_data_list:
    #             if input_data != offset:
    #                 neighbour[offset].append(input_data)
    
    for f_data_list in face:
        for i, vertex_num in enumerate(f_data_list):
                if f_data_list[(i+1)%3] not in neighbour[vertex_num]:
                    neighbour[vertex_num].append(f_data_list[(i+1)%3])
                if f_data_list[(i+2)%3] not in neighbour[vertex_num]:
                    neighbour[vertex_num].append(f_data_list[(i+2)%3])
                
                
    

    

    




    pointnum = len(vertex)
    maxdegree = 0 
    degree = np.zeros(pointnum, dtype=np.int32)

    
    for i in range(pointnum):
        degree[i] = len(neighbour[i])
        if degree[i] > maxdegree:
            maxdegree = degree[i]
    after_neighbour = np.zeros((pointnum, maxdegree), dtype=np.int32)
    
    for i in range(pointnum):
        zero_adding_size = maxdegree - degree[i]
        after_neighbour[i] = neighbour[i]+[0]*zero_adding_size
    
    degree = degree.astype(np.float32)
    degree = degree.reshape(pointnum, 1)
    return after_neighbour, degree, maxdegree

