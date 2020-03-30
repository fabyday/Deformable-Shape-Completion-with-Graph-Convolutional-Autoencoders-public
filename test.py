import GraphicCodeColection.serialize as se
import GraphicCodeColection.mesh_sampling as ms 
import numpy as np 
v,f = se.load_from_mesh("./data/template.obj")


def ne(v, f ): 

    adj = np.zeros([len(v), len(v)])
    print(f[:, 1])
    for i in range(3) : 
        adj[f[:,i], f[:,(i+1)%3]] = 1
        adj[f[:,i], f[:,(i+2)%3]] = 1


    a = [[] for _ in range(len(v))]
    for row, elem in enumerate(adj) : 
        for col, data in enumerate(elem):
            if data != 0 : 
                a[row].append(col)
    max_size =0
    for row in a : 
        if len(row) > max_size : 
            max_size = len(row)

    for row in a : 
        size = max_size - len(row)
        row+=[0]*size

    return np.array(a)


e1, _, _  = ms.generate_neighborhood(v, f)
e2 = ne(v,f)

print(e1.shape)
print(e2.shape)

test = True
for i in range(5023) : 
    for j in range(32):
        if e1[i,j] not in e2[i,:]:
            test = False
for i in range(5023) : 
    for j in range(32):
        if e2[i,j] not in e1[i,:]:
            test = False


print("test : ", test)

    
    

    
    



        

        
