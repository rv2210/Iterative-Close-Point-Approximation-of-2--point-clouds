import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d
import copy

#Read Point CLoud Data #Snippet given in overleaf
demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

#Tranformation Initialization matrix from slack
T= [[0.862  , 0.011 , -0.507,   0.5],
[-0.139 , 0.967,  -0.215 ,  0.7],
[0.487 ,  0.255 ,  0.835 , -1.4],
[0.0  ,   0.0   ,  0.0   ,  1.0]]
#print(np.asarray(source.points[0]))
source.transform(T)
#print(source.shape)
#Converting to numpy array
source_array = np.asarray(source.points)
target_array = np.asarray(target.points)

#Definig Cost function

T_last = np.identity(4)
cost_function = 100000
iteration=0
prev_cost = 0
while cost_function>1:

        tree = KDTree(source_array)              
        dist, ind = tree.query(target_array, k=1)
        source_array = source_array[ind]
        source_array = source_array.reshape((len(source_array),3))

        Mean_source= np.mean(source_array, axis= 0) 
        Mean_target= np.mean(target_array, axis= 0)

        source_new= source_array- Mean_source
        target_new= target_array- Mean_target
        H= np.matmul(target_new.T, source_new)
        #print(H)
        U, sigma, VT = np.linalg.svd(H)

        R_hat = np.matmul(U,VT)
        #print(np.linalg.det(R_hat))
        t_hat = Mean_target.T - np.matmul(R_hat, Mean_source.T)

        T = np.identity(3 + 1)
        T[:3, :3] = R_hat
        T[:3, 3] = t_hat

        T_new = np.matmul(T_last,T)
        T_last = T_new


        #cost_function= (np.sum(np.square(target_array- (np.matmul(T_prev[:3, :3],source_array.T)).T-T_prev[:3, 3])))
        cost_function= (np.sum(np.square(target_new- (np.matmul(T[:3, :3],source_new.T)).T)))
        print("Cost_function for the current Transformation is ",cost_function)
        source_array = (np.matmul(R_hat,source_array.T)).T+t_hat

        #error = cost_e - prev_cost
        #if error<1:
        #        break
        #prev_cost = cost_e
        iteration = iteration+1

        if iteration==20:
            break


T = T_last

source_temp = copy.deepcopy(source)
target_temp = copy.deepcopy(target)
o3d.visualization.draw_geometries([source_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],up=[-0.3402, -0.9189, -0.1996])
o3d.visualization.draw_geometries([target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],up=[-0.3402, -0.9189, -0.1996])
source_temp.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([source_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],up=[-0.3402, -0.9189, -0.1996])

target_temp.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],up=[-0.3402, -0.9189, -0.1996])

#T= np.eye(4,4)
source_temp.transform(T)

print(T)
o3d.visualization.draw_geometries([source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],up=[-0.3402, -0.9189, -0.1996])