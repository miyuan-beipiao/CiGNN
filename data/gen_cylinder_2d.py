#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: gen_cylinder_2d.py 
@time: 2023/02/15
@contact: miyuan@ruc.edu.cn
@site:  
@software: PyCharm 

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓          ┏┓
            ┏┛┻━━━━━━━━━━┛┻┓
            ┃      ☃       ┃
            ┃    ┳┛  ┗┳    ┃
            ┃       ┻      ┃
            ┗━━━┓      ┏━━━┛
                ┃      ┗━━━━━━━━┓
                ┃  神兽保佑       ┣┓
                ┃　永无BUG！      ┏┛
                ┗━┓┓┏━━━━━━━━┳┓┏┛
                  ┃┫┫        ┃┫┫
                  ┗┻┛        ┗┻┛ 
"""
import os
import torch
import numpy as np
from cylinderUtils import read_dataset

"""
import numpy as np
import networkx as nx

G = nx.erdos_renyi_graph(10, 0.15)
print(G.gra)

mean, std = self._compute_stats(t[train], u[train])
np.savez(self.stats_file, mean=mean, std=std)
"""

if __name__ == '__main__':

    dataset = 'cf'
    path = '/mnt/miyuan/AI4Physics/Data/{}/raw/'.format(dataset)

    seed = np.linspace(0, 17, 18, dtype=int)
    # seed = np.linspace(1, 1, 1, dtype=int)
    for i, item in enumerate(seed):
        print(i)
        np.random.seed(item)

        filenames = [fn for fn in os.listdir(path) if
                     fn.endswith('_{}.csv'.format(item)) and fn.startswith('{}_u'.format(dataset))]
        file_path = os.path.join(path + filenames[0])

        meshnames = [fn for fn in os.listdir(path) if
                     fn.endswith('_{}.txt'.format(item)) and fn.startswith('{}_u'.format(dataset))]
        mesh_path = os.path.join(path + meshnames[0])

        # datafile = 'cylinder_flow_comsol.csv'
        # meshfile = 'mesh_comsol_output.txt'
        # cf_u1_h0.41_w1.2_r0.08_x0.10_y0.20_7
        R = float(filenames[0][18:22])
        cylinder_x = float(filenames[0][24:28])
        cylinder_y = float(filenames[0][30:34])
        data, pos, edge_index, node_types, edge_attr, mesh_elements = read_dataset(datafile=file_path,
                                                                                   meshfile=mesh_path,
                                                                                   center=[cylinder_x, cylinder_y], R=R)
        num_node = 3400  # 2520
        # [601,2520,3] [p,u,v]
        u_path = os.path.join(path + '/2d_{}_u_t{}_n{}_{}.npy'.format(dataset, data.shape[0], num_node, item))
        np.save(u_path, data)
        # [2520,2]
        x_path = os.path.join(path + '/2d_{}_x_t{}_n{}_{}.npy'.format(dataset, data.shape[0], num_node, item))
        np.save(x_path, pos)
        # [2,28452]
        eid_path = os.path.join(path + '/2d_{}_eid_t{}_n{}_{}.npy'.format(dataset, data.shape[0], num_node, item))
        np.save(eid_path, edge_index)
        # [2520,1]
        type_path = os.path.join(
            path + '/2d_{}_type_t{}_n{}_{}.npy'.format(dataset, data.shape[0], num_node, item))
        np.save(type_path, node_types)
        face_path = os.path.join(
            path + '/2d_{}_face_t{}_n{}_{}.npy'.format(dataset, data.shape[0], num_node, item))
        np.save(face_path, mesh_elements)
        v_path = os.path.join(path + '/2d_{}_v_t{}_n{}_{}.npy'.format(dataset, data.shape[0], num_node, item))
        temp_list = []
        for _ in range(data.shape[1]):
            temp_list.append([R, cylinder_x, cylinder_y])
        v = np.stack(temp_list, axis=0)
        np.save(v_path, v)
