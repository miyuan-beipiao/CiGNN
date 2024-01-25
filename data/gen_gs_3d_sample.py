#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: gen_gs_3d.py 
@time: 2023/01/31
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
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import scipy.io
import os
import torch

# device = torch.device('cuda:5')

if __name__ == "__main__":
    dataset = 'gs'
    # Diffusion coefficients
    DA = 0.2
    DB = 0.1
    # define birth/death rates
    f = 0.025
    k = 0.055
    # grid size
    x_num, y_num, z_num = 48, 48, 48
    x_max, y_max, z_max = 96.0, 96.0, 96.0

    # x_num, y_num, z_num = 10, 10, 10
    # x_max, y_max, z_max = 10.0,10.0,10.0

    node_num = x_num * y_num * z_num
    # update in time
    # delta_t = 0.25
    delta_t = 0.25
    # spatial step
    dx = x_max // x_num  # 1.0
    dy = y_max // y_num  # 1.0
    dz = z_max // z_num  # 1.0
    # 3000
    N_simulation_steps = 3000
    # seed = np.linspace(1, 100, 100, dtype=int)
    seed = np.linspace(1, 60, 60, dtype=int)
    pad_len = 1
    space_step = 4

    for i, item in enumerate(seed):
        print(item)
        np.random.seed(item)
        u_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/3d_{}_u_t{}_n{}_{}.npy'.format(dataset, dataset, N_simulation_steps + 1,
                                                                           node_num, item))

        UV = np.load(u_path).reshape(N_simulation_steps + 1, x_num, y_num, z_num, 2)
        sample_UV = UV[:, ::space_step, ::space_step, ::space_step, :].reshape(N_simulation_steps + 1, -1, 2)
        sample_u_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/3d_{}_u_t{}_n{}_{}.npy'.format(dataset, dataset, N_simulation_steps + 1,
                                                                           node_num // (space_step ** 3), item))
        np.save(sample_u_path, sample_UV)

        # scipy.io.savemat(u_path, {'uv': UV_RK4_dt05})
        # for i in range(0, N_simulation_steps, 300):
        #     plot3D(UV_RK4_dt05, i, N)

        # fig_save_path = '/mnt/miyuan/AI4Physics/Data/{}/'.format(dataset)
        # for i in range(0, 150, 10):
        #     postprocess3D(A_record, B_record, i, fig_save_path)

        x = np.linspace(0 - pad_len * dx * space_step, x_max + pad_len * dx * space_step,
                        x_num // space_step + 2 * pad_len + 1)
        y = np.linspace(0 - pad_len * dy * space_step, y_max + pad_len * dy * space_step,
                        y_num // space_step + 2 * pad_len + 1)
        z = np.linspace(0 - pad_len * dz * space_step, z_max + pad_len * dz * space_step,
                        z_num // space_step + 2 * pad_len + 1)

        x_star, y_star, z_star = np.meshgrid(x[:-1], y[:-1], z[:-1])
        x_star = np.reshape(x_star, (-1, 1))
        y_star = np.reshape(y_star, (-1, 1))
        z_star = np.reshape(z_star, (-1, 1))
        pos = np.concatenate((x_star, y_star, z_star), axis=1)

        x_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/3d_{}_x_t{}_n{}_{}.npy'.format(dataset, dataset, N_simulation_steps + 1,
                                                                           node_num // (space_step ** 3), item))
        np.save(x_path, pos)
