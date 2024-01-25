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


# ============ define relevant functions =============
# https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb

def apply_laplacian(mat, dx=2.0):
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    For more information see
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    neigh_mat = -7.5 * mat.copy()
    neighbors = [
        (4 / 3, (1, 0, 0)),
        (4 / 3, (0, 1, 0)),
        (4 / 3, (0, 0, 1)),
        (4 / 3, (-1, 0, 0)),
        (4 / 3, (0, -1, 0)),
        (4 / 3, (0, 0, -1)),
        (-1 / 12, (-2, 0, 0)),
        (-1 / 12, (0, -2, 0)),
        (-1 / 12, (0, 0, -2)),
        (-1 / 12, (2, 0, 0)),
        (-1 / 12, (0, 2, 0)),
        (-1 / 12, (0, 0, 2)),
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1, 2))

    return neigh_mat / dx ** 2


# def apply_laplacian(mat, dx=1.0):
#     """This function applies a discretized Laplacian
#     in periodic boundary conditions to a matrix
#     For more information see
#     https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
#     """
#
#     # Seven-point stencil discrete Laplacian operator
#     L = -6*mat
#     L += np.roll(mat, (0,-1,0), (0,1,2))
#     L += np.roll(mat, (0,+1,0), (0,1,2))
#     L += np.roll(mat, (-1,0,0), (0,1,2))
#     L += np.roll(mat, (+1,0,0), (0,1,2))
#     L += np.roll(mat, (0,0,-1), (0,1,2))
#     L += np.roll(mat, (0,0,+1), (0,1,2))
#
#     return L/dx**2

# Define the update formula for chemicals A and B
def update(A, B, DA=0.2, DB=0.1, f=0.025, k=0.055, dt=1.0):
    """Apply the Gray-Scott update formula"""

    # compute the diffusion part of the update
    diff_A = DA * apply_laplacian(A, dx)
    diff_B = DB * apply_laplacian(B, dx)

    # Apply chemical reaction
    reaction = A * B ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A)
    diff_B -= (k + f) * B

    A += diff_A * dt
    B += diff_B * dt

    return A, B


def update_rk4(A0, B0, DA=0.2, DB=0.1, f=0.025, k=0.055, delta_t=1.0):
    """Update with Runge-kutta-4 method
       See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    ############# Stage 1 ##############
    # compute the diffusion part of the update
    diff_A = DA * apply_laplacian(A0, dx)
    diff_B = DB * apply_laplacian(B0, dx)
    # diff_A = DA * apply_laplacian(A0)
    # diff_B = DB * apply_laplacian(B0)

    # Apply chemical reaction
    reaction = A0 * B0 ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A0)
    diff_B -= (k + f) * B0

    K1_a = diff_A
    K1_b = diff_B

    ############# Stage 1 ##############
    A1 = A0 + K1_a * delta_t / 2.0
    B1 = B0 + K1_b * delta_t / 2.0

    diff_A = DA * apply_laplacian(A1, dx)
    diff_B = DB * apply_laplacian(B1, dx)
    # diff_A = DA * apply_laplacian(A1)
    # diff_B = DB * apply_laplacian(B1)

    # Apply chemical reaction
    reaction = A1 * B1 ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A1)
    diff_B -= (k + f) * B1

    K2_a = diff_A
    K2_b = diff_B

    ############# Stage 2 ##############

    A2 = A0 + K2_a * delta_t / 2.0
    B2 = B0 + K2_b * delta_t / 2.0

    diff_A = DA * apply_laplacian(A2, dx)
    diff_B = DB * apply_laplacian(B2, dx)
    # diff_A = DA * apply_laplacian(A2)
    # diff_B = DB * apply_laplacian(B2)

    # Apply chemical reaction
    reaction = A2 * B2 ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A2)
    diff_B -= (k + f) * B2

    K3_a = diff_A
    K3_b = diff_B

    ############# Stage 3 ##############
    A3 = A0 + K3_a * delta_t
    B3 = B0 + K3_b * delta_t

    diff_A = DA * apply_laplacian(A3, dx)
    diff_B = DB * apply_laplacian(B3, dx)
    # diff_A = DA * apply_laplacian(A3)
    # diff_B = DB * apply_laplacian(B3)

    # Apply chemical reaction
    reaction = A3 * B3 ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A3)
    diff_B -= (k + f) * B3

    K4_a = diff_A
    K4_b = diff_B

    # Final solution
    A = A0 + delta_t * (K1_a + 2 * K2_a + 2 * K3_a + K4_a) / 6.0
    B = B0 + delta_t * (K1_b + 2 * K2_b + 2 * K3_b + K4_b) / 6.0

    return A, B


def get_initial_A_and_B(N, random_influence=0.2):
    """get the initial chemical concentrations"""
    # N = 48

    # get initial homogeneous concentrations
    A = (1 - random_influence) * np.ones((N, N, N))
    B = np.zeros((N, N, N))

    # put some noise on there
    A += random_influence * np.random.random((N, N, N))
    B += random_influence * np.random.random((N, N, N))

    #    # get center and radius for initial disturbance
    #    # apply initial disturbance - IC 1
    #    N2, r = N // 2, 4
    #    A[N2-r:N2+r, N2-r:N2+r, N2-r:N2+r] = 0.50
    #    B[N2-r:N2+r, N2-r:N2+r, N2-r:N2+r] = 0.25
    #
    #    N1, N2, N3, r = N // 4, 8, 6, 4
    #    A[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.50
    #    B[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.25
    #
    #    N1, N2, N3, r = 6, N // 4, 8, 4
    #    A[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.50
    #    B[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.25

    #    # get center and radius for initial disturbance
    #    # apply initial disturbance - IC 2
    #    N2, r = N // 2, 4
    #    A[N2-r:N2+r, N2-r:N2+r, N2-r:N2+r] = 0.50
    #    B[N2-r:N2+r, N2-r:N2+r, N2-r:N2+r] = 0.25
    #
    #    N1, N2, N3, r = N // 4, 8, 6, 4
    #    A[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.50
    #    B[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.25
    #
    #    N1, N2, N3, r = 6, 8, N // 4, 4
    #    A[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.50
    #    B[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.25

    # get center and radius for initial disturbance
    # apply initial disturbance - IC 3
    """ """
    N2, r = N // 2, 4
    A[N2 - r:N2 + r, N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
    B[N2 - r:N2 + r, N2 - r:N2 + r, N2 - r:N2 + r] = 0.25

    N1, N2, N3, r = 6, N // 8, 8, 4
    A[N1 - r:N1 + r, N2 - r:N2 + r, N3 - r:N3 + r] = 0.50
    B[N1 - r:N1 + r, N2 - r:N2 + r, N3 - r:N3 + r] = 0.25

    N1, N2, N3, r = 6, 8, N // 4, 4
    A[N1 - r:N1 + r, N2 - r:N2 + r, N3 - r:N3 + r] = 0.50
    B[N1 - r:N1 + r, N2 - r:N2 + r, N3 - r:N3 + r] = 0.25

    """
    N2, r = N // 2, 2
    A[N2 - r:N2 + r, N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
    B[N2 - r:N2 + r, N2 - r:N2 + r, N2 - r:N2 + r] = 0.25

    N1, N2, N3, r = 3, N // 8, 8, 2
    A[N1 - r:N1 + r, N2 - r:N2 + r, N3 - r:N3 + r] = 0.50
    B[N1 - r:N1 + r, N2 - r:N2 + r, N3 - r:N3 + r] = 0.25

    N1, N2, N3, r = 3, 4, N // 4, 2
    A[N1 - r:N1 + r, N2 - r:N2 + r, N3 - r:N3 + r] = 0.50
    B[N1 - r:N1 + r, N2 - r:N2 + r, N3 - r:N3 + r] = 0.25
    """

    #    # get center and radius for initial disturbance
    #    # apply initial disturbance - IC 4 (4)
    #    N2, r = N // 2, 4
    #    A[N2-r:N2+r, N2-r:N2+r, N2-r:N2+r] = 0.50
    #    B[N2-r:N2+r, N2-r:N2+r, N2-r:N2+r] = 0.25
    #
    #    N1, N2, N3, r = N // 4, 8, 6, 4
    #    A[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.50
    #    B[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.25
    #
    #    N1, N2, N3, r = 6, 8, N // 4, 4
    #    A[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.50
    #    B[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.25
    #
    #    N1, N2, N3, r = 6, N // 4, 8, 4
    #    A[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.50
    #    B[N1-r:N1+r, N2-r:N2+r, N3-r:N3+r] = 0.25
    """  
    # N1, N2, N3 = N // 4, N // 2, 3 * N // 4
    r = int(N / 10.0)
    N1 = torch.randint(low=r, high=N - r, size=(1,))

    A[N1 - r:N1 + r, N1 - r:N1 + r, N1 - r:N1 + r] = 0.50
    B[N1 - r:N1 + r, N1 - r:N1 + r, N1 - r:N1 + r] = 0.25
    
    """

    return A, B


def postProcess(UV, xmin, xmax, ymin, ymax, num):
    ''' num: Number of time step
    '''
    x = np.linspace(-50, 50, 50)
    y = np.linspace(-50, 50, 50)
    x_star, y_star = np.meshgrid(x, y)
    u_pred = UV[0, num]
    v_pred = UV[1, num]
    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    #
    cf = ax[0].scatter(x_star, y_star, c=u_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=9, vmin=0,
                       vmax=1)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title('u-FD')
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
    #
    cf = ax[1].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=9, vmin=0,
                       vmax=1)
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title('v-FD')
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)
    #
    # plt.draw()
    plt.savefig('./figures/uv_' + str(num).zfill(3) + '.png')
    plt.close('all')


def postprocess3D(A_record, B_record, num, fig_save_path):
    x = np.linspace(-50, 50, 50)
    y = np.linspace(-50, 50, 50)
    z = np.linspace(-50, 50, 50)
    x, y, z = np.meshgrid(x, y, z)
    # x, y, z = np.meshgrid(x[:-1], y[:-1], z[:-1])

    # ellipsoid
    # values = B
    values = A_record[num]

    fig = go.Figure(data=go.xz(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=0.3,  # ,np.min(values)
        isomax=0.5,  # 0.5,
        opacity=0.2,
        colorscale='RdBu',  # 'BlueRed',
        surface_count=2,  # number of isosurfaces, 2 by default: only min and max
        # slices_z=dict(show=True, locations=[-0, ]),
        # slices_y=dict(show=True, locations=[-0, ]),
        # slices_x=dict(show=True, locations=[-0, ]),
        # colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
        # caps=dict(x_show=False, y_show=False)
    ))
    # fig.show()
    # fig.write_image('./figures/Iso_surf_%d.png' % num)
    fig.write_image(fig_save_path + 'Iso_surf_%d.png' % num)
    plt.close('all')


def plot3D(output, num=0, N=128):
    def rmv_tick():
        ax = plt.gca()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

    x = np.linspace(-16, 16, N)
    y = np.linspace(-16, 16, N)
    z = np.linspace(-16, 16, N)
    x, y, z = np.meshgrid(x, y, z)
    #
    u_pred = output[num, 0, :, :, :]
    v_pred = output[num, 1, :, :, :]
    x_f, y_f, z_f, color_u, color_v = x.flatten(), y.flatten(), z.flatten(), u_pred.flatten(), v_pred.flatten()
    #
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # ax.set_aspect('equal')
    ax.set_aspect('auto')
    ax.set_title('u-CRNN')
    rmv_tick()
    cf = ax.scatter(x, y, z, c=color_u, marker='s', alpha=0.4, s=4, cmap='hot')  # , vmin=-.4, vmax=0.8
    fig.colorbar(cf, orientation='horizontal', ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
    #
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.set_aspect('equal')
    ax.set_aspect('auto')
    ax.set_title('v-CRNN')
    rmv_tick()
    cf = ax.scatter(x, y, z, c=color_v, marker='s', alpha=0.4, s=4, cmap='hot')  # , vmin=-.2, vmax=0.20
    fig.colorbar(cf, orientation='horizontal', ax=ax, fraction=0.046, pad=0.1, shrink=0.7)
    #
    fig.suptitle('Time = ' + str(round(num * 0.01, 3)) + 's')
    fig_save_path = '/mnt/miyuan/AI4Physics/Data/{}/'.format(dataset)
    plt.savefig(fig_save_path + 'uv_[i=%d].png' % (i))
    plt.close('all')


def genData(N, N_simulation_steps, DA, DB, f, k, delta_t):
    # intialize the chemical concentrations
    A, B = get_initial_A_and_B(N, random_influence=0.1)
    # A, B = get_initial_A_and_B(N, random_influence=0.)
    # load initial condition
    # IC = scipy.io.loadmat('./IC_2x50x50x50.mat')['uv']
    # A, B = IC[0], IC[1]

    A_record = A.copy()[None, ...]
    B_record = B.copy()[None, ...]

    for step in range(N_simulation_steps):
        # A, B = update(A, B, DA, DB, f, k, delta_t)
        A, B = update_rk4(A, B, DA, DB, f, k, delta_t)

        A_record = np.concatenate((A_record, A[None, ...]), axis=0)
        B_record = np.concatenate((B_record, B[None, ...]), axis=0)

        if (step + 1) % 100 == 0:
            print("已生成", step + 1, "步")

    UV = np.concatenate((A_record[None, ...], B_record[None, ...]), axis=0)
    UV = np.transpose(UV, (1, 2, 3, 4, 0))  # [3001,48,48,48,2]
    UV = UV[:, ::space_step, ::space_step, ::space_step, :]

    UV = UV.reshape(UV.shape[0], -1, UV.shape[-1])
    return UV


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
    seed = np.linspace(53, 60, 8, dtype=int)
    pad_len = 1
    space_step = 1

    for i, item in enumerate(seed):
        print(item)
        np.random.seed(item)
        u_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/3d_{}_u_t{}_n{}_{}.npy'.format(dataset, dataset, N_simulation_steps + 1,
                                                                           node_num // (space_step ** 3), item))

        UV = genData(x_num, N_simulation_steps, DA, DB, f, k, delta_t)

        if np.isnan(UV).any() or np.isinf(UV).any():
            print('存在异常值')

        np.save(u_path, UV)
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

    # for i, item in enumerate(seed):
    #     x = np.linspace(0 - pad_len * dx * space_step, x_max + pad_len * dx * space_step,
    #                     x_num // space_step + 2 * pad_len + 1)
    #     y = np.linspace(0 - pad_len * dy * space_step, y_max + pad_len * dy * space_step,
    #                     y_num // space_step + 2 * pad_len + 1)
    #     z = np.linspace(0 - pad_len * dz * space_step, z_max + pad_len * dz * space_step,
    #                     z_num // space_step + 2 * pad_len + 1)
    #
    #     x_star, y_star, z_star = np.meshgrid(x[:-1], y[:-1], z[:-1])
    #     x_star = np.reshape(x_star, (-1, 1))
    #     y_star = np.reshape(y_star, (-1, 1))
    #     z_star = np.reshape(z_star, (-1, 1))
    #     pos = np.concatenate((x_star, y_star, z_star), axis=1)
    #
    #     x_path = os.path.join(
    #         '/mnt/miyuan/AI4Physics/Data/{}/3d_{}_x_t{}_n{}_{}.npy'.format(dataset, dataset, N_simulation_steps + 1,
    #                                                                        node_num // (space_step ** 3), item))
    #     np.save(x_path, pos)
