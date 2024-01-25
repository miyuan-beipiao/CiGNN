#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: gen_burgers_2d.py 
@time: 2023/02/01
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
'''FD solver for 2d Buergers equation'''
# spatial diff: 4th order laplacian
# temporal diff: O(dt^5) due to RK4
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# from random_fields import GaussianRF
import torch


# torch.manual_seed(66)
# np.random.seed(66)


def apply_laplacian(mat, dx=1.0):
    # dx is inversely proportional to N
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    For more information see
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    # the cell appears 4 times in the formula to compute
    # the total difference
    neigh_mat = -5 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (4 / 3, (-1, 0)),
        (4 / 3, (0, -1)),
        (4 / 3, (0, 1)),
        (4 / 3, (1, 0)),
        (-1 / 12, (-2, 0)),
        (-1 / 12, (0, -2)),
        (-1 / 12, (0, 2)),
        (-1 / 12, (2, 0)),
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dx ** 2


def apply_dx(mat, dx=1.0):
    ''' central diff for dx'''

    # np.roll, axis=0 -> row
    # the total difference
    neigh_mat = -0 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0 / 12, (2, 0)),
        (-8.0 / 12, (1, 0)),
        (8.0 / 12, (-1, 0)),
        (-1.0 / 12, (-2, 0))
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dx


def apply_dy(mat, dy=1.0):
    ''' central diff for dx'''

    # the total difference
    neigh_mat = -0 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0 / 12, (0, 2)),
        (-8.0 / 12, (0, 1)),
        (8.0 / 12, (0, -1)),
        (-1.0 / 12, (0, -2))
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dy


def get_temporal_diff(U, V, R, dx):
    # u and v in (h, w)

    laplace_u = apply_laplacian(U, dx)
    laplace_v = apply_laplacian(V, dx)

    u_x = apply_dx(U, dx)
    v_x = apply_dx(V, dx)

    u_y = apply_dy(U, dx)
    v_y = apply_dy(V, dx)

    # governing equation
    u_t = (1.0 / R) * laplace_u - U * u_x - V * u_y
    v_t = (1.0 / R) * laplace_v - U * v_x - V * v_y

    return u_t, v_t


def update(U0, V0, R=100.0, dt=0.05, dx=1.0):
    u_t, v_t = get_temporal_diff(U0, V0, R, dx)

    U = U0 + dt * u_t
    V = V0 + dt * v_t
    return U, V


def update_rk4(U0, V0, R=100.0, dt=0.05, dx=1.0):
    """Update with Runge-kutta-4 method
       See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    ############# Stage 1 ##############
    # compute the diffusion part of the update

    u_t, v_t = get_temporal_diff(U0, V0, R, dx)

    K1_u = u_t
    K1_v = v_t

    ############# Stage 1 ##############
    U1 = U0 + K1_u * dt / 2.0
    V1 = V0 + K1_v * dt / 2.0

    u_t, v_t = get_temporal_diff(U1, V1, R, dx)

    K2_u = u_t
    K2_v = v_t

    ############# Stage 2 ##############
    U2 = U0 + K2_u * dt / 2.0
    V2 = V0 + K2_v * dt / 2.0

    u_t, v_t = get_temporal_diff(U2, V2, R, dx)

    K3_u = u_t
    K3_v = v_t

    ############# Stage 3 ##############
    U3 = U0 + K3_u * dt
    V3 = V0 + K3_v * dt

    u_t, v_t = get_temporal_diff(U3, V3, R, dx)

    K4_u = u_t
    K4_v = v_t

    # Final solution
    U = U0 + dt * (K1_u + 2 * K2_u + 2 * K3_u + K4_u) / 6.0
    V = V0 + dt * (K1_v + 2 * K2_v + 2 * K3_v + K4_v) / 6.0

    return U, V


def postProcess(output, reso, xmin, xmax, ymin, ymax, num, fig_save_dir):
    ''' num: Number of time step
    '''

    x = np.linspace(0, reso, reso + 1)
    y = np.linspace(0, reso, reso + 1)
    x_star, y_star = np.meshgrid(x, y)
    x_star, y_star = x_star[:-1, :-1], y_star[:-1, :-1]

    u_pred = output[num, 0, :, :]
    v_pred = output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0].scatter(x_star, y_star, c=u_pred, alpha=0.95, edgecolors='none', cmap='RdYlBu',
                       marker='s', s=3, vmin=-1, vmax=1)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('black')
    # cf.cmap.set_over('whitesmoke')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('u-FDM')
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)

    cf = ax[1].scatter(x_star, y_star, c=v_pred, alpha=0.95, edgecolors='none', cmap='RdYlBu',
                       marker='s', s=3, vmin=-1, vmax=1)
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('black')
    # cf.cmap.set_over('whitesmoke')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('v-FDM')
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)

    # plt.draw()
    plt.savefig(fig_save_dir + 'uv_%s.png' % str(num).zfill(4))
    plt.close('all')


def rand_gaussian_ic(Num_a, Num_b, Nx, Ny, plot=True):
    assert Nx == Ny
    x, y = np.linspace(0, x_max, Nx + 1), np.linspace(0, y_max, Ny + 1)
    xx, yy = np.meshgrid(x[:-1], y[:-1])
    # print("xx",xx)
    # print("yy",yy)
    Wx, Wy = [0] * 2
    # np.random.seed(222)  # 5
    # np.random.seed(444)  # 4
    # np.random.seed(666)  # 3
    # np.random.seed(888)  # 2
    # np.random.seed(101010)  # 1
    Ax = np.random.normal(0, 1, size=(Num_a, Num_b))
    Bx = np.random.normal(0, 1, size=(Num_a, Num_b))
    Ay = np.random.normal(0, 1, size=(Num_a, Num_b))
    By = np.random.normal(0, 1, size=(Num_a, Num_b))
    cxy = np.random.normal(-1, 1, size=2)
    for i in range(Num_a):
        for j in range(Num_b):
            # Wx = Wx + Ax[i, j] * np.sin(2 * np.pi * ((i - 00 // 2) * xx + (j - Num_b // 2) * yy)) + Bx[
            #     i, j] * np.cos(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy))
            Wx = Wx + Ax[i, j] * np.sin(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy)) + Bx[
                i, j] * np.cos(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy))
            Wy = Wy + Ay[i, j] * np.sin(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy)) + By[
                i, j] * np.cos(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy))
    # print(cxy)
    # print(cxy[0])
    # print(cxy[1])
    Ux = 2 * Wx / Wx.max() + cxy[0]
    Uy = 2 * Wy / Wy.max() + cxy[1]

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        #
        ax[0].axis('square')
        cf = ax[0].contourf(xx, yy, Ux, levels=101, cmap='jet')
        ax[0].set_xlim([0, 1])
        ax[0].set_ylim([0, 1])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title(r'$u_0$', )
        fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
        #
        ax[1].axis('square')
        cf = ax[1].contourf(xx, yy, Uy, levels=101, cmap='jet')
        ax[1].set_xlim([0, 1])
        ax[1].set_ylim([0, 1])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title(r'$v_0$')
        fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)
        #
        plt.show()

    return Ux, Uy


def gen_data(a_num, b_num, x_num, y_num, t_num, R, dt, dx):
    # get initial condition from random field
    # device = torch.device('cuda')
    # GRF = GaussianRF(2, M, alpha=2, tau=5, device=device)
    # U, V = GRF.sample(2) # U and V have shape of [128, 128]
    # U = U.cpu().numpy()
    # V = V.cpu().numpy()

    # Generate periodic initial condition
    # U, V = rand_gaussian_ic(Num_a=1, Num_b=1, Nx=100, Ny=100, plot=True)
    # print("rand_gaussian_ic")
    U, V = rand_gaussian_ic(Num_a=a_num, Num_b=b_num, Nx=x_num, Ny=y_num, plot=False)
    U, V = U / 3.0, V / 3.0

    U_record = U.copy()[None, ...]
    V_record = V.copy()[None, ...]

    for step in range(t_num):

        # U, V = update(U, V, R, dt, dx)  # [h,w]
        U, V = update_rk4(U, V, R, dt, dx)  # [h,w]

        if (step + 1) % 1 == 0:
            U_record = np.concatenate((U_record, U[None, ...]), axis=0)
            V_record = np.concatenate((V_record, V[None, ...]), axis=0)

    UV = np.concatenate((U_record[None, ...], V_record[None, ...]), axis=0)
    UV = np.transpose(UV, (1, 2, 3, 0))  # [3001,48,48,2]
    UV = UV.reshape(UV.shape[0], -1, UV.shape[-1])

    return UV


if __name__ == '__main__':
    seed = np.linspace(1, 100, 100, dtype=int)
    dataset_name = 'burgers'
    x_max, y_max, t_max = 1.0, 1.0, 1.0
    a_num, b_num = 10, 10
    x_num, y_num, t_num = 50, 50, 1000
    dt = t_max / t_num  # 0.00025 still not converge for FWE, 0.001, 0.002 works for RK4
    dx = x_max / x_num
    dy = y_max / y_num
    node_num = x_num * y_num
    # R = 120.0
    R = 100
    pad_len = [1, 1]

    for i, item in enumerate(seed):
        print(item)
        np.random.seed(item)

        UV = gen_data(a_num, b_num, x_num, y_num, t_num, R, dt, dx)

        u_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/2d_{}_u_t{}_n{}_{}.npy'.format(dataset_name, dataset_name, t_num + 1,
                                                                           node_num, item))
        np.save(u_path, UV)

    for i, item in enumerate(seed):
        np.random.seed(item)
        x = np.linspace(0 - pad_len[0] * dx, x_max + pad_len[1] * dx, x_num + pad_len[0] + pad_len[1] + 1)
        y = np.linspace(0 - pad_len[0] * dy, y_max + pad_len[1] * dy, y_num + pad_len[0] + pad_len[1] + 1)

        # x = np.linspace(0, x_max, x_num + 1)
        # y = np.linspace(0, y_max, y_num + 1)

        x_star, y_star = np.meshgrid(x[:-1], y[:-1])
        # x_star, y_star = np.meshgrid(x, y)

        x_star = np.reshape(x_star, (-1, 1))
        y_star = np.reshape(y_star, (-1, 1))
        pos = np.concatenate((x_star, y_star), axis=1)

        x_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/2d_{}_x_t{}_n{}_{}.npy'.format(dataset_name, dataset_name, t_num + 1,
                                                                           node_num, item))
        np.save(x_path, pos)
