#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: gen_ns_2d.py 
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
import torch
import os
import math

from timeit import default_timer

import scipy.io

import torch
import math
import numpy as np
from timeit import default_timer


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device
        # 2,64,3,7
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))  # 7**2=49

        k_max = size // 2  # 32

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size * math.sqrt(2.0) * sigma * (
                    (4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            # [64,64]
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                    (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                    (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff
        # coeff = 100 * coeff
        real = torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real

        return real


# w0: initial vorticity
# f: forcing term
# visc: viscosity (1/Re)
# T: final time
# delta_t: internal time-step for solve (descrease if blow-up)
# record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    # Grid size - must be power of 2
    N = w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N / 2.0)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    # Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Record solution every this number of steps
    record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0
    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in range(steps):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        # Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        # Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        # Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q * w_x + v * w_y)

        # Dealias
        F_h = dealias * F_h

        # Crank-Nicolson update
        w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / (
                1.0 + 0.5 * delta_t * visc * lap)

        # Update real time (used only for recording)
        t += delta_t

        if (j + 1) % record_time == 0:
            # print(j, steps)
            # Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            # Record solution and time
            sol[..., c] = w
            sol_t[c] = t

            c += 1

    return sol, sol_t


# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import scipy.io
import os
import seaborn as sns


def postProcess(w0, i):
    ''' num: Number of time step
    '''
    x = np.linspace(0, 1, x_num + 1)
    y = np.linspace(0, 1, y_num + 1)
    x_star, y_star = np.meshgrid(x[:-1], y[:-1])
    u_pred = w0[0, ...]
    # v_pred = w0[1, num]
    #
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # viridis icefire rainbow
    # cf = sns.heatmap(u_pred, annot=False, ax=ax, cmap="viridis", xticklabels=False, yticklabels=False, vmin=-1, vmax=1,
    #                  cbar=True, cbar_kws={'format': '%.2f',
    #                                       # "shrink": 0.9, vertical, horizontal
    #                                       'fraction': 0.146, 'orientation': 'horizontal', 'pad': 0.01, 'shrink': 0.7})
    # ax.set_aspect('equal')
    #
    # cf = ax.scatter(x_star, y_star, c=u_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=9, vmin=0, vmax=1)
    cf = ax.scatter(x_star, y_star, c=u_pred, cmap='hot', marker='s', s=9)
    ax.axis('square')
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax.set_title(f'u-w{i}')
    # fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    #
    # plt.draw()
    # plt.savefig('./figures/uv_' + str(num).zfill(3) + '.png')
    plt.show()
    # plt.close('all')
    print('over')


device = torch.device('cuda:5')
import numpy as np

# GRF = GaussianRF(2, 100, alpha=2.5, tau=7, device=device)
# w0 = GRF.sample(1).cpu()
# postProcess(w0)
import random

if __name__ == '__main__':
    x_num, y_num = 64, 64  # Resolution 64,64
    node_num = x_num * y_num
    x_max, y_max = 1.28, 1.28  # 1.28, 1.28
    dx, dy = x_max / x_num, y_max / y_num
    Num = 1  # Number of solutions to generate
    # record_steps = 2000  # Number of snapshots from solution
    record_steps = 1000  # Number of snapshots from solution

    # a = torch.zeros(N, s, s)  # Inputs
    # u = torch.zeros(N, s, s, record_steps)  # Solutions

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, x_num, alpha=3, tau=7, device=device)
    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, x_max, x_num + 1, device=device)
    t = t[0:-1]

    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Solve equations in batches (order of magnitude speed-up)
    bsize = 1  # Batch size
    pad_len = 1

    dataset_name = 'ns'
    # T
    Final_Time = 100
    # Final_Time = 200

    # delta_t
    dt = 1e-3
    # dt = 1e-2
    # dt = Final_Time / record_steps
    # v
    viscocity = 1e-3
    # viscocity = 1e-5

    # seed = np.linspace(1, Num, Num, dtype=int)
    seed = np.linspace(1, 1, 1, dtype=int)
    for i, item in enumerate(seed):
        print(item)
        torch.manual_seed(item)
        torch.cuda.manual_seed_all(item)
        np.random.seed(item)
        # random.seed(item)
        t0 = default_timer()
        # Sample random feilds
        w0 = GRF.sample(bsize)
        # print(w0.shape)

        # Solve NS
        sol, sol_t = navier_stokes_2d(w0, f, viscocity, Final_Time, dt, record_steps)
        # print(sol.shape)
        # print(sol_t.shape)
        # 1,S,S,1001 ->1001,s,s,1
        w0 = torch.unsqueeze(w0, dim=-1)
        sol = torch.cat((w0, sol), dim=-1)
        sol = sol.permute(3, 1, 2, 0)
        sol = sol.reshape(sol.shape[0], -1, sol.shape[-1])

        t1 = default_timer()
        print(i + 1, t1 - t0)

        # file_path = '/mnt/miyuan/AI4Physics/Data/{}/2d_{}_N{}_T{}_v{}_dt{}_{}.mat'.format(dataset_name, dataset_name, N,
        #                                                                                   T_str, v_str, dt_str, item)
        # mdict = {'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()}
        # scipy.io.savemat(file_path, mdict)
        u_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/2d_{}_u_t{}_n{}_{}.npy'.format(dataset_name, dataset_name, record_steps + 1,
                                                                           node_num, item))
        # np.save(u_path, sol.cpu().numpy())

        for _ in range(0, sol.shape[0], 100):
            w = sol[_, :, :].cpu().reshape(x_num, y_num, -1)
            w = w.permute(2, 0, 1)
            postProcess(w, _)

    for i, item in enumerate(seed):
        x = np.linspace(0 - pad_len * dx, x_max + pad_len * dx, x_num + 2 * pad_len + 1)
        y = np.linspace(0 - pad_len * dy, y_max + pad_len * dy, y_num + 2 * pad_len + 1)

        # x = np.linspace(0, x_max, x_num + 1)
        # y = np.linspace(0, y_max, y_num + 1)

        x_star, y_star = np.meshgrid(x[:-1], y[:-1])

        x_star = np.reshape(x_star, (-1, 1))
        y_star = np.reshape(y_star, (-1, 1))
        pos = np.concatenate((x_star, y_star), axis=1)

        x_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/2d_{}_x_t{}_n{}_{}.npy'.format(dataset_name, dataset_name, record_steps + 1,
                                                                           node_num, item))
        # np.save(x_path, pos)

    for i, item in enumerate(seed):
        print(item)
        force_path = os.path.join(
            '/mnt/miyuan/AI4Physics/Data/{}/2d_{}_force_t{}_n{}_{}.npy'.format(dataset_name, dataset_name,
                                                                               record_steps + 1,
                                                                               node_num, item))
        # np.save(force_path, f.view(-1, 1).cpu().numpy())
