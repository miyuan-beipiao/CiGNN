#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: gen_ns_2d_jax.py 
@time: 2023/03/05
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
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
from datetime import datetime

import torch.cuda
from jax_cfd.base import grids
import jax_cfd.spectral as spectral

import xarray
import time
import tqdm

jax.config.update('jax_platform_name', 'gpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def run(seed):
    # physical parameters
    viscosity = 1e-3
    max_velocity = 7
    grid = grids.Grid((2048, 2048), domain=((0, x_max), (0, y_max)))
    dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)
    print(dt)
    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True  # use anti-aliasing
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.NavierStokes2D(viscosity, grid, smooth=smooth), dt)

    # run the simulation up until time 25.0 but only save 10 frames for visualization
    final_time = 15
    # inner_steps = (final_time // dt) // outer_steps
    inner_steps = 32

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed), grid, max_velocity, 3)

    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)
    _, trajectory = trajectory_fn(vorticity_hat0)
    savenpy(trajectory, seed, grid, dt, outer_steps, inner_steps)


def savenpy(trajectory, seed, grid, dt, outer_steps, inner_steps):
    spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]  # same for x and y
    coords = {
        'time': dt * jnp.arange(outer_steps) * inner_steps,
        'x': spatial_coord,
        'y': spatial_coord,
    }
    trajectory_array = xarray.DataArray(
        jnp.fft.irfftn(trajectory, axes=(1, 2)),
        dims=["time", "x", "y"], coords=coords)
    vorticity_array = trajectory_array.data

    vor = np.expand_dims(vorticity_array, axis=-1)
    vor = vor[:, ::32, ::32, :]

    x_num, y_num = vor.shape[1], vor.shape[2]
    node_num = x_num * y_num
    dx, dy = x_max / x_num, y_max / y_num

    u_path = os.path.join(
        '/mnt/miyuan/AI4Physics/Data/{}/2d_{}_u_t{}_n{}_{}.npy'.format(dataset_name, dataset_name, outer_steps,
                                                                       node_num, seed))
    np.save(u_path, vor.reshape(vor.shape[0], -1, vor.shape[-1]))

    x = np.linspace(0 - pad_len * dx, x_max + pad_len * dx, x_num + 2 * pad_len + 1)
    y = np.linspace(0 - pad_len * dy, y_max + pad_len * dy, y_num + 2 * pad_len + 1)

    x_star, y_star = np.meshgrid(x[:-1], y[:-1])

    x_star = np.reshape(x_star, (-1, 1))
    y_star = np.reshape(y_star, (-1, 1))
    pos = np.concatenate((x_star, y_star), axis=1)

    x_path = os.path.join(
        '/mnt/miyuan/AI4Physics/Data/{}/2d_{}_x_t{}_n{}_{}.npy'.format(dataset_name, dataset_name, outer_steps,
                                                                       node_num, seed))
    np.save(x_path, pos)

    # vor = np.expand_dims(vorticity_array, axis=1)
    # vor = vor[:, :, ::32, ::32]
    # np.save('.data/2000x1x64x64' + str(seed) + '.npy', vor)
    # print(vor.shape)
    print("seed: ", seed, " over")
    # print(len(vor))


if __name__ == '__main__':
    dataset_name = 'jax_ns'
    pad_len = 1
    x_max, y_max = 2 * jnp.pi, 2 * jnp.pi
    outer_steps = 600
    Num = 1
    seed_list = np.linspace(1, Num, Num, dtype=int)
    for i, seed in enumerate(seed_list):
        time1 = time.time()
        dt1 = datetime.now()
        print(
            'seed' + str(seed) + f'start at：{dt1.year}年{dt1.month}月{dt1.day}日 {dt1.hour}:{dt1.minute}:{dt1.second}')
        run(seed)
        # print(torch.cuda.is_available())
        # print(jax.devices()[0])
        dt2 = datetime.now()
        print(f'seed' + str(
            seed) + f'finish at：{dt2.year}年{dt2.month}月{dt2.day}日 {dt2.hour}:{dt2.minute}:{dt2.second}')
        time2 = time.time()
        print("cost: ", time2 - time1)

    # x_num, y_num = 64, 64
    # node_num = x_num * y_num
    # dx, dy = x_max / x_num, y_max / y_num
    # for i, seed in enumerate(seed_list):
    #     x = np.linspace(0 - pad_len * dx, x_max + pad_len * dx, x_num + 2 * pad_len + 1)
    #     y = np.linspace(0 - pad_len * dy, y_max + pad_len * dy, y_num + 2 * pad_len + 1)
    #
    #     x_star, y_star = np.meshgrid(x[:-1], y[:-1])
    #
    #     x_star = np.reshape(x_star, (-1, 1))
    #     y_star = np.reshape(y_star, (-1, 1))
    #     pos = np.concatenate((x_star, y_star), axis=1)
    #
    #     x_path = os.path.join(
    #         '/mnt/miyuan/AI4Physics/Data/{}/2d_jax_{}_x_t{}_n{}_{}.npy'.format(dataset_name, dataset_name, outer_steps,
    #                                                                        node_num, seed))
    #     np.save(x_path, pos)
