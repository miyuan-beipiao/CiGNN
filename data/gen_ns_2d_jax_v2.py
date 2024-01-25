# -*- coding: utf-8 -*-
"""
@Time ： 2023/03/26
@Auth ： qiwang
@File ：solver_2dns.py
@IDE ：pycharm
@Contact: qi_wang@ruc.edu.cn
@Motto：IWMW(I Work Model Work)
"""

import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
from datetime import datetime
from jax_cfd.base import grids
import jax_cfd.spectral as spectral
from jax_cfd.spectral.utils import vorticity_to_velocity
import math
import seaborn
# from tools import get_lim
# from tools import plotgif2
# from tools import merge2Group
import xarray
import time
# from tools import mergeData
from jax.config import config

config.update("jax_enable_x64", True)
import torch
# from fd_cal_p import GenerateP
import os

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def vorticity2velocity(vorticity):
    vorticity0hat = jnp.fft.rfftn(vorticity)
    vxhat, vyhat = velocity_solve(vorticity0hat)
    vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)
    vx_array = xarray.DataArray(vx).data
    vy_array = xarray.DataArray(vy).data
    cur_uv = np.concatenate((vx_array[np.newaxis, np.newaxis, :, :], vy_array[np.newaxis, np.newaxis, :, :]), axis=1)
    return cur_uv


def get_V0type():
    max_velocity = 7
    grid = grids.Grid((2048, 2048), domain=((0, 1), (0, 1)))
    smooth = True  # use anti-aliasing
    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed=1), grid, max_velocity, 4)
    return v0


def run(seed):
    # physical parameters
    viscosity = 0.002
    max_velocity = 7
    grid = grids.Grid((2048, 2048), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    velocity_solve = vorticity_to_velocity(grid)
    dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)  # 0.00175
    print(dt)
    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True  # use anti-aliasing
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.NavierStokes2D(viscosity, grid, smooth=smooth), dt)

    # run the simulation up until time 25.0 but only save 10 frames for visualization
    # final_time = 1
    outer_steps = 160
    # inner_steps = (final_time // dt)//10
    inner_steps = 32

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed), grid, max_velocity, 4)

    for i in range(sub):
        time1 = time.time()
        if i == 0:
            lastv0 = v0
            vorticity0 = cfd.finite_differences.curl_2d(lastv0).data
            # test_garray = grids.GridArray(xarray.DataArray(v0[0].data).data,lastv0[0].offset, grid)
            # test_gv = grids.GridVariable(test_garray,v0[0].bc)
            vorticity2velocity(vorticity0)
            lastvorticity_hat0 = jnp.fft.rfftn(vorticity0)
        _, trajectory = trajectory_fn(lastvorticity_hat0)
        savenpy(trajectory, seed, i, grid, dt, outer_steps, inner_steps)
        lastvorticity_hat0 = trajectory[-1]
        del trajectory
        time2 = time.time()
        print("***sub cost:****", time2 - time1)


def run2(seed):  # not save vorticity,directy save uvpw
    # physical parameters

    max_velocity = 7
    grid = grids.Grid((2048, 2048), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    # velocity_solve = vorticity_to_velocity(grid)
    dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)  # 0.0002191401125550916 or 0.00175
    print(dt)
    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True  # use anti-aliasing
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.NavierStokes2D(viscosity, grid, smooth=smooth), dt)

    # run the simulation up until time 25.0 but only save 10 frames for visualization
    # final_time = 1
    outer_steps = 200
    # inner_steps = (final_time // dt)//10
    inner_steps = 32

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed), grid, max_velocity, 4)

    for i in range(sub):  # 试一下gpu最大支持多少，如果是4800，每个是600
        time0 = time.time()
        print("*****numerical solver [start] seed{0}_{1}*****".format(seed, i))
        if i == 0:
            lastv0 = v0
            vorticity0 = cfd.finite_differences.curl_2d(lastv0).data
            lastvorticity_hat0 = jnp.fft.rfftn(vorticity0)
        time1 = time.time()
        _, trajectory = trajectory_fn(lastvorticity_hat0)
        time2 = time.time()
        print("*****numerical solver [end] seed{0}_{1}***** cost{2}".format(seed, i, time2 - time1))
        vor_array = simpleTrans(trajectory, seed, i, grid, dt, outer_steps, inner_steps)

        time1 = time.time()
        Vorticitysave(vor_array, seed, i)
        time2 = time.time()
        lastvorticity_hat0 = trajectory[-1]
        del trajectory
        time3 = time.time()
        print("***this sub cost:****", time3 - time0)
        # lastv0 = np.concatenate((trajectory[0].data[-1:-2,...],trajectory[1].data[-1:-2,...]),axis=1)


def simpleTrans(trajectory, seed, i, grid, dt, outer_steps, inner_steps):
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

    vor = np.expand_dims(vorticity_array, axis=1)
    return vor


def savenpy(trajectory, seed, i, grid, dt, outer_steps, inner_steps):
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

    vor = np.expand_dims(vorticity_array, axis=1)

    # vor = vor[:, :, ::32, ::32]

    # np.save('uv_pnas_seed20.npy', uv)
    # np.save('uv_pnas_seed30.npy', uv)
    # np.save('uv_pnas_seed40.npy', uv)
    # np.save('uv_pnas_seed50.npy', uv)

    # np.save('uv_pnas_seed60.npy', uv)
    # np.save('uv_pnas_seed70.npy', uv)
    # np.save('uv_pnas_seed70.npy', uv)
    # np.save('uv_pnas_256re1000seed'+str(seed)+'.npy', uv)
    np.save(sub_save_path + sub_save_name + str(seed) + '-' + str(i) + '.npy', vor)

    # np.save('../../data/decaying_re500_pi__256x256_64x64dt0_001753_seed20_' + str(seed) + '.npy', vor)
    print(vor.shape)
    # np.save('../../data/demo.npy', uv)
    print("seed: ", seed, " over")
    print(len(vor))
    print("seed{0}_{1} over".format(seed, i))


def loaddata(num):
    path = uvpw_save_path
    # filename = "decaying_re500_pi__2048_256_dt0_00036523_seed5_"
    # filename = "decaying_re500_pi__2048x2048_256x256dt0_00036523x2_seed20_"
    # filename = "decaying_re500_pi__2048x2048_256x256dt0_00036523x32_seed20_"
    # filename = "fvm_decaying_re500_pi__35x2560x2048_64x64dt0_00036523"
    # filename = "fvm_forcing_35x2560x2048_64x64dt0_00021914"
    # filename = "fvm_decaying_re500_pi__320timestepx4_2560x2048_64x64dt0_00036523"
    # filename = "fvm_decaying_re500_pi_37x2560x2x64x64dt0_00036523"
    # filename = "fvm_decaying_re500_pi__320timestepx15_2560x2048_64x64dt0_00036523"
    # filename = "spetraldecaying_re500_pi__vorticity_160x2048x2048_64x64dt0_00021914"
    filename = all_uvpw_save_name
    data_all = []

    cur_data = torch.from_numpy(np.load(path + filename + ".npy"))

    trainStage = True
    # if trainStage:
    #     cur_data = cur_data[:num,:timesteps,...]
    # ifplot = False
    # if ifplot:
    #     name = "tesfddecaying_re500_pi__w"
    #     for i in range(num):
    #         print("seed", i)
    #         truth = cur_data[i]
    #         u_min, u_max = get_lim(truth[:, 0:1, ...])
    #         v_min, v_max = get_lim(truth[:, 1:2, ...])
    #         # plotgif2(u_min, u_max, name + "u", output[:, 0:1, ...], rate=64)
    #         # plotgif2(v_min, v_max, name + "v", output[:, 1:2, ...], rate=64)
    #         plotgif2(u_min, u_max, name + str(i) + "u", truth[:, 0:1, ...], rate=64)
    #         plotgif2(v_min, v_max, name + str(i) + "v", truth[:, 1:2, ...], rate=64)

    return cur_data


def Vorticitysave(vorticity, seed, i):
    print("vorticity shape", vorticity.shape)
    np.save(sub_uvpw_save_path + sub_uvpw_save_name + str(seed) + '-' + str(i) + '.npy',
            vorticity[:, :, ::down, ::down])
    print("seed: ", seed, " over")
    print("seed{0}_{1} over".format(seed, i))
    # ifplot = False
    # if ifplot:
    #     cur_uvp = UVPW
    #     pnas = torch.from_numpy(cur_uvp)
    #     pnas_u = pnas[:, 0:1, ...]
    #     pnas_v = pnas[:, 1:2, ...]
    #     pnas_p = pnas[:, 2:3, ...]
    #     pnas_w = pnas[:, 3:4, ...]
    #     u_min, u_max = get_lim(pnas_u)
    #     v_min, v_max = get_lim(pnas_v)
    #     p_min, p_max = get_lim(pnas_p)
    #     w_min, w_max = get_lim(pnas_w)
    # 
    #     name = "rp_vorticity4uvpw_"
    # 
    #     plotgif2(u_min, u_max, name + "u", pnas_u, rate)
    #     plotgif2(v_min, v_max, name + "v", pnas_v, rate)
    #     plotgif2(p_min, p_max, name + "p", pnas_p, rate)
    #     plotgif2(w_min, w_max, name + "w", pnas_w, rate)


# def plot():
#     pnas = loaddata(1)
#     pnas_u = pnas[0, :, 0:1, ...]
#     pnas_v = pnas[0, :, 1:2, ...]
#     # res = vor(pnas[0]).cpu()
#     res = pnas[0][:, :, ::8, ::8]
#     # u_min, u_max = get_lim(pnas[0,:, 0:1, ...])
#     # v_min, v_max = get_lim(pnas[0,:, 1:2, ...])
#     vor_min, vor_max = get_lim(res)
#     name = "2"
#     rate = 200
#     # plotgif2(u_min, u_max, name + "u", pnas[0,:, 0:1, ...].cpu(), rate)
#     # plotgif2(v_min, v_max, name + "v", pnas[0,:, 1:2, ...].cpu(), rate)
#     plotgif2(vor_min, vor_max, name + "vorticitydouble", res, rate)


def mergeDataandcreateuvp(sub_save_path, uvpw_save_path, name, uvpw_save_name, start, end, sub, group):
    data_all = []
    for i in range(start, end + 1):
        data = []
        for j in range(sub):
            # cur_UV = np.load('./data/decaying_re500_pi__2048x2048_256x256dt0_00036523x32_seed20_'+str(i)+'.npy')
            cur_UV = np.load(
                sub_save_path + "doublespetraldecaying_re500_pi__2048x2048dt0_00021914_seed32_" + str(i) + "-" + str(
                    j) + '.npy')
            print("load:" + str(i) + "-" + str(j))
            data.extend(cur_UV)
        data_all.append(np.array(data))
    vorticity_array = np.array(data_all)
    # uvpw = transformVorticity(vorticity_array,group)
    # np.save(uvpw_save_path+uvpw_save_name+str(group),uvpw)
    # print("******save success*****:"+uvpw_save_path+uvpw_save_name+str(group)+"npy")


if __name__ == '__main__':
    # for seed in [10,20,30,40]:
    sub_uvpw_save_path = "/mnt/miyuan/AI4Physics/Data/jax_ns/"
    all_uvpw_save_path = "/mnt/miyuan/AI4Physics/Data/jax_ns/"
    group_uvpw_save_path = "/mnt/miyuan/AI4Physics/Data/jax_ns/"

    sub_uvpw_save_name = "doublespectraldecaying_re1000_2pi_128x128dt0_0002191401125550916_w_inner32seed32_"
    all_uvpw_save_name = "doublespectraldecaying_re1000_2pi_128x128dt0_0002191401125550916_inner32w"
    group_uvpw_save_name = "doublespectraldecaying_re1000_2pi_128x128dt0_0002191401125550916_w_inner32group_"
    sub_save_path = "/mnt/miyuan/AI4Physics/Data/jax_ns/test/"

    uvpw_save_path = "/mnt/miyuan/AI4Physics/Data/jax_ns/test/"
    sub_save_name = "doublespectraldecaying_re1000_2pi_4800x128x128dt0_0002191401125550916_w_inner32seed32_"
    # data_save_name = "doublespetraldecaying_re500_pi__2048x2048dt0_00021914" 
    # data_save_name = "NMI_uvpw_4800x4x256x256"    

    viscosity = 0.001
    sub = 1
    rate = 100
    down = 16  # 256
    grid = grids.Grid((2048, 2048), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    velocity_solve = vorticity_to_velocity(grid)
    # solv_p = GenerateP(vis=viscosity).cuda()
    start = 1
    end = 1
    # time1 = time.time()
    # mergeDataandcreateuvp(sub_save_path, uvpw_save_path, data_save_name, data_save_name, start, end, sub, group=1)
    # time2 = time.time()
    # print("merge&create cost: ",time2-time1)

    """gen"""
    for seed in range(start, end + 1):
        time1 = time.time()
        dt1 = datetime.now()
        # print('今天是：{dt.year}年{dt.month}月{dt.day}日 {dt.hour}:{dt.minute}:{dt.second}')
        print(
            'seed' + str(seed) + f'start at：{dt1.year}年{dt1.month}月{dt1.day}日 {dt1.hour}:{dt1.minute}:{dt1.second}')
        run2(seed)
        dt2 = datetime.now()
        print(f'seed' + str(seed) + f'finish at：{dt2.year}年{dt2.month}月{dt2.day}日 {dt2.hour}:{dt2.minute}:{dt2.second}')


    """比较"""
    # cur_UV_1 = np.load(sub_save_path + "doublespectraldecaying_re1000_2pi_128x128dt0_0002191401125550916_w_inner32seed32_1-0.npy")
    # cur_UV_2 = np.load(sub_save_path + "doublespectraldecaying_re1000_2pi_128x128dt0_0002191401125550916_w_inner32seed32_1-1.npy")
    # cur_UV_3 = np.load(sub_save_path + "doublespectraldecaying_re1000_2pi_128x128dt0_0002191401125550916_w_inner32seed32_2-0.npy")
    #
    # diff_cur = cur_UV_3-np.concatenate((cur_UV_1,cur_UV_2),axis=0)
    # print(diff_cur)

    """比较"""
    # time2 = time.time()
    # print("cost: ",time2-time1)
    # mergeData(sub_save_path,data_save_path,sub_save_name, savename=data_save_name, start=1,end=4, sub=sub)

    # mergeData(sub_uvpw_save_path, all_uvpw_save_path, sub_uvpw_save_name, all_uvpw_save_name, start, end, sub)#all
    # merge2Group(sub_uvpw_save_path, group_uvpw_save_path, sub_uvpw_save_name, group_uvpw_save_name, start, end, sub=sub)
    #
    # time1 = time.time()
    # transformVorticity(group=1)#groupID 每组4个IC
    # time2 = time.time()
    # print("transoform cost:",time2-time1)
