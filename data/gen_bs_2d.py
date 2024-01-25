#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: gen_bs_2d.py 
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
import torch
import xarray as xr
import numpy as np
import einops as eo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import scipy.io
import os

import skfem
from sklearn_extra.cluster import KMedoids

from dataclasses import dataclass, replace
from functools import cached_property
from itertools import combinations
from typing import Optional

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import Delaunay
from bsUtils import default_mesh_config, sample_mesh, OutOfDomainPredicate, MeshConfig, Domain


def gen_domain(ds, filepath, mesh_config):
    # path = 'cmems_mod_blk_phy-tem_my_2.5km_P1D-m_1674021225322.nc'
    # path = 'cmems_mod_blk_phy-cur_my_2.5km_P1D-m_1674011953335.nc'
    # ds = xr.open_mfdataset('*.nc')
    lat, lon = np.array(ds.lat), np.array(ds.lon)
    # Construct all grid points
    x, y = np.meshgrid(lon, lat)
    grid_points = np.vstack((x.ravel(), y.ravel())).T

    # Find points where data is available in all years
    temperature = ds.thetao
    mask = temperature.to_masked_array().mask
    in_domain = np.logical_and.reduce(~mask, axis=0).ravel()

    valid_points = grid_points[in_domain]

    mesh_config = default_mesh_config() if mesh_config is None else mesh_config

    def predicate(tri: Delaunay):
        # Filter out mesh boundary cells that are too acute or contain mostly land
        angle_predicate = mesh_config.angle_predicate(tri)
        ood_predicate = OutOfDomainPredicate(tri, grid_points, in_domain)

        return lambda *args: ood_predicate(*args) or angle_predicate(*args)

    node_indices, domain = sample_mesh(mesh_config, valid_points, predicate)

    # Translate node indices from `valid_points` to `grid_points`
    node_indices = np.nonzero(in_domain)[0][node_indices]

    data = {
        "config": mesh_config,
        "domain": domain,
        "all_points": grid_points,
        "in_domain": in_domain,
        "node_indices": node_indices,
    }

    torch.save(data, filepath)
    return data


if __name__ == '__main__':
    # import numpy as np
    # import networkx as nx
    #
    # G = nx.erdos_renyi_graph(10, 0.15)
    # print(G.gra)

    dataset = 'bs'
    path = '/mnt/miyuan/AI4Physics/Data/{}/'.format(dataset)

    k = 3000

    domain_path = path + f'domain_{k}.pt'

    ds = xr.open_mfdataset(path + '*.nc')
    time = np.array(ds.time)
    velocity_east = ds.uo  # [366,1,215,395]
    velocity_north = ds.vo  # [366,1,215,395]
    temperature = ds.thetao  # [366,1,215,395]
    features = (velocity_east.values, velocity_north.values, temperature.values)
    u = np.stack(features, axis=-1)
    u = eo.rearrange(u, "t 1 y x f -> t (y x) f")

    mesh_config = MeshConfig(k=k, epsilon=10.0, seed=803452658411725)
    if os.path.exists(domain_path):
        domain = torch.load(domain_path)
    else:
        domain = gen_domain(ds, domain_path, mesh_config)
    # "config": mesh_config,
    # "domain": domain,
    # "all_points": grid_points,
    # "in_domain": in_domain,
    # "node_indices": node_indices,

    boundary_idx = domain['domain'].basis.mesh.boundary_nodes()
    node_types = np.zeros((k, 1), dtype='int')
    node_types[boundary_idx] = 6  # boundary
    print(node_types.shape)

    # domain = torch.load(domain_path)
    node_indices = domain["node_indices"]
    u = u[:, node_indices]
    print(u.shape)

    # torch.save(os.path.join(path + "train.pt"), {"t": t[train], "u": u[train]})
    # torch.save(os.path.join(path + "val.pt"), {"t": t[val], "u": u[val]})
    # torch.save(os.path.join(path + "test.pt"), {"t": t[test], "u": u[test]})

    all_points = domain['all_points']
    pos = all_points[node_indices]
    print(pos.shape)

    # mean, std = self._compute_stats(t[train], u[train])
    # np.savez(self.stats_file, mean=mean, std=std)
    mesh = domain['domain'].basis.mesh.t
    print(mesh.shape)

    seed = np.linspace(1993, 2021, 29, dtype=int)
    for i, item in enumerate(seed):
        np.random.seed(item)

        jan_1st = lambda year: np.datetime64(f"{year}-01-01")
        # train = time < jan_1st(item)
        idx = (time >= jan_1st(item)) & (time < jan_1st(item + 1))
        print(idx.shape)
        # test = time >= jan_1st(2019)
        u_path = os.path.join(path + '2d_{}_u_t{}_n{}_{}.npy'.format(dataset, u.shape[0], u.shape[1], item))
        print(u_path)
        np.save(u_path, u[idx])

        x_path = os.path.join(path + '2d_{}_x_t{}_n{}_{}.npy'.format(dataset, u.shape[0], u.shape[1], item))
        print(x_path)
        np.save(x_path, pos)

        mesh_path = os.path.join(path + '2d_{}_mesh_t{}_n{}_{}.npy'.format(dataset, u.shape[0], u.shape[1], item))
        print(mesh_path)
        np.save(mesh_path, mesh)

        type_path = os.path.join(path + '2d_{}_type_t{}_n{}_{}.npy'.format(dataset, u.shape[0], u.shape[1], item))
        print(type_path)
        np.save(type_path, node_types)

        # Convert timestamps to days
        t = time.astype("datetime64[D]").astype(float)
        time_path = os.path.join(path + '2d_{}_time_t{}_n{}_{}.npy'.format(dataset, u.shape[0], u.shape[1], item))
        np.save(time_path, t[idx])
        print(time_path)
