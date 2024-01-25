#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: cylinderUtils.py 
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

import h5py
import random
import scipy.io
from torch.distributions import normal
import torch.nn as nn
import scipy.sparse as sp
from scipy.spatial import Delaunay
import torch_geometric
import enum

from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T

# from torch.utils.data import Dataset

from numpy.random import normal

from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index, to_scipy_sparse_matrix, \
    mask_to_index
import json
import pygsp
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable
from sklearn.preprocessing import normalize
import torch
from torch_cluster import knn_graph
from scipy.spatial.distance import cdist

import os
import pandas as pd
import numpy as np
import pickle
from glob import glob

from numpy import concatenate

from torch import from_numpy, no_grad

from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RadiusGraph, Cartesian, Distance, Compose, KNNGraph, Delaunay, ToUndirected
from torch_geometric.utils import to_networkx

from networkx import is_weakly_connected
from warnings import warn

from torch.utils.data import IterableDataset
import os, numpy as np
import os.path as osp
import h5py
from torch_geometric.data import Data
import torch
import math
import time

from torch import tensor
from torch.nn.functional import one_hot
from numpy import concatenate, ones, int64, squeeze, diff, tile, where, isin
from numpy.random import normal


def process_node_window(node_data, node_coordinates, node_types,
                        apply_onehot=False,
                        onehot_classes=0,
                        inlet_velocity=None):
    '''
    Concatenates with node features with one-hot encoding of node types

    node_data: time x node x feature array
    node_types: node x one-hot dim array
    node_coordinates: node x dimension array
    apply_onehot: boolean
    onehot_classes: integer
    #inlet_velocity: float; we'll use it as a node feature for now, unless there's a better place to use it

    '''

    num_nodes = node_data.shape[1]

    node_types_ = one_hot(tensor(node_types.astype(int64).flatten()),
                          onehot_classes).numpy() if apply_onehot else node_types

    node_data = [node_data.transpose(1, 0, 2).reshape((num_nodes, -1)), node_coordinates, node_types_]
    if inlet_velocity is not None:
        node_data.append(inlet_velocity * ones((num_nodes, 1)))

    return concatenate(node_data, axis=-1)


def get_sample(dataset, source_node_idx,
               time_idx,
               window_length=5,
               output_type='acceleration',
               noise_sd=None,
               noise_gamma=0.1,
               shift_source=True):
    '''
    Returns position data window (with noise) and output velocity

    source_node_idx: source node indices
    time_idx: current time index
    window_length: input window length
    output_type: output type; one of 'state', 'velocity', or 'acceleration'
    noise_sd: noise standard deviation
    noise_gamma: noise gamma (see noise details in arXiv:2010.03409)
    shift_source: if True, shift input source nodes ahead by one timestep; noise is not included
    '''

    node_data = dataset[time_idx:(time_idx + window_length), :, :].copy()

    # compute output
    if output_type == 'acceleration':
        outputs = dataset[(time_idx + window_length - 2):(time_idx + window_length + 1), :, :]
        outputs = squeeze(diff(diff(outputs, axis=0), axis=0), axis=0)
    elif output_type == 'velocity':
        outputs = dataset[(time_idx + window_length - 1):(time_idx + window_length + 1), :, :]
        outputs = squeeze(diff(outputs, axis=0), axis=0)
    else:
        outputs = dataset[time_idx + window_length, :, :].copy()

    # add noise to position and output
    if noise_sd is not None:
        noise = tile(noise_sd, (node_data.shape[1], 1))
        noise = normal(0, noise)
        # input noise
        node_data[-1] += noise
        # output adjustment
        if output_type == 'acceleration':
            # acceleration_p = x_{t+1} - 2 * x_{t} + x_{t-1} - 2 * noise
            # acceleration_v = x_{t+1} - 2 * x_{t} + x_t{-1} - noise
            # adjustment = 2 * gamma * noise + (1-gamma) * noise = noise * (1 + gamma)
            outputs -= (1 + noise_gamma) * noise
        elif output_type == 'velocity':
            # velocity_adj = x_{t+1} - (x_{t} + noise)
            outputs -= noise
        # else: nothing for state

    # for sources, shift window ahead by 1 timepoint, do not add noise
    # also need to update MeshGraphNets.py update_state and rollout functions (and loss?)
    # to do: add a config parameter to turn this off
    if shift_source:
        node_data[:, source_node_idx, :] = dataset[(time_idx + 1):(time_idx + window_length + 1), source_node_idx,
                                           :].copy()
    else:
        pass  # no noise
        node_data[:, source_node_idx, :] = dataset[time_idx:(time_idx + window_length), source_node_idx, :].copy()

    # note: do not set source output to 0 (outputs[source_node_idx, :] = 0);
    # current version still affects training loss

    return node_data, outputs


# Preprocesses COMSOL cylinder flow simulation output.
def get_comsol_data(filename='cylinder_flow_comsol.csv'):
    D = pd.read_csv(filename)
    x = D['x']
    y = D['y']
    D = D.drop(columns=['x', 'y'])

    X = D.values

    inds = np.arange(0, X.shape[1], 4)
    times = X[:, inds]
    t = times[0]

    inds = np.arange(1, X.shape[1], 4)
    vel_x = X[:, inds]

    inds = np.arange(2, X.shape[1], 4)
    vel_y = X[:, inds]

    inds = np.arange(3, X.shape[1], 4)
    p = X[:, inds]

    return x, y, t, vel_x, vel_y, p

def get_variable_data(filename='variable.csv'):
    # item	u_mean	height	width	radius	cylinder_x	cylinder_y
    D = pd.read_csv(filename)
    x = D['x']
    y = D['y']
    D = D.drop(columns=['x', 'y'])

    X = D.values

    inds = np.arange(0, X.shape[1], 4)
    times = X[:, inds]
    t = times[0]

    inds = np.arange(1, X.shape[1], 4)
    vel_x = X[:, inds]

    inds = np.arange(2, X.shape[1], 4)
    vel_y = X[:, inds]

    inds = np.arange(3, X.shape[1], 4)
    p = X[:, inds]

    return x, y, t, vel_x, vel_y, p


# Preprocesses COMSOL cylinder flow mesh
def get_comsol_edges(node_coordinates, mesh_file='mesh_comsol_output.txt'):
    # Node coordinates and comsol mesh are in a different order
    # Need to re-order the edge list from the mesh file

    def splitFloatLine(line):
        return list(map(float, line.split()[:2]))

    def splitElementLine(line):
        return list(map(int, line.split()[:3]))

    def simplexToEdgeList(simp):
        edges = [(simp[0], simp[1]), (simp[1], simp[2]), (simp[2], simp[0])]
        r_edges = [(e[1], e[0]) for e in edges]
        return edges + r_edges

    with open(mesh_file) as fid:
        mesh = fid.readlines()

    # get nodes
    nodeLine = mesh[4]
    numNodes = int(nodeLine.split()[2])
    mesh_nodes = mesh[10:(10 + numNodes)]
    mesh_nodes = np.array(list(map(splitFloatLine, mesh_nodes)))

    # get mesh elements
    mesh_elements = mesh[11 + numNodes:]
    mesh_elements = np.array(list(map(splitElementLine, mesh_elements)))
    mesh_elements = mesh_elements - 1  # comsol starts from 1 not 0.

    # match mesh and node coordinates
    Y = cdist(mesh_nodes, node_coordinates)
    index = np.argmin(Y, axis=1)

    # get edge list
    simplex = index[mesh_elements]
    A = list(map(simplexToEdgeList, simplex))
    edge_list = [b for sublist in A for b in sublist]
    edge_list = np.unique(edge_list, axis=1)

    return edge_list, simplex


def read_dataset(datafile='cylinder_flow_comsol.csv',
                 meshfile='mesh_comsol_output.txt',
                 center=[0.2, 0.2],
                 R=0.05,
                 output_type='velocity',  # acceleration, velocity, state
                 window_length=1,
                 noise=None,
                 noise_gamma=0.1,
                 apply_onehot=False,
                 boundary_nodes=[1, 5],  # list of integer node types corresponding to boundaries
                 source_nodes=[4],  # list of integer node types corresponding to sources
                 normalize=False,
                 ):
    x, y, t, vel_x, vel_y, p = get_comsol_data(datafile)
    data = []
    for i in range(len(t)):
        data.append([vel_x[:, i], vel_y[:, i], p[:, i]])
    data = np.array(data, dtype=np.float32)
    data = np.rollaxis(data, 2, 1)

    # normalize data;
    # for larger datasets, replace with online normalization
    if normalize:
        mean = np.mean(data, axis=(0, 1))
        std = np.std(data, axis=(0, 1))
        data = (data - mean) / std

    # Find the boundary nodes
    node_coordinates = np.vstack([x, y]).T
    mn, mx = np.min(node_coordinates, axis=0), np.max(node_coordinates, axis=0)
    source_inds = np.where(node_coordinates[:, 0] == mn[0])[0]
    bottom_inds = np.where(node_coordinates[:, 1] == mn[1])[0]
    top_inds = np.where(node_coordinates[:, 1] == mx[1])[0]
    right_inds = np.where(node_coordinates[:, 0] == mx[0])[0]
    non_source_boundary_inds = set(bottom_inds).union(set(right_inds)).union(set(top_inds)).difference(source_inds)

    # cylinder
    center = np.array(center).reshape(1, 2)
    distFromCircleCenter = cdist(node_coordinates, center)

    interior_boundary_inds = np.where(distFromCircleCenter <= R)[0]
    boundary_inds = sorted(list(non_source_boundary_inds.union(interior_boundary_inds)))

    # class NodeType(enum.IntEnum):
    #     NORMAL = 0
    #     OBSTACLE = 1
    #     AIRFOIL = 2
    #     HANDLE = 3
    #     INFLOW = 4
    #     OUTFLOW = 5
    #     WALL_BOUNDARY = 6
    #     SIZE = 9
    node_types = np.zeros((len(x), 1), dtype='int')
    node_types[boundary_inds] = boundary_nodes[0]  # top, bottom, interior
    node_types[right_inds] = boundary_nodes[1]  # right
    node_types[source_inds] = source_nodes[0]  # left or source

    # one-hot
    apply_onehot_ = (np.min(node_types) >= 0)  # check if one-hot encoding can be applied
    onehot_dim = -1 if not apply_onehot_ else (np.max(node_types) + 1)

    if apply_onehot and not apply_onehot_:
        raise Exception(datafile + ': cannot apply one-hot encoding')

    # graph construction
    transforms = [
        Cartesian(norm=False, cat=True),
        Distance(norm=False, cat=True)]

    edge_list, mesh_elements = get_comsol_edges(node_coordinates, meshfile)

    pos = from_numpy(node_coordinates.astype(np.float32))
    edge_index = from_numpy(edge_list.T)

    graph = Data(pos=pos, edge_index=edge_index)

    transforms = Compose(transforms)
    graph = transforms(graph)
    edge_attr = graph.edge_attr

    # #remove other-to-source edges #keep?
    # sources = np.where(np.isin(self.node_types[-1].flatten(), source_nodes))[0]
    # drop_edges = np.isin(graph.edge_index[1].numpy(), sources)
    # graph.edge_index = graph.edge_index[:, ~drop_edges].contiguous()
    # graph.edge_attr = graph.edge_attr[~drop_edges].contiguous()

    # if not is_weakly_connected(to_networkx(graph)):
    #     warn(datafile + ': disconnected graph')

    # dataset_length = np.array(data.shape[0] - window_length, dtype=np.int64)
    # output_dim = data.shape[-1]

    # update function; update momentum based on predicted change ('velocity'), predict pressure
    # def update_function(mgn_output_np, output_type,
    #                     current_state=None, previous_state=None,
    #                     source_data=None):
    #
    #     num_states = current_state.shape[-1]
    #
    #     with no_grad():
    #         if output_type == 'acceleration':
    #             assert current_state is not None
    #             assert previous_state is not None
    #             next_state = np.concatentate([2 * current_state - previous_state, np.zeros((len(current_state), 1))],
    #                                          axis=1) + mgn_output_np
    #         elif output_type == 'velocity':
    #             assert current_state is not None
    #             next_state = np.concatenate([current_state, np.zeros((len(current_state), 1))], axis=1) + mgn_output_np
    #         else:  # state
    #             next_state = mgn_output_np.copy()
    #
    #         if type(source_data) is dict:
    #             for key in source_data:
    #                 next_state[key, :num_states] = source_data[key]
    #         elif type(source_data) is tuple:
    #             next_state[source_data[0], :num_states] = source_data[1]
    #         # else: warning?
    #
    #     return next_state

    # data = node_data.transpose(1, 0, 2).reshape((num_nodes, -1))

    return from_numpy(data), pos, edge_index, from_numpy(node_types), edge_attr, mesh_elements


# from mgdo.utils.imageUtils import get_lim
# from matplotlib import tri as mtri
# import matplotlib.pyplot as plt
#
# def subplot_ax_tri(ax, triangulation, x_star, y_star, c, title, fig, u_min, u_max, label_title):
#     ax.set_aspect('equal')
#     # ax.set_title(title, fontsize=20)
#     ax.set_title(title)
#     ax.set_axis_off()
#
#     cf = ax.tripcolor(triangulation, c.flatten(), cmap='rainbow', zorder=1, vmin=u_min, vmax=u_max)
#     cf_cb = fig.colorbar(cf, orientation='horizontal', ax=ax, fraction=0.046, pad=0.1, shrink=0.7, format='%.2f',
#                          ticks=[u_min, (u_max + u_min) / 2, u_max])  # vertical
#     cf_cb.set_label(label=label_title, loc='center')
#
#     # ax.triplot(triangulation, 'ko-', ms=0.05, lw=0.05)
#     ax.triplot(triangulation, linewidth=0.1, color='black')
#
#     # cbar = ax.collections[0].colorbar
#     # cbar.ax.tick_params(labelsize=20)
#
# # 生成3行3列png
# def plotR3C3_cf(feature_dim, position, output, truth_data, num):
#     x_star, y_star = position[:, 0], position[:, 1]
#
#     # predicted_v = np.linalg.norm(predicted, axis=-1)
#     # for ax in axes:
#     #     ax.cla()
#     #     ax.triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)
#     #
#     # handle1 = axes[0].tripcolor(triang, target_v, vmax=v_max, vmin=v_min, edgecolors='k')
#     # axes[1].tripcolor(triang, predicted_v, vmax=v_max, vmin=v_min, edgecolors='k')
#     # # handle2 = axes[2].tripcolor(triang, diff, vmax=1, vmin=0)
#     #
#     # axes[0].set_title('Target\nTime @ %.2f s' % (step * 0.01))
#     # axes[1].set_title('Prediction\nTime @ %.2f s' % (step * 0.01))
#     # # axes[2].set_title('Difference\nTime @ %.2f s'%(step*0.01))
#     # colorbar1 = fig.colorbar(handle1, ax=[axes[0], axes[1]])
#
#     # (3001, 100* 100, 2)
#     output = output[..., 0:feature_dim]
#     truth_data = truth_data[..., 0:feature_dim]
#     # [1000,1,1000,2]
#     output = output.permute(0, 2, 1)
#     truth_data = truth_data.permute(0, 2, 1)
#
#     # if args.dataset_name == 'cf':
#     #     face_path = os.path.join('{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
#     #                                                                     args.dataset_name, args.all_time_steps,
#     #                                                                     args.mesh_node, item))
#     #     face = np.load(face_path)
#     #     face = torch.from_numpy(face).float()
#
#     # 不指定三角形，根据坐标按照Delaunay 三角剖分确定三角形
#     triangulation = mtri.Triangulation(x_star, y_star)
#     # 指定三角形，comsol生成的mesh elements
#     # triangulation = mtri.Triangulation(x_star, y_star, mesh_elements)
#     # triangulation = face
#
#     # Padding x,y axis due to periodic boundary condition
#     u_star = truth_data[num, 0:1, ...]
#     u_pred = output[num, 0:1, ...]
#     v_star = truth_data[num, 1:2, ...]
#     v_pred = output[num, 1:2, ...]
#     p_star = truth_data[num, 2:3, ...]
#     p_pred = output[num, 2:3, ...]
#
#     # if is_3D:
#     #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
#     # else:
#     #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7))
#     # fig.subplots_adjust(hspace=0.3, wspace=0.3)
#     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 9))
#
#     list = [u_star]
#     u_min, u_max = get_lim(torch.cat(list, dim=0))
#     # 未加噪音数据前的真实模拟数据 U
#     subplot_ax_tri(ax[0, 0], triangulation, x_star, y_star, u_star, 'u (Groud Truth.)' + str(num), fig, u_min,
#                    u_max, label_title='velocity')
#
#     list = [u_pred]
#     u_min, u_max = get_lim(torch.cat(list, dim=0))
#     # 模型预测数据 U
#     subplot_ax_tri(ax[0, 1], triangulation, x_star, y_star, u_pred, 'u (Pred.)' + str(num), fig, u_min, u_max,
#                    label_title='velocity')
#
#     list = [torch.abs(u_pred - u_star)]
#     u_min, u_max = get_lim(torch.cat(list, dim=0))
#     # 误差数据 U
#     subplot_ax_tri(ax[0, 2], triangulation, x_star, y_star, torch.abs(u_pred - u_star), 'u (Abs Err.)' + str(num), fig,
#                     u_min, u_max, label_title='velocity')
#
#     list = [v_star]
#     v_min, v_max = get_lim(torch.cat(list, dim=0))
#     # 未加噪音数据前的真实模拟数据 V
#     subplot_ax_tri(ax[1, 0], triangulation, x_star, y_star, v_star, 'v (Groud Truth.)' + str(num), fig, v_min,
#                    v_max, label_title='velocity')
#
#     list = [v_pred]
#     v_min, v_max = get_lim(torch.cat(list, dim=0))
#     # 模型预测数据 V
#     subplot_ax_tri(ax[1, 1], triangulation, x_star, y_star, v_pred, 'v (Pred.)' + str(num), fig, v_min, v_max,
#                    label_title='velocity')
#
#     list = [torch.abs(v_pred - v_star)]
#     v_min, v_max = get_lim(torch.cat(list, dim=0))
#     # 误差数据 V
#     subplot_ax_tri(ax[1, 2], triangulation, x_star, y_star, torch.abs(v_pred - v_star), 'v (Abs Err.)' + str(num), fig, v_min, v_max, label_title='velocity')
#
#     list = [p_star]
#     p_min, p_max = get_lim(torch.cat(list, dim=0))
#     # 未加噪音数据前的真实模拟数据 P
#     subplot_ax_tri(ax[2, 0], triangulation, x_star, y_star, p_star, 'p (Groud Truth.)' + str(num), fig, p_min,
#                    p_max, label_title='pressure')
#
#     list = [p_pred]
#     p_min, p_max = get_lim(torch.cat(list, dim=0))
#     # 模型预测数据  P
#     subplot_ax_tri(ax[2, 1], triangulation, x_star, y_star, p_pred, 'p (Pred.)' + str(num), fig, p_min, p_max,
#                    label_title='pressure')
#
#     list = [torch.abs(p_pred - p_star)]
#     p_min, p_max = get_lim(torch.cat(list, dim=0))
#     # 误差数据 P
#     subplot_ax_tri(ax[2, 2], triangulation, x_star, y_star, torch.abs(p_pred - p_star), 'p (Abs Err.)' + str(num), fig,
#                    p_min, p_max, label_title='pressure')
#
#     # 存储图片
#     plt.show()
#     plt.close('all')

# datafile='/mnt/miyuan/AI4Physics/Data/cf/cylinder_flow_comsol_1.csv'
# meshfile='/mnt/miyuan/AI4Physics/Data/cf/mesh_comsol_output_1.txt'
# data, pos, edge_index, node_types, edge_attr,mesh_elements = read_dataset(datafile=datafile,meshfile=meshfile)
# print(data.shape, pos.shape, edge_index.shape, node_types.shape, edge_attr.shape)
# plotR3C3_cf(3, pos, data, data, 0)
# import h5py
# train_file_path = '/mnt/miyuan/AI4Physics/Data/cylinder_flow/datapkls/'+'train.h5'
# valid_file_path = '/mnt/miyuan/AI4Physics/Data/cylinder_flow/datapkls/'+'valid.h5'
# test_file_path = '/mnt/miyuan/AI4Physics/Data/cylinder_flow/datapkls/'+'test.h5'
# train_data = h5py.File(train_file_path)
# valid_data = h5py.File(valid_file_path)
# test_data = h5py.File(test_file_path)
# data_keys = list(train_data.keys())
# files = {k: train_data[k] for k in data_keys}
# print(files['0'].keys()) # ['cells', 'node_type', 'pos', 'pressure', 'velocity']
# print(files['0']['cells'].shape)
# datas = []
# for f in files:
#     for k in data_keys:
#         if k in ["velocity", "pressure"]:
#             r = np.array((f[k], f[k]), dtype=np.float32)
#         else:
#             r = data[k][selected_frame]
#             if k in ["node_type", "cells"]:
#                 r = r.astype(np.int32)
#         datas.append(r)
#     datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))
