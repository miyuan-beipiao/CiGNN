#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: dataUtils.py 
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
import h5py
import random
import scipy.io
from torch.distributions import normal
import torch.nn as nn
import scipy.sparse as sp
from scipy.spatial import Delaunay
import torch_geometric
import enum
from decimal import Decimal

from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T

# from torch.utils.data import Dataset

from numpy.random import normal

from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index, to_scipy_sparse_matrix, \
    mask_to_index, to_undirected
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
import time

from utils.imageUtils import subplot_mesh
from configs import bs_args_2d, gs_args_3d, burgers_args_2d, ns_args_2d, cf_args_2d, ns_jax_args_2d


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    PERIODIC_BOUNDARY = 7
    SIZE = 9


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    x, edge_index, edge_attr, pos, v = graph.x, graph.edge_index, graph.edge_attr, graph.pos, graph.v

    ret = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, v=v)

    return ret


class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10 ** 6, std_epsilon=1e-8, name='Normalizer', device=None):
        super(Normalizer, self).__init__()
        self.name = name
        self._max_accumulations = max_accumulations

        # self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device='cpu')
        # self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device='cpu')
        # self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device='cpu')
        # self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device='cpu')
        # self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device='cpu')

        # # (2)使用register_buffer()定义一组参数
        # self.register_buffer('reg_buf', torch.randn(1, 2))
        # # (3)使用register_parameter()定义一组参数
        # self.register_parameter('reg_param', nn.Parameter(torch.randn(1, 2)))
        # # (4)使用python类的属性方式定义一组变量
        # self.param_attr = torch.randn(1, 2)
        # net.modules()
        # net.parameters()
        # net.buffers()
        self.register_buffer('_std_epsilon', torch.tensor(std_epsilon, dtype=torch.float32, device=device))
        self.register_buffer('_acc_count', torch.tensor(0, dtype=torch.float32, device=device))
        self.register_buffer('_num_accumulations', torch.tensor(0, dtype=torch.float32, device=device))
        self.register_buffer('_acc_sum', torch.ones(size, dtype=torch.float32, device=device))
        self.register_buffer('_acc_sum_squared', torch.ones(size, dtype=torch.float32, device=device))

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self._num_accumulations < self._max_accumulations:
            # stop accumulating after a million updates, to prevent accuracy issues
            self._accumulate(batched_data.detach())
        return (batched_data - self._mean().to(batched_data.device)) / self._std_with_epsilon().to(batched_data.device)

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon().to(normalized_batch_data.device) + self._mean().to(
            normalized_batch_data.device)

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = torch.tensor(batched_data.shape[0]).float()
        data_sum = torch.sum(batched_data, dim=0)
        squared_data_sum = torch.sum(batched_data ** 2, dim=0)

        self._acc_sum += data_sum.to(self._acc_sum.device)
        self._acc_sum_squared += squared_data_sum.to(self._acc_sum_squared.device)
        self._acc_count += count.to(self._acc_count.device)
        self._num_accumulations += 1

    def _mean(self):
        # safe_count = torch.maximum(self._acc_count,
        #                            torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.).float())
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        # safe_count = torch.maximum(self._acc_count,
        #                            torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.).float())

        # std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        std = self._acc_sum_squared / safe_count - self._mean() ** 2
        std[std < self._std_epsilon] = self._std_epsilon
        std = torch.sqrt(std)

        # std[std < self._std_epsilon] = self._std_epsilon
        # return std
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):
        dict = {'_max_accumulations': self._max_accumulations, '_std_epsilon': self._std_epsilon,
                '_acc_count': self._acc_count, '_num_accumulations': self._num_accumulations, '_acc_sum': self._acc_sum,
                '_acc_sum_squared': self._acc_sum_squared, 'name': self.name}
        return dict


# class Normalizer(nn.Module):
#     def __init__(self, size, max_accumulations=10 ** 6, std_epsilon=1e-8, name='Normalizer'):
#         super(Normalizer, self).__init__()
#         self.name = name
#         self._max_accumulations = max_accumulations
#         # self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device='cpu')
#         # self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device='cpu')
#         # self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device='cpu')
#         # self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device='cpu')
#         # self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device='cpu')
#
#         self._std_epsilon = torch.tensor(std_epsilon, requires_grad=False)
#         self._acc_count = torch.tensor(0, requires_grad=False)
#         self._num_accumulations = torch.tensor(0, requires_grad=False)
#         self._acc_sum = torch.zeros((1, size), requires_grad=False)
#         self._acc_sum_squared = torch.zeros((1, size), requires_grad=False)
#
#     def forward(self, batched_data, accumulate=True):
#         device = batched_data.device
#         batched_data = batched_data.detach().cpu()
#
#         """Normalizes input data and accumulates statistics."""
#         if accumulate:
#             # stop accumulating after a million updates, to prevent accuracy issues
#             if self._num_accumulations < self._max_accumulations:
#                 self._accumulate(batched_data)
#         result = (batched_data - self._mean()) / self._std_with_epsilon()
#         return result.to(device)
#
#     def inverse(self, normalized_batch_data):
#         device = normalized_batch_data.device
#         normalized_batch_data = normalized_batch_data.detach().cpu()
#         """Inverse transformation of the normalizer."""
#         result = normalized_batch_data * self._std_with_epsilon() + self._mean()
#         return result.to(device)
#
#     def _accumulate(self, batched_data):
#         """Function to perform the accumulation of the batch_data statistics."""
#         count = batched_data.shape[0]
#         data_sum = torch.sum(batched_data, axis=0, keepdims=True)
#         squared_data_sum = torch.sum(batched_data ** 2, axis=0, keepdims=True)
#
#         self._acc_sum += data_sum
#         self._acc_sum_squared += squared_data_sum
#         self._acc_count += count
#         self._num_accumulations += 1
#
#     def _mean(self,batched_data):
#         # safe_count = torch.maximum(self._acc_count,
#         #                            torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
#         safe_count = torch.maximum(self._acc_count, torch.tensor(1.0))
#         return self._acc_sum / safe_count
#
#     def _std_with_epsilon(self,batched_data):
#         # safe_count = torch.maximum(self._acc_count,
#         #                            torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
#         safe_count = torch.maximum(self._acc_count, torch.tensor(1.0))
#
#         std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
#         return torch.maximum(std, self._std_epsilon)
#
#     def get_variable(self):
#         dict = {'_max_accumulations': self._max_accumulations, '_std_epsilon': self._std_epsilon,
#                 '_acc_count': self._acc_count, '_num_accumulations': self._num_accumulations, '_acc_sum': self._acc_sum,
#                 '_acc_sum_squared': self._acc_sum_squared, 'name': self.name}
#         return dict

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        # return x * torch.nn.Sigmoid()(x * beta)
        # return x * nn.Sigmoid(x)
        return x * torch.sigmoid(x)


# def add_velocity_noise(x, type, args, noise_std=2e-2):
#     type = type[:, :args.input_step, :, 0]
#     noise = torch.normal(std=noise_std, mean=0.0, size=x.shape).to(x.device)
#     mask = type != NodeType.NORMAL
#     noise[mask] = 0
#     return x + noise

def add_velocity_noise(data, args, noise_std=2e-2):
    # data_last = copy_geometric_data(data)
    # type = data.type.reshape(-1, args.input_step, 1)[:, :args.input_step, 0]
    type = data.type[..., 0]
    noise = torch.normal(std=noise_std, mean=0.0, size=data.x.reshape(-1, args.input_step, args.feature_dim).shape).to(
        data.x.device)
    mask = type != NodeType.NORMAL
    noise[mask] = 0
    data.x = (data.x.reshape(-1, args.input_step, args.feature_dim) + noise).reshape(-1,
                                                                                     args.input_step * args.feature_dim)

    # return Data(x=x, edge_attr=data_last.edge_attr, edge_index=data_last.edge_index, pos=data_last.pos, y=data_last.y,
    #             v=data_last.v)
    return data


# 增加噪声 10%
# by 师兄
def add_noise(ground_truth, args):
    assert ground_truth.shape[3] == 2
    uv = [ground_truth[:args.train_time_steps, :, :, 0:1], ground_truth[:args.train_time_steps, :, :, 1:2]]
    uv_noi = []
    torch.manual_seed(66)
    for truth in uv:
        n_distr = normal.Normal(0.0, 1.0)
        R = n_distr.sample(truth.shape)
        std_R = torch.std(R)  # std of samples
        std_T = torch.std(truth)
        noise = R * std_T / std_R * args.pec
        uv_noi.append(truth + noise)
    return torch.cat(uv_noi, dim=3)


def set_random_seed(seed):
    """
        设置随机数种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    # 为了确定算法，保证得到一样的结果。
    # torch.backends.cudnn.deterministic = True
    # 使用非确定性算法
    # torch.backends.cudnn.enabled = False
    # 是否自动加速。
    # torch.backends.cudnn.benchmark = False


def check_path(args):
    if not os.path.exists(args.data_save_path):
        print("data存储路径不存在，已新建")
        os.makedirs(args.data_save_path)

    if not os.path.exists(args.pt_save_path):
        print("pt存储路径不存在，已新建")
        os.makedirs(args.pt_save_path)

    if not os.path.exists(args.fig_save_path):
        print("figure存储路径不存在，已新建")
        os.makedirs(args.fig_save_path)

    if not os.path.exists(args.log_save_path):
        print("log存储路径不存在，已新建")
        os.makedirs(args.log_save_path)


def get_config(dataset='bs', dimension=2):
    print('调用 {}d {} config'.format(dimension, dataset))
    if dimension == 3:
        if dataset == 'gs':
            return gs_args_3d.get_config_gs(dataset)
    elif dimension == 2:
        if dataset == 'bs':
            return bs_args_2d.get_config_bs(dataset)
        elif dataset == 'burgers':
            return burgers_args_2d.get_config_burgers(dataset)
        elif dataset == 'ns':
            return ns_args_2d.get_config_ns(dataset)
        elif dataset == 'jax_ns':
            return ns_jax_args_2d.get_config_ns(dataset)
        elif dataset == 'cf':
            return cf_args_2d.get_config_cf(dataset)


def load_model(path):
    """
        加载模型
    """
    checkpoint = torch.load(path)
    net = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    best_valid_loss = checkpoint['best_valid_loss']

    return net, optimizer, scheduler, best_valid_loss


def random_masking(x, mask_ratio, seed):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [L, D], sequence
    """
    L, D = x.shape  # length, dim
    len_keep = int(L * (Decimal(str(1)) - Decimal(str(mask_ratio))))
    set_random_seed(seed)
    noise = torch.rand(L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=0, descending=True)  # descending
    ids_keep = ids_shuffle[:len_keep]
    ids_keep, _ = torch.sort(ids_keep, dim=0, descending=False)

    return ids_keep


def filter_adj(edge_index, perm):
    row, col = edge_index[0], edge_index[1]
    row, col = perm[row], perm[col]
    mask = (row >= 0) & (col >= 0)
    row_mask, col_mask = row[mask], col[mask]

    edge_index_mask = torch.stack([row_mask, col_mask], dim=0)

    return edge_index_mask


def prepare_graph_data(pos):
    """
    :param x: [n, 2], numpy
    :return edge_index: [2, e], numpy
    """
    tri = Delaunay(pos)
    neighbors_tri = tri.vertex_neighbor_vertices
    edge_index = []
    for i in range(len(neighbors_tri[0]) - 1):
        # edge_index.append([i, i])
        for j in range(neighbors_tri[0][i], neighbors_tri[0][i + 1]):
            edge_index.append([i, neighbors_tri[1][j]])

    edge_index = np.array(edge_index).T
    return edge_index


import matplotlib.pyplot as plt


def gen_multi_mesh(args, edge_index, pos):
    edge_index_list = [edge_index]
    pos_list = [pos]
    ori_pos = pos
    mask_list = []
    for _ in range(args.pool_layer):
        L = torch.zeros((pos.shape[0], 1))
        mask = random_masking(L, mask_ratio=args.mask_ratio, seed=args.seed)
        mask_list.append(mask)
        pos_mask = pos[mask]
        # edge_index_new = knn_graph(pos_mask, self.args.num_knn * torch.randint(1, 4, (1,)), loop=False, flow='target_to_source')
        edge_index_new = torch.tensor(prepare_graph_data(pos_mask), device=edge_index.device)
        edge_index_list.append(edge_index_new)
        pos = pos_mask
        pos_list.append(pos)

    for _ in range(args.pool_layer):
        j = args.pool_layer - _ - 1
        edge_index_new = filter_adj(edge_index_list[j + 1], mask_list[j])
        edge_index_list[j] = torch.unique(torch.cat((edge_index_list[j], edge_index_new), dim=-1), dim=1)

    return edge_index_list[0]


class PeriodicTimeEncoder():
    """Encode linear, periodic time into 2-dimensional positions on a circle.

    This ensures that the time features have a smooth transition at the periodicity
    boundary, e.g. new year's eve for data with yearly periodicity.
    """

    def __init__(self, base, period):
        # super(PeriodicTimeEncoder, self).__init__()
        self.base = base
        self.period = period

    def encode(self, t):
        phase = (t - self.base) * (2 * torch.pi / self.period)
        return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def process_ux(args, seed):
    """
        加载数据
        # input u [T,x,y,z,f_dim] # (3001,(100*100*100),1 or 2 or 3)
        # input pos [x*y*z,2 or 3]
    """
    u_path = '{}/{}d_{}_u_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                                 args.all_time_steps, args.mesh_node, seed)
    # dataloader = MatReader(u_path)
    # u = dataloader.read_field('uv')
    u = np.load(u_path)
    u = torch.from_numpy(u).float()
    """periodic boundary condition"""
    ##########################################################################
    u = fix_bc_in_load(args, u)
    ########################################################################
    x_path = '{}/{}d_{}_x_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                                 args.all_time_steps, args.mesh_node, seed)
    pos = np.load(x_path)
    pos = torch.from_numpy(pos).float()
    # pos = pos.reshape(args.length,args.width,args.height,-1)

    # edge_index = torch.tensor(prepare_graph_data(pos), dtype=torch.long)
    # 为图中的节点添加自环，对于有自环的节点，它会再为该节点添加一个自环
    # edge_index,_ = torch_geometric.utils.add_self_loops(edge_index)
    # 为图中还没有自环的节点添加自环
    # edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index)
    pos = pos.view(-1, pos.shape[-1])

    "space subsample"
    ########################################################################
    # if args.dimension == 2:
    #     u = u.reshape(u.shape[0], args.height, args.width, u.shape[-1])
    #     pos = pos.reshape(args.height, args.width, 2)
    # elif args.dimension == 3:
    #     u = u.reshape(u.shape[0], args.height, args.width, args.length, u.shape[-1])
    #     pos = pos.reshape(args.height, args.width, args.length, 3)
    ########################################################################

    # edge_index = knn_graph(pos, args.num_knn, loop=False) # flow='target_to_source'
    # edge_index = edge_index.index_select(0, torch.LongTensor([1, 0]))
    # edge_index_new = torch.index_select(edge_index, 0, torch.tensor([1, 0], device=edge_index.device))
    if args.dataset_name == 'cf':
        eid_path = os.path.join('{}/{}d_{}_eid_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
                                                                      args.dataset_name, args.all_time_steps,
                                                                      args.mesh_node, seed))
        edge_index = np.load(eid_path)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = torch.index_select(edge_index, 0, torch.tensor([1, 0], device=edge_index.device))
    else:
        edge_index = prepare_graph_data(pos)  # [2, e]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    time_num = u.shape[0]
    node_num = u.shape[1]

    edge_index = gen_multi_mesh(args, edge_index, pos)

    edge_index = torch.unique(edge_index, dim=1)

    """节点类型"""
    if args.dataset_name == 'cf' or args.dataset_name == 'bs':
        type_path = os.path.join('{}/{}d_{}_type_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
                                                                        args.dataset_name, args.all_time_steps,
                                                                        args.mesh_node, seed))
        type = np.load(type_path)
        type = torch.from_numpy(type).float()
    else:
        type = torch.zeros([node_num, 1])
        # if args.pad_len != 0:
        # if args.dimension == 2:
        #     type = type.reshape((args.height + 2 * args.pad_len), (args.width + 2 * args.pad_len), 1)
        #     type[:(args.pad_len+1), :, :] = NodeType.WALL_BOUNDARY
        #     type[-(args.pad_len+1):, :, :] = NodeType.WALL_BOUNDARY
        #     type[:, :(args.pad_len+1), :] = NodeType.WALL_BOUNDARY
        #     type[:, -(args.pad_len+1):, :] = NodeType.WALL_BOUNDARY
        #     type = type.reshape(-1, 1)
        # elif args.dimension == 3:
        #     type = type.reshape((args.height + 2 * args.pad_len), (args.width + 2 * args.pad_len),
        #                         (args.length + 2 * args.pad_len), 1)
        #     type[:(args.pad_len+1), :, :, :] = NodeType.WALL_BOUNDARY
        #     type[-(args.pad_len+1):, :, :, :] = NodeType.WALL_BOUNDARY
        #     type[:, :(args.pad_len+1), :, :] = NodeType.WALL_BOUNDARY
        #     type[:, -(args.pad_len+1):, :, :] = NodeType.WALL_BOUNDARY
        #     type[:, :, :(args.pad_len+1), :] = NodeType.WALL_BOUNDARY
        #     type[:, :, -(args.pad_len+1):, :] = NodeType.WALL_BOUNDARY
        #     type = type.reshape(-1, 1)
        if args.pad_len != 0:
            if args.dimension == 2:
                type = type.reshape((args.height + 2 * args.pad_len), (args.width + 2 * args.pad_len), 1)
                type[:args.pad_len, :, :] = NodeType.PERIODIC_BOUNDARY
                type[-args.pad_len:, :, :] = NodeType.PERIODIC_BOUNDARY
                type[:, :args.pad_len, :] = NodeType.PERIODIC_BOUNDARY
                type[:, -args.pad_len:, :] = NodeType.PERIODIC_BOUNDARY
                type = type.reshape(-1, 1)
            elif args.dimension == 3:
                type = type.reshape((args.height + 2 * args.pad_len), (args.width + 2 * args.pad_len),
                                    (args.length + 2 * args.pad_len), 1)
                type[:args.pad_len, :, :, :] = NodeType.PERIODIC_BOUNDARY
                type[-args.pad_len:, :, :, :] = NodeType.PERIODIC_BOUNDARY
                type[:, :args.pad_len, :, :] = NodeType.PERIODIC_BOUNDARY
                type[:, -args.pad_len:, :, :] = NodeType.PERIODIC_BOUNDARY
                type[:, :, :args.pad_len, :] = NodeType.PERIODIC_BOUNDARY
                type[:, :, -args.pad_len:, :] = NodeType.PERIODIC_BOUNDARY
                type = type.reshape(-1, 1)

    """时间"""
    timestep = torch.ones([node_num, 1])

    """other variables"""
    #############################################
    eq_variables = {}
    variables = torch.Tensor()
    if args.dataset_name == 'gs':
        # Diffusion coefficients
        eq_variables['DA'] = args.DA * torch.ones([node_num, 1])
        eq_variables['DB'] = args.DB * torch.ones([node_num, 1])
        # define birth/death rates
        eq_variables['f'] = args.f * torch.ones([node_num, 1])
        eq_variables['k'] = args.k * torch.ones([node_num, 1])
        eq_variables['dx'] = args.dx * torch.ones([node_num, 1])
        eq_variables['dy'] = args.dy * torch.ones([node_num, 1])
        eq_variables['dz'] = args.dz * torch.ones([node_num, 1])
        eq_variables['dt'] = args.dt * torch.ones([node_num, 1])
        variables = torch.cat((variables, eq_variables["DA"]), dim=-1)
        variables = torch.cat((variables, eq_variables["DB"]), dim=-1)
        variables = torch.cat((variables, eq_variables["f"]), dim=-1)
        variables = torch.cat((variables, eq_variables["k"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dx"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dy"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dz"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dt"]), dim=-1)
    elif args.dataset_name == 'burgers':
        eq_variables['viscocity'] = args.viscocity * torch.ones([node_num, 1])
        eq_variables['dx'] = args.dx * torch.ones([node_num, 1])
        eq_variables['dy'] = args.dy * torch.ones([node_num, 1])
        eq_variables['dt'] = args.dt * torch.ones([node_num, 1])
        variables = torch.cat((variables, eq_variables["viscocity"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dx"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dy"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dt"]), dim=-1)
    elif args.dataset_name == 'ns':
        # force_path = os.path.join('{}/{}d_{}_force_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
        #                                                                   args.dataset_name, args.all_time_steps,
        #                                                                   args.mesh_node, seed))
        # force = torch.from_numpy(np.load(force_path)).float()
        # force = fix_bc_in_load(args, force).view(-1, 1)
        eq_variables['viscocity'] = args.viscocity * torch.ones([node_num, 1])
        eq_variables['dx'] = args.dx * torch.ones([node_num, 1])
        eq_variables['dy'] = args.dy * torch.ones([node_num, 1])
        eq_variables['dt'] = args.dt * torch.ones([node_num, 1])
        # variables = torch.cat((variables, force), dim=-1)
        variables = torch.cat((variables, eq_variables["viscocity"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dx"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dy"]), dim=-1)
        variables = torch.cat((variables, eq_variables["dt"]), dim=-1)
    elif args.dataset_name == 'jax_ns':
        # force_path = os.path.join('{}/{}d_{}_force_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
        #                                                                   args.dataset_name, args.all_time_steps,
        #                                                                   args.mesh_node, seed))
        # force = torch.from_numpy(np.load(force_path)).float()
        # force = fix_bc_in_load(args, force).view(-1, 1)
        eq_variables['viscocity'] = args.viscocity * torch.ones([node_num, 1])
        # eq_variables['dx'] = args.dx * torch.ones([node_num, 1])
        # eq_variables['dy'] = args.dy * torch.ones([node_num, 1])
        # eq_variables['dt'] = args.dt * torch.ones([node_num, 1])
        # variables = torch.cat((variables, force), dim=-1)
        variables = torch.cat((variables, eq_variables["viscocity"]), dim=-1)
        # variables = torch.cat((variables, eq_variables["dx"]), dim=-1)
        # variables = torch.cat((variables, eq_variables["dy"]), dim=-1)
        # variables = torch.cat((variables, eq_variables["dt"]), dim=-1)
    elif args.dataset_name == 'cf':
        eq_variables['dt'] = 1.0 * torch.ones([node_num, 1])
        v_path = os.path.join('{}/{}d_{}_v_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
                                                                  args.dataset_name, args.all_time_steps,
                                                                  args.mesh_node, seed))
        v = torch.from_numpy(np.load(v_path)).float()
        variables = torch.cat((variables, v), dim=-1)
        variables = torch.cat((variables, eq_variables["dt"]), dim=-1)
    elif args.dataset_name == 'bs':
        eq_variables['dt'] = 1.0 * torch.ones([node_num, 1])
        variables = torch.cat((variables, eq_variables["dt"]), dim=-1)

    """扩大第一维度"""
    # repeat(1,N)
    pos = overlay(time_num, pos)
    edge_index = overlay(time_num, edge_index)
    type = overlay(time_num, type)
    variables = overlay(time_num, variables)

    # time is treated as periodic equation variable
    if args.dataset_name == 'bs':
        time_path = '{}/{}d_{}_time_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                                           args.all_time_steps, args.mesh_node, seed)
        all_time = np.load(time_path)
        all_time = torch.from_numpy(all_time).float()
        base = "{}-01-01".format(seed)
        period = 366.0 if (int(seed) % 4 == 0 and int(seed) % 100 != 0) or int(seed) % 400 == 0 else 365.0
        time_encoder = PeriodicTimeEncoder(torch.tensor(np.datetime64(base).astype(float)).float(),
                                           torch.tensor(period).float())
        time = time_encoder.encode(all_time).view(-1, 1, 2).expand(-1, node_num, 2)

        variables = torch.cat((variables, time), dim=-1)  # time is also treated as equation variable
    else:
        time_list = []
        for step in range(time_num):
            timestep = timestep * step / time_num
            time_list.append(timestep)
        time = torch.stack(time_list, dim=0)
        # t = torch.linspace(0, (u.shape[0] - 1) * args.dt, u.shape[0]).reshape(u.shape[0], 1).cuda()

    # variables = torch.cat((variables, time), dim=-1)  # time is also treated as equation variable

    """time truncate"""
    ##########################################################################
    # if args.dataset_name == 'cf':
    #     u = u[10:, ...]  # [301,1000,2]
    #     pos = pos[10:, ...]  # [301,1000,3]
    #     edge_index = edge_index[10:, ...]  # [301,n,2]
    #     type = type[10:, ...]  # [301,1000,1]
    #     time = time[10:, ...]  # [301,1000,1]

    # u = u[:100, ...]  # [301,1000,2]
    # pos = pos[:100, ...]  # [301,1000,3]
    # edge_index = edge_index[:100, ...]  # [301,n,2]
    # type = type[:100, ...]  # [301,1000,1]
    # time = time[:100, ...]  # [301,1000,1]
    ##########################################################################
    """time subsample"""
    ##########################################################################
    # u = u[:1000, ...]  # [301,1000,2]
    # pos = pos[:1000, ...]  # [301,1000,3]
    # edge_index = edge_index[:1000, ...]  # [301,n,2]
    # type = type[:1000, ...]  # [301,1000,1]
    # variables = variables[:1000, ...]  # [301,1000,1]

    u = u[::args.time_step, ...]  # [301,1000,2]
    pos = pos[::args.time_step, ...]  # [301,1000,3]
    edge_index = edge_index[::args.time_step, ...]  # [301,n,2]
    type = type[::args.time_step, ...]  # [301,1000,1]
    variables = variables[::args.time_step, ...]  # [301,1000,1]
    ##########################################################################
    if args.dataset_name == 'cf':
        if args.feature_dim == 2:
            pressure = u[..., args.feature_dim:]
            variables = torch.cat((variables, pressure), dim=-1)  # pressure is also treated as equation variable
            u = u[..., :args.feature_dim]
    elif args.dataset_name == 'bs':
        if args.feature_dim == 2:
            temperature = u[..., args.feature_dim:]
            variables = torch.cat((variables, temperature), dim=-1)  # pressure is also treated as equation variable
            u = u[..., :args.feature_dim]
        elif args.feature_dim == 1:
            velocity = u[..., :-args.feature_dim]
            variables = torch.cat((variables, velocity), dim=-1)  # pressure is also treated as equation variable
            u = u[..., -args.feature_dim:]

    # print(torch.where(torch.isnan(u)))

    return u, u, pos, edge_index, type, variables


def overlay(num, temp):
    temp_list = []
    for _ in range(num):
        temp_list.append(temp)
    temp = torch.stack(temp_list, dim=0)
    return temp


def downsample_ux(args, u, pos, edge_index, type, interval=1):
    u = u.reshape(-1, (args.height + args.pad_len), (args.width + args.pad_len), u.shape[2])
    u = u[:, ::interval, ::interval, :]

    pos = pos.reshape((args.height + args.pad_len), (args.width + args.pad_len), pos.shape[1])
    pos = pos[::interval, ::interval, :]

    type = type.reshape((args.height + args.pad_len), (args.width + args.pad_len), type.shape[1])
    type = type[::interval, ::interval, :]

    adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index).todense()
    adj = adj[::interval, ::interval]

    adj = sp.coo_matrix(adj)
    # adj = torch.sparse_coo_tensor(adj)
    # indices = np.vstack((adj.row, adj.col))
    # index = torch.LongTorch(indices)
    # value = torch.FloatTorch(adj.data)
    # edge_index = torch.sparse_coo_tensor(index, value, adj.shape)

    edge_index, _ = torch_geometric.utils.from_scipy_sparse_matrix(adj)

    return u, pos, edge_index, type


def downsample_adj(edge_index, interval=1):
    adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index).todense()
    adj = adj[::interval, ::interval]
    adj = sp.coo_matrix(adj)
    # adj = torch.sparse_coo_tensor(adj)
    # indices = np.vstack((adj.row, adj.col))
    # index = torch.LongTorch(indices)
    # value = torch.FloatTorch(adj.data)
    # edge_index = torch.sparse_coo_tensor(index, value, adj.shape)
    edge_index, _ = torch_geometric.utils.from_scipy_sparse_matrix(adj)

    return edge_index


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class TimeseriesDataset(Dataset):
    def __init__(self, X, seq_len=1):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, idx):
        return self.X[idx:idx + self.seq_len]


"""
class TimeseriesGraphDataset(Dataset):
    def __init__(self, X, Y, input_len=1, roll_len=1, pos=None, edge_index=None, type=None, timestamp=None, args=None):
        self.X = X
        self.Y = Y
        self.input_len = input_len
        self.roll_len = roll_len
        self.pos = pos
        self.edge_index = edge_index
        self.type = type
        self.timestamp = timestamp
        self.args = args

    def __len__(self):
        return self.X.__len__() - (self.roll_len + self.input_len - 1)

    def __getitem__(self, idx):
        # norm = Normalizer(size=self.args.num_classes, name='output_normalizer')
        # if self.args.predict_type=='increment':
        #     mean =


        return dict(x=self.X[idx:idx + self.input_len],
                    y=self.Y[idx + self.input_len:idx + self.input_len + self.roll_len],
                    pos=self.pos[idx:idx + self.input_len + self.roll_len],
                    edge_index=self.edge_index[idx:idx + self.input_len + self.roll_len],
                    type=self.type[idx:idx + self.input_len + self.roll_len],
                    timestamp=self.timestamp[idx:idx + self.input_len + self.roll_len])


"""


class TimeseriesGraphDataset(torch_geometric.data.Dataset):
    # class TimeseriesGraphDataset(Dataset):
    def __init__(self, X, Y, input_len=1, roll_len=1, pos=None, edge_index=None, type=None, v=None, args=None):
        super(TimeseriesGraphDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.input_len = input_len
        self.roll_len = roll_len
        self.pos = pos
        self.edge_index = edge_index
        self.type = type
        self.v = v
        self.args = args

    # def __len__(self):
    #     return self.X.__len__() - (self.roll_len + self.input_len - 1)
    def len(self):
        return self.X.__len__() - (self.roll_len + self.input_len - 1)

    def generate_torchgeometric_dataset(self):
        """
        Generate dataset that can be used to train with PyG Graph.
        """
        """
        Args:
            data (dict): Data  with keys t, uv,x,y.
        Returns:
            dataset (list): Array of torchgeometric Data objects.
        """

        n_sims = self.len()
        dataset = []

        for idx in range(0, n_sims, self.roll_len):
            x = self.X[idx:idx + self.input_len].permute(1, 0, 2).reshape(-1, self.args.feature_dim * self.input_len)
            y = self.Y[idx + self.input_len:idx + self.input_len + self.roll_len].permute(1, 0, 2).reshape(-1,
                                                                                                           self.args.feature_dim * self.roll_len)
            # pos = self.pos[idx:idx + self.input_len].permute(1, 0, 2).reshape(-1,
            #                                                                   self.args.pos_feature_dim * self.input_len)
            # edge_idx = self.edge_index[idx:idx + self.input_len].permute(1, 0, 2).reshape(2 * self.input_len, -1)
            # type = self.type[idx:idx + self.input_len].permute(1, 0, 2).reshape(-1, 1 * (self.input_len))
            # v = self.v[idx:idx + self.input_len].permute(1, 0, 2).reshape(-1,
            #                                                               self.args.addition_feature_dim * self.input_len)
            pos = self.pos[idx:idx + 1].permute(1, 0, 2).reshape(-1, self.args.pos_feature_dim)
            edge_idx = self.edge_index[idx:idx + 1].permute(1, 0, 2).reshape(2, -1)
            type = self.type[idx:idx + 1].permute(1, 0, 2).reshape(-1, 1)
            v = self.v[idx:idx + 1].permute(1, 0, 2).reshape(-1, self.args.addition_feature_dim)

            graph = Data(x=x, edge_index=edge_idx, y=y, pos=pos, type=type, v=v)

            dataset.append(graph)

        return dataset


# 初始化加载数据过程中修补BC
def fix_bc_in_load(args, u):
    if args.pad_len != 0:
        if args.dimension == 2:
            u = u.reshape(-1, args.height, args.width, u.shape[-1])
            u = torch.cat((u[:, -args.pad_len:, :, :], u, u[:, :args.pad_len, :, :]), dim=1)  # (29,25,2)
            u = torch.cat((u[:, :, -args.pad_len:, :], u, u[:, :, :args.pad_len, :]), dim=2)  # (29,29,2)
            u = u.reshape(u.shape[0], -1, u.shape[-1])
            return u
        elif args.dimension == 3:
            u = u.reshape(-1, args.length, args.height, args.width, u.shape[-1])
            u = torch.cat((u[:, -args.pad_len:, :, :, :], u, u[:, :args.pad_len, :, :, :]), dim=1)  # (14,10,10,2)
            u = torch.cat((u[:, :, -args.pad_len:, :, :], u, u[:, :, :args.pad_len, :, :]), dim=2)  # (14,14,10,2)
            u = torch.cat((u[:, :, :, -args.pad_len:, :], u, u[:, :, :, :args.pad_len, :]), dim=3)  # (14,14,14,2)
            u = u.reshape(u.shape[0], -1, u.shape[-1])
            return u
    return u


# 在训练过程中修补BC
def fix_bc_in_solver(args, u):
    if args.pad_len != 0:
        if args.dimension == 2:
            u = u.reshape(-1, (args.height + 2 * args.pad_len), (args.width + 2 * args.pad_len), u.shape[-1])
            u = u[:, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, :]
            u = torch.cat((u[:, -args.pad_len:, :, :], u, u[:, :args.pad_len, :, :]), dim=1)  # (-1,29,25,2)
            u = torch.cat((u[:, :, -args.pad_len:, :], u, u[:, :, :args.pad_len, :]), dim=2)  # (-1,29,29,2)
            u = u.reshape(-1, u.shape[-1])
            return u
        elif args.dimension == 3:
            u = u.reshape(-1, (args.length + 2 * args.pad_len), (args.height + 2 * args.pad_len),
                          (args.width + 2 * args.pad_len), u.shape[-1])
            u = u[:, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, :]
            u = torch.cat((u[:, -args.pad_len:, :, :, :], u, u[:, :args.pad_len, :, :, :]), dim=1)  # (-1,10,10,2)
            u = torch.cat((u[:, :, -args.pad_len:, :, :], u, u[:, :, :args.pad_len, :, :]), dim=2)  # (-1,14,10,2)
            u = torch.cat((u[:, :, :, -args.pad_len:, :], u, u[:, :, :, :args.pad_len, :]), dim=3)  # (-1,14,14,2)
            u = u.reshape(-1, u.shape[-1])
            return u
    return u


# 在训练过程中修补obstacle
def fix_obst_in_solver(args, u, y, type):
    # y = y[..., :args.num_classes] - x[..., :args.num_classes]
    node_type = type[:, 0]  # [40000,1]
    mask = (node_type == NodeType.NORMAL)
    # mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
    boundary_mask = torch.logical_not(mask)
    u[boundary_mask] = y[boundary_mask]
    # batch_input[boundary_mask] = y[:, :, :, 0:2][boundary_mask].cuda()
    return u


# 在计算loss过程中删除BC
def minus_bc_in_loss(args, u):
    # [b,t,n,f] [32,10,841,2]
    if args.pad_len != 0:
        if args.dimension == 2:
            u = u.reshape(u.shape[0], u.shape[1], (args.height + 2 * args.pad_len), (args.width + 2 * args.pad_len),
                          args.feature_dim)
            u = u[:, :, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, :]
            u = u.reshape(u.shape[0], u.shape[1], -1, args.feature_dim)
            return u
        elif args.dimension == 3:
            u = u.reshape(u.shape[0], u.shape[1], (args.length + 2 * args.pad_len), (args.height + 2 * args.pad_len),
                          (args.width + 2 * args.pad_len), args.feature_dim)
            u = u[:, :, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, :]
            u = u.reshape(u.shape[0], u.shape[1], -1, args.feature_dim)
            return u
    return u


# 在计算loss过程中删除BC
def minus_bc_in_pos(args, u):
    # [b,t,n,f] [32,10,841,2]
    if args.pad_len != 0:
        if args.dimension == 2:
            u = u.reshape(u.shape[0], u.shape[1], (args.height + 2 * args.pad_len), (args.width + 2 * args.pad_len),
                          args.dimension)
            u = u[:, :, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, :]
            u = u.reshape(u.shape[0], u.shape[1], -1, args.dimension)
            return u
        elif args.dimension == 3:
            u = u.reshape(u.shape[0], u.shape[1], (args.length + 2 * args.pad_len), (args.height + 2 * args.pad_len),
                          (args.width + 2 * args.pad_len), args.dimension)
            u = u[:, :, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, args.pad_len:-args.pad_len, :]
            u = u.reshape(u.shape[0], u.shape[1], -1, args.dimension)
            return u
    return u


def cal_physics_loss(args, pred, label, loss_func, criterion):
    """ calculate the physics loss """
    assert pred.shape == label.shape

    pred = pred.permute(0, 2, 1).reshape(-1, args.feature_dim, args.height + args.pad_len, args.width + args.pad_len)
    label = label.permute(0, 2, 1).reshape(-1, args.feature_dim, args.height + args.pad_len, args.width + args.pad_len)

    return loss_func(pred, label, criterion)


dy_2d_op = [
    [[[0, 0, 1 / 12, 0, 0], [0, 0, -8 / 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 8 / 12, 0, 0], [0, 0, -1 / 12, 0, 0]]]]

dx_2d_op = [
    [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]]

lap_2d_op = [[[[0, 0, -1 / 12, 0, 0], [0, 0, 4 / 3, 0, 0], [-1 / 12, 4 / 3, - 5, 4 / 3, -1 / 12], [0, 0, 4 / 3, 0, 0],
               [0, 0, -1 / 12, 0, 0]]]]


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.deno = deno
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=self.padding, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, feature):
        derivative = self.filter(feature)
        return derivative / self.deno


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=5, name=''):
        super(Conv2dDerivative, self).__init__()
        self.deno = deno
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        # self.padding = 0
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=self.padding, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, feature):
        derivative = self.filter(feature)
        return derivative / self.deno


class physics_loss_generator(nn.Module):
    """ Loss generator for physics loss"""

    def __init__(self, args):
        """ Construct the derivatives """
        super(physics_loss_generator, self).__init__()
        self.args = args

        """ spatial derivative operator """
        self.laplace = Conv2dDerivative(DerFilter=lap_2d_op, deno=self.args.dx ** 2, kernel_size=5,
                                        name='laplace_operator')

        """ Spatial derivative operator """
        self.partial_x = Conv2dDerivative(DerFilter=dx_2d_op, deno=self.args.dx, kernel_size=5,
                                          name='dx_operator')

        """ Spatial derivative operator """
        self.partial_y = Conv2dDerivative(DerFilter=dy_2d_op, deno=self.args.dy, kernel_size=5,
                                          name='dy_operator')

        """ temporal derivative operator """
        # self.partial_t = Conv1dDerivative(DerFilter=[[[-1, 1, 0]]], deno=(self.args.delta_t * 1), kernel_size=3,
        #                            name='partial_t')
        self.partial_t = Conv1dDerivative(DerFilter=[[[-1, 1]]], deno=(self.args.dt * 1), kernel_size=2,
                                          name='partial_t')

    def get_grid_physics_operator_dim1(self, u):
        """ spatial derivatives """
        laplace_u = self.laplace(u)
        u_x = self.partial_x(u)
        u_y = self.partial_y(u)

        return {"laplace_u": laplace_u, "u_x": u_x, "u_y": u_y}

    def get_grid_physics_operator_dim2(self, u, v):
        """ make sure the dimensions consistent """
        assert u.shape == v.shape

        """ spatial derivatives """
        laplace_u = self.laplace(u)
        laplace_v = self.laplace(v)

        u_x = self.partial_x(u)
        v_x = self.partial_x(v)

        u_y = self.partial_y(u)
        v_y = self.partial_y(v)

        return {"laplace_u": laplace_u, "laplace_v": laplace_v, "u_x": u_x, "v_x": v_x, "u_y": u_y, "v_y": v_y}

    def get_physics_loss_burgers(self, op_dict, u, v, nu=0.01):
        """ Burgers eqn """
        u_t = nu * op_dict["laplace_u"] - u * op_dict["u_x"] - v * op_dict["u_y"]
        v_t = nu * op_dict["laplace_v"] - u * op_dict["v_x"] - v * op_dict["v_y"]

        return u_t, v_t

    def get_physics_loss_gs(self, op_dict, u, v, DA=0.01, DB=0.08, F=0.06, K=0.062):
        """ GS eqn """
        u_t = DA * op_dict["laplace_u"] - u * v ** 2 + F * (1 - u)
        v_t = DB * op_dict["laplace_v"] + u * v ** 2 - v * (K + F)

        return u_t, v_t

    def get_physics_loss_fn(self, op_dict, u, v, DA=1, DB=100, alpha=0.01, beta=0.25):
        """ FN eqn """
        u_t = DA * op_dict["laplace_u"] + u - u ** 3 - v + alpha
        v_t = DB * op_dict["laplace_v"] + (u - v) * beta

        return u_t, v_t

    def get_physics_loss_random(self, op_dict):
        """ Random eqn """
        u_t = op_dict["u_x"] - op_dict["u_y"]
        v_t = op_dict["v_x"] - op_dict["v_y"]

        return u_t, v_t

    def get_physics_loss_heat(self, op_dict, u, D=1):
        """ Heat eqn """
        u_t = D * op_dict["laplace_u"]
        return u_t

    def get_physics_loss_dim1(self, pred, label, criterion, equation):
        pred_u = pred[:, 0:1, :, :]
        label_u = label[:, 0:1, :, :]

        """ spatial derivatives """
        op_dict = self.get_grid_physics_operator_dim1(label_u)
        pred_op_dict = self.get_grid_physics_operator_dim1(pred_u)

        """ physics loss item """
        if equation == 'heat':
            u_t = self.get_physics_loss_heat(op_dict, label_u, D=1)
            p_u_t = self.get_physics_loss_heat(pred_op_dict, pred_u, D=1)

        f_u = (p_u_t - u_t) / 1

        return criterion(f_u, torch.zeros_like(f_u))

    def get_physics_loss_dim2(self, pred, label, criterion, equation):
        pred_u = pred[:, 0:1, :, :]
        pred_v = pred[:, 1:2, :, :]
        label_u = label[:, 0:1, :, :]
        label_v = label[:, 1:2, :, :]

        """ spatial derivatives """
        op_dict = self.get_grid_physics_operator_dim2(label_u, label_v)
        pred_op_dict = self.get_grid_physics_operator_dim2(pred_u, pred_v)

        """ physics loss item """
        if equation == 'burgers':
            u_t, v_t = self.get_physics_loss_burgers(op_dict, label_u, label_v, nu=0.01)
            p_u_t, p_v_t = self.get_physics_loss_burgers(pred_op_dict, pred_u, pred_v, nu=0.01)
        elif equation == 'gs':
            u_t, v_t = self.get_physics_loss_gs(op_dict, label_u, label_v, DA=0.01, DB=0.08, F=0.06, K=0.062)
            p_u_t, p_v_t = self.get_physics_loss_gs(pred_op_dict, pred_u, pred_v, DA=0.01, DB=0.08, F=0.06, K=0.062)
        elif equation == 'fn':
            u_t, v_t = self.get_physics_loss_fn(op_dict, label_u, label_v, DA=1, DB=100, alpha=0.01, beta=0.25)
            p_u_t, p_v_t = self.get_physics_loss_fn(pred_op_dict, pred_u, pred_v, DA=1, DB=100, alpha=0.01, beta=0.25)

        f_u = (p_u_t - u_t) / 1
        f_v = (p_v_t - v_t) / 1

        return criterion(f_u, torch.zeros_like(f_u)) + criterion(f_v, torch.zeros_like(f_v))

    def get_random_physics_loss_dim2(self, pred, label, criterion, equation):
        pred_u = pred[:, 0:1, :, :]
        pred_v = pred[:, 1:2, :, :]
        label_u = label[:, 0:1, :, :]
        label_v = label[:, 1:2, :, :]

        """ spatial derivatives """
        op_dict = self.get_grid_physics_operator_dim2(label_u, label_v)
        pred_op_dict = self.get_grid_physics_operator_dim2(pred_u, pred_v)

        """ physics loss item """
        if equation == 'burgers':
            u_t, v_t = self.get_physics_loss_burgers(op_dict, label_u, label_v, nu=0.01)
            p_u_t, p_v_t = self.get_physics_loss_burgers(pred_op_dict, pred_u, pred_v, nu=0.01)
        elif equation == 'gs':
            u_t, v_t = self.get_physics_loss_gs(op_dict, label_u, label_v, DA=0.01, DB=0.08, F=0.06, K=0.062)
            p_u_t, p_v_t = self.get_physics_loss_gs(pred_op_dict, pred_u, pred_v, DA=0.01, DB=0.08, F=0.06, K=0.062)
        elif equation == 'fn':
            u_t, v_t = self.get_physics_loss_fn(op_dict, label_u, label_v, DA=1, DB=100, alpha=0.01, beta=0.25)
            p_u_t, p_v_t = self.get_physics_loss_fn(pred_op_dict, pred_u, pred_v, DA=1, DB=100, alpha=0.01, beta=0.25)
        elif equation == 'random':
            u_t, v_t = self.get_physics_loss_random(op_dict)
            p_u_t, p_v_t = self.get_physics_loss_random(pred_op_dict)

        # f_u = (p_u_t - u_t) / 1
        # f_v = (p_v_t - v_t) / 1
        #
        # return criterion(f_u, torch.zeros_like(f_u)) + criterion(f_v, torch.zeros_like(f_v))

        return p_u_t, p_v_t, pred_op_dict

    def forward(self, pred, label, criterion):
        if self.args.dataset_name in ['heat']:
            return self.get_physics_loss_dim1(pred, label, criterion, self.args.dataset_name)
        elif self.args.dataset_name in ['burgers', 'gs', 'fn']:
            return self.get_physics_loss_dim2(pred, label, criterion, self.args.dataset_name)
        elif self.args.dataset_name in ['random']:
            return self.get_random_physics_loss_dim2(pred, label, criterion, self.args.dataset_name)


#####################################################
# 分割线 基础函数
#####################################################


# 读取mat文件
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


#
# # 处理mat文件，生成所需要的格式数据
# def process_mat(UV_matFile, matFile):
#     UV = UV_matFile.data['uv']  # shape(2001,2,100,100)
#     UV = torch.from_numpy(UV).float()  # shape(2001,2,100,100)
#     UV = torch.cat((UV, UV[:, :, :, 0:1]), dim=3)  # (2001,2,100,101)
#     UV = torch.cat((UV, UV[:, :, 0:1, :]), dim=2)  # (2001,2,101,101)
#     UV = UV.permute(0, 2, 3, 1)  # (2001,101,101,2)
#
#     FineGrid = matFile.data['FineGrid']  # shape(3,10201)
#     EncoderCluster = matFile.data['EncoderCluster']  # shape(4,10201)
#     GraphGrid = matFile.data['GraphGrid']  # shape(3,261)
#     GraphConnectivity = matFile.data['GraphConnectivity'].astype(np.float32)  # shape(6,116)
#     DecoderCluster = matFile.data['DecoderCluster']  # shape(5,10201)
#
#     GraphGrid = torch.from_numpy(GraphGrid).float()  # size:(3,261)
#     FineGrid = torch.from_numpy(FineGrid).float()  # size:(3,10201)
#     EncoderCluster = torch.from_numpy(EncoderCluster[:2, :])  # shape(2,10201)
#     GraphConnectivity = torch.from_numpy(GraphConnectivity)  # shape(6,116)
#     labels = list(DecoderCluster[1:, :])
#     DecoderCluster = torch.LongTensor(labels)
#     # DecoderCluster = torch.from_numpy(DecoderCluster[1:, :]).float()  # shape(4,10201)
#     DecoderCluster = DecoderCluster.permute(1, 0)  # shape(10201,4)
#
#     """
#     print("开始添加位置坐标及FineGridNode编号")
#     # 特征增加位置坐标信息及FineGridNode编号
#     tempFineGrid = FineGrid
#     tempFineGrid = tempFineGrid.permute(1, 0)  # size:(10201,3)
#     tempFineGrid = tempFineGrid.view(101, 101, 3)  # shape:(101,101,3)
#     tempFineGrid = torch.unsqueeze(tempFineGrid, dim=0)  # shape:(1,101,101,3)
#     tempFineGrid = tempFineGrid.expand(2001, 101, 101, 3)  # shape:(2001,101,101,3)
#     UV = torch.cat((UV, tempFineGrid[:, :, :, 1:2]), dim=3)  # (2001,101,101,3)
#     UV = torch.cat((UV, tempFineGrid[:, :, :, 2:3]), dim=3)  # (2001,101,101,4)
#     print("已成功添加位置坐标及FineGridNode编号")
#     """
#     return UV, FineGrid, EncoderCluster, GraphGrid, GraphConnectivity, DecoderCluster


# 处理mat文件，生成所需要的格式数据
def process_mat(UV_matFile, matFile):
    UV = UV_matFile.data['uv']  # shape(2001,2,100,100)
    UV = torch.from_numpy(UV).float()  # shape(2001,2,100,100)
    # UV = UV[:, :, ::2, ::2]
    UV = torch.cat((UV, UV[:, :, :, 0:1]), dim=3)  # (2001,2,100,101)
    UV = torch.cat((UV, UV[:, :, 0:1, :]), dim=2)  # (2001,2,101,101)
    UV = UV.permute(0, 2, 3, 1)  # (2001,101,101,2)
    time = UV.shape[0]
    width = UV.shape[1]
    height = UV.shape[2]
    f_dim = UV.shape[3]

    FineGrid = matFile.data['FineGrid']  # shape(3,10201)
    EncoderCluster = matFile.data['EncoderCluster']  # shape(4,10201)
    GraphGrid = matFile.data['GraphGrid']  # shape(3,261)
    GraphConnectivity = matFile.data['GraphConnectivity'].astype(np.float32)  # shape(6,116)
    DecoderCluster = matFile.data['DecoderCluster']  # shape(5,10201)

    GraphGrid = torch.from_numpy(GraphGrid).float()  # size:(3,261)
    FineGrid = torch.from_numpy(FineGrid).float()  # size:(3,10201)
    FineGrid = FineGrid.reshape(-1, width, height)[:, :-1, :-1]  # size:(3,10000)
    # FineGrid = FineGrid[:, ::2, ::2]
    FineGrid = torch.cat((FineGrid, FineGrid[:, :, 0:1]), dim=2)
    FineGrid = torch.cat((FineGrid, FineGrid[:, 0:1, :]), dim=1)
    FineGrid = FineGrid.reshape(-1, width * height)

    EncoderCluster = torch.from_numpy(EncoderCluster[:2, :])  # shape(2,10201)
    GraphConnectivity = torch.from_numpy(GraphConnectivity)  # shape(6,116)
    labels = list(DecoderCluster[1:, :])
    DecoderCluster = torch.LongTensor(labels)
    # DecoderCluster = torch.from_numpy(DecoderCluster[1:, :]).float()  # shape(4,10201)
    DecoderCluster = DecoderCluster.permute(1, 0)  # shape(10201,4)

    # 特征增加位置坐标信息及时间戳
    tempFineGrid = FineGrid[1:, :]
    tempFineGrid = tempFineGrid.permute(1, 0)  # size:(10201,2)
    tempFineGrid = tempFineGrid.view(width, height, f_dim)  # shape:(101,101,2)
    tempFineGrid = torch.unsqueeze(tempFineGrid, dim=0)  # shape:(1,101,101,2)
    tempFineGrid = tempFineGrid.expand(time, width, height, f_dim)  # shape:(2000,101,101,2)
    UV = torch.cat((UV, tempFineGrid[:, :, :, 0:1]), dim=3)  # (2000,101,101,3)
    UV = torch.cat((UV, tempFineGrid[:, :, :, 1:2]), dim=3)  # (2000,101,101,4)
    # print("已生成位置坐标")

    # time_list = []
    # timestep = torch.ones([101, 101, 1])
    # for step in range(time):
    #     timestep = timestep * step
    #     time_list.append(timestep)
    # time_UV = torch.stack(time_list, dim=0)
    # UV = torch.cat((UV, time_UV[:, :, :, 0:1]), dim=3)  # (2000,101,101,5)
    # print("已添加时间戳")

    uv_feature = UV[:500, :, :, 0:2]
    # uv_feature = UV[:500, :, :, :]
    pos_feature = UV[0:1, :, :, 2:4]
    # time_feature = UV[:, :, :, 4:5]
    return uv_feature, FineGrid, EncoderCluster, GraphGrid, GraphConnectivity, DecoderCluster, pos_feature


def process_connectivity(matFile):
    # quadGrid = matFile.data['e2vcg'].astype(np.float32)  # shape(3061,4)
    quadGrid = matFile.data['e2vcg'].astype(np.int64)
    # quadGrid = torch.from_numpy(quadGrid).float()
    edge_index = []
    for i in range(quadGrid.shape[0]):
        edge_index.append([quadGrid[i, 0], quadGrid[i, 1]])
        edge_index.append([quadGrid[i, 0], quadGrid[i, 3]])
        edge_index.append([quadGrid[i, 1], quadGrid[i, 0]])
        edge_index.append([quadGrid[i, 1], quadGrid[i, 2]])
        edge_index.append([quadGrid[i, 2], quadGrid[i, 1]])
        edge_index.append([quadGrid[i, 2], quadGrid[i, 3]])
        edge_index.append([quadGrid[i, 3], quadGrid[i, 0]])
        edge_index.append([quadGrid[i, 3], quadGrid[i, 2]])
    edge_index = np.unique(edge_index, axis=0)
    edge_index = np.array(edge_index).T
    return edge_index


def process_heat_mat(UV_matFile, matFile):
    UV = UV_matFile.data['uv']  # shape(2000,1,100,100)
    UV = torch.from_numpy(UV).float()  # shape(2000,1,100,100)
    # 减少node节点数
    # UV = UV[:, :, ::4, ::4]

    UV = torch.cat((UV, UV[:, :, :, 0:1]), dim=3)  # (2000,1,100,101)
    UV = torch.cat((UV, UV[:, :, 0:1, :]), dim=2)  # (2000,1,101,101)
    UV = UV.permute(0, 2, 3, 1)  # (2000,101,101,1)
    time = UV.shape[0]  # 2000
    width = UV.shape[1]  # 101
    height = UV.shape[2]  # 101
    f_dim = UV.shape[3]  # 1

    FineGrid = matFile.data['FineGrid']  # shape(3,10201)

    FineGrid = torch.from_numpy(FineGrid).float()  # size:(3,10201)
    # FineGrid = FineGrid.reshape(-1, width, height)[:, :-1, :-1]  # size:(3,10000)
    FineGrid = FineGrid.reshape(-1, 101, 101)[:, :-1, :-1]  # size:(3,10000)
    # 减少node节点数
    # FineGrid = FineGrid[:, ::4, ::4]
    #
    FineGrid = torch.cat((FineGrid, FineGrid[:, :, 0:1]), dim=2)
    FineGrid = torch.cat((FineGrid, FineGrid[:, 0:1, :]), dim=1)
    FineGrid = FineGrid.reshape(-1, width * height)

    # 特征增加位置坐标信息及时间戳
    tempFineGrid = FineGrid[1:, :]
    tempFineGrid = tempFineGrid.permute(1, 0)  # size:(10201,2)
    tempFineGrid = tempFineGrid.view(width, height, 2)  # shape:(101,101,2)
    tempFineGrid = torch.unsqueeze(tempFineGrid, dim=0)  # shape:(1,101,101,2)
    tempFineGrid = tempFineGrid.expand(time, width, height, 2)  # shape:(2000,101,101,2)
    # UV = torch.cat((UV, tempFineGrid[:, :, :, 0:1]), dim=3)  # (2000,101,101,3)
    # UV = torch.cat((UV, tempFineGrid[:, :, :, 1:2]), dim=3)  # (2000,101,101,4)
    # print("已生成位置坐标")

    # time_list = []
    # timestep = torch.ones([101, 101, 1])
    # for step in range(time):
    #     timestep = timestep * step
    #     time_list.append(timestep)
    # time_UV = torch.stack(time_list, dim=0)
    # UV = torch.cat((UV, time_UV[:, :, :, 0:1]), dim=3)  # (2000,101,101,5)
    # print("已添加时间戳")

    uv_feature = UV[:, :-1, :-1, 0:1]
    # uv_feature = UV[:500, :, :, :]
    pos_feature = tempFineGrid[0:1, :-1, :-1, :]
    # pos_feature = tempFineGrid[0:1, :, :, :]
    # time_feature = UV[:, :, :, 3:4]
    return uv_feature, FineGrid, pos_feature


def generate_pos(UV, FineGrid):
    time = UV.shape[0]
    width = UV.shape[1]
    height = UV.shape[2]
    f_dim = UV.shape[3]
    tempFineGrid = FineGrid[1:, :]
    tempFineGrid = tempFineGrid.permute(1, 0)  # size:(10201,2)
    tempFineGrid = tempFineGrid.view(width, height, f_dim)  # shape:(101,101,2)
    tempFineGrid = torch.unsqueeze(tempFineGrid, dim=0)  # shape:(1,101,101,2)
    tempFineGrid = tempFineGrid.expand(time, width, height, f_dim)  # shape:(2000,101,101,2)
    UV = torch.cat((UV, tempFineGrid[:, :, :, 0:1]), dim=3)  # (2000,101,101,3)
    UV = torch.cat((UV, tempFineGrid[:, :, :, 1:2]), dim=3)  # (2000,101,101,4)
    return UV[:, :, :, 2:4]


# 处理mat文件，生成所需要的格式数据
def process_grayscott_mat(UV_matFile, matFile):
    UV = UV_matFile.data['uv']  # shape(2,5001,256,256)
    UV = torch.from_numpy(UV).float()  # shape(2,5001,256,256)
    UV = UV.permute(1, 0, 2, 3)  # shape(5001,2,256,256)
    UV = torch.cat((UV, UV[:, :, :, 0:1]), dim=3)  # shape(5001,2,256,257)
    UV = torch.cat((UV, UV[:, :, 0:1, :]), dim=2)  # shape(5001,2,257,257)
    UV = UV.permute(0, 2, 3, 1)  # shape(5001,257,257,2)

    FineGrid = matFile.data['FineGrid']  # shape(3,66049)
    EncoderCluster = matFile.data['EncoderCluster']  # shape(4,66049)
    GraphGrid = matFile.data['GraphGrid']  # shape(3,261)
    GraphConnectivity = matFile.data['GraphConnectivity'].astype(np.float32)  # shape(6,116)
    DecoderCluster = matFile.data['DecoderCluster']  # shape(5,66049)

    GraphGrid = torch.from_numpy(GraphGrid).float()  # size:(3,261)
    FineGrid = torch.from_numpy(FineGrid).float()  # size:(3,66049)
    EncoderCluster = torch.from_numpy(EncoderCluster[:2, :])  # shape(2,66049)
    GraphConnectivity = torch.from_numpy(GraphConnectivity)  # shape(6,116)
    labels = list(DecoderCluster[1:, :])
    DecoderCluster = torch.LongTensor(labels)
    # DecoderCluster = torch.from_numpy(DecoderCluster[1:, :]).float()  # shape(4,66049)
    DecoderCluster = DecoderCluster.permute(1, 0)  # shape(66049,4)

    return UV, FineGrid, EncoderCluster, GraphGrid, GraphConnectivity, DecoderCluster


# 处理mat文件，生成所需要的格式数据
def process_FN_mat(UV_matFile, matFile):
    UV = UV_matFile.data['uv']  # shape(1001,2,128,128)
    UV = torch.from_numpy(UV).float()  # shape(1001,2,128,128)
    UV = torch.cat((UV, UV[:, :, :, 0:1]), dim=3)  # shape(1001,2,128,129)
    UV = torch.cat((UV, UV[:, :, 0:1, :]), dim=2)  # shape(1001,2,129,129)
    UV = UV.permute(0, 2, 3, 1)  # shape(1001,129,129,2)

    FineGrid = matFile.data['FineGrid'].astype(np.float32)  # shape(3,16641)
    EncoderCluster = matFile.data['EncoderCluster'].astype(np.float32)  # shape(4,16641)
    GraphGrid = matFile.data['GraphGrid'].astype(np.float32)  # shape(3,261)
    GraphConnectivity = matFile.data['GraphConnectivity'].astype(np.float32)  # shape(6,116)
    DecoderCluster = matFile.data['DecoderCluster'].astype(np.float32)  # shape(5,16641)

    GraphGrid = torch.from_numpy(GraphGrid).float()  # size:(3,261)
    FineGrid = torch.from_numpy(FineGrid).float()  # size:(3,16641)
    EncoderCluster = torch.from_numpy(EncoderCluster[:2, :])  # shape(2,16641)
    GraphConnectivity = torch.from_numpy(GraphConnectivity)  # shape(6,116)
    labels = list(DecoderCluster[1:, :])
    DecoderCluster = torch.LongTensor(labels)
    # DecoderCluster = torch.from_numpy(DecoderCluster[1:, :]).float()  # shape(4,16641)
    DecoderCluster = DecoderCluster.permute(1, 0)  # shape(16641,4)

    return UV, FineGrid, EncoderCluster, GraphGrid, GraphConnectivity, DecoderCluster


# 处理mat文件，生成所需要的格式数据
def process_image_mat(data_matFile, conn_matFile):
    IMAGE = data_matFile.data['some_array']  # shape(9,888,1230,1)
    IMAGE = torch.from_numpy(IMAGE[:, ::2, ::2, :]).float()  # shape(9,888,1230,1)

    FineGrid = conn_matFile.data['FineGrid']  # shape(3,273060)
    EncoderCluster = conn_matFile.data['EncoderCluster']  # shape(4,273060)
    GraphGrid = conn_matFile.data['GraphGrid']  # shape(3,214)
    GraphConnectivity = conn_matFile.data['GraphConnectivity']  # shape(3,374)
    DecoderCluster = conn_matFile.data['DecoderCluster']  # shape(5,273060)

    GraphGrid = torch.from_numpy(GraphGrid).float()  # shape(3,214)
    FineGrid = torch.from_numpy(FineGrid).float()  # # shape(3,273060)
    EncoderCluster = torch.from_numpy(EncoderCluster[:2, :])  # shape(2,273060)
    GraphConnectivity = torch.from_numpy(GraphConnectivity)  # shape(3,374)
    labels = list(DecoderCluster[1:, :])
    DecoderCluster = torch.LongTensor(labels)
    # DecoderCluster = torch.from_numpy(DecoderCluster[1:, :]).float()  # shape(4,273060)
    DecoderCluster = DecoderCluster.permute(1, 0)  # shape(273060,4)

    return IMAGE, FineGrid, EncoderCluster, GraphGrid, GraphConnectivity, DecoderCluster


#####################################################
# 分割线 graph
#####################################################


# 归一化
def feature_normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normalize(data):
    """Standardize a tensor by dim=0.
    Args: X: A `n` or `n x d`-dim tensor
    Returns:The standardized `X`.
    Example:
        >>> X = torch.rand(4, 3)
        >>> X_standardized = standardize(X)
    """
    data_std = data.std(dim=0)
    data_mean = data.mean(dim=0)
    # data_std = data_std.where(data_std >= 1e-9, torch.full_like(data_std, 1.0))
    data = (data - data_mean) / data_std
    return data, data_mean, data_std


def reverse_normalize(data, data_mean, data_std):
    return data * data_std + data_mean


# 标准化
def feature_standardize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# 如果归一化后的范围是[-1, 1]的话，可以将normalization()函数改为：
def normalization(data):
    _range = np.max(abs(data))
    return data / _range


# 归一化邻接矩阵
def adj_normalize(adj):
    # row_sum = np.array(mx.sum(1))
    # r_inv = np.power(row_sum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = np.diag(r_inv)
    # mx = r_mat_inv.dot(mx)

    row_sum = np.array(adj.sum(1), dtype=np.float32)
    d_inv_sqrt = np.power(row_sum, -1).flatten()  # 输出rowsum ** -1
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 溢出部分赋值为0
    for i in range(adj.shape[0]):
        adj[i] = adj[i] * d_inv_sqrt[i]
    # torch.mul()

    return adj


# 计算生成邻接矩阵
def cal_adj(GraphGrid, EncoderCluster, GraphConnectivity, DecoderCluster):
    encoder_adj = torch.ones([EncoderCluster.shape[1], EncoderCluster.shape[1]])
    # encoder_adj = torch.ones([GraphGrid.shape[1], EncoderCluster.shape[1]])
    # encoder_adj = torch.zeros([GraphGrid.shape[1], EncoderCluster.shape[1]])
    # for j in range(encoder_adj.shape[1]):
    #     start_node = EncoderCluster[0, j].type(torch.int64)
    #     end_node = EncoderCluster[1, j].type(torch.int64)
    #     encoder_adj[start_node - 1, end_node - 1] = 1.0
    # encoder_adj = adj_normalize(encoder_adj)

    processor_adj = torch.ones([GraphGrid.shape[1], GraphGrid.shape[1]])
    # processor_adj = torch.zeros([GraphGrid.shape[1], GraphGrid.shape[1]])
    # for i in range(GraphConnectivity.shape[1]):
    #     start_node1 = GraphConnectivity[0, i].type(torch.int64) - 1
    #     end_node1 = GraphConnectivity[1, i].type(torch.int64) - 1
    #     start_node2 = GraphConnectivity[1, i].type(torch.int64) - 1
    #     end_node2 = GraphConnectivity[0, i].type(torch.int64) - 1
    #     start_node3 = GraphConnectivity[1, i].type(torch.int64) - 1
    #     end_node3 = GraphConnectivity[2, i].type(torch.int64) - 1
    #     start_node4 = GraphConnectivity[2, i].type(torch.int64) - 1
    #     end_node4 = GraphConnectivity[1, i].type(torch.int64) - 1
    #     start_node5 = GraphConnectivity[0, i].type(torch.int64) - 1
    #     end_node5 = GraphConnectivity[2, i].type(torch.int64) - 1
    #     start_node6 = GraphConnectivity[2, i].type(torch.int64) - 1
    #     end_node6 = GraphConnectivity[0, i].type(torch.int64) - 1
    #
    #     processor_adj[start_node1, end_node1] = 1.0
    #     processor_adj[start_node2, end_node2] = 1.0
    #     processor_adj[start_node3, end_node3] = 1.0
    #     processor_adj[start_node4, end_node4] = 1.0
    #     processor_adj[start_node5, end_node5] = 1.0
    #     processor_adj[start_node6, end_node6] = 1.0
    # for i in range(GraphGrid.shape[1]):
    #     processor_adj[i, i] = 1.0
    # processor_adj = adj_normalize(processor_adj)

    decoder_adj = torch.ones([DecoderCluster.shape[0], GraphGrid.shape[1]])
    # decoder_adj = torch.zeros([DecoderCluster.shape[0], GraphGrid.shape[1]])
    # for i in range(decoder_adj.shape[0]):
    #     for k in range(DecoderCluster.shape[1]):
    #         end_node = DecoderCluster[i, k].type(torch.int64)
    #         decoder_adj[i, end_node - 1] = 1.0
    # decoder_adj = adj_normalize(decoder_adj)
    # decoder_adj = encoder_adj.t()

    return encoder_adj, processor_adj, decoder_adj


# 通过filter计算生成导数
def cal_derivative_by_filter(filter, UV):
    UV_xy = torch.zeros([UV.shape[0], UV.shape[0], 2]).detach()
    for i in range(UV.shape[0]):
        for j in range(UV.shape[1]):
            if j == 0:
                # [UV[i, j], UV[i, j + 1], UV[i, j + 2], UV[i, j + 3], UV[i, j + 4]]
                u_x = np.dot(filter[0], UV[i, j:j + 5].detach().numpy())
            elif j == 1:
                # [UV[i, j - 1], UV[i, j], UV[i, j + 1], UV[i, j + 2], UV[i, j + 3]]
                u_x = np.dot(filter[1], UV[i, j - 1:j + 4].detach().numpy())
            elif j == UV.shape[1] - 2:
                # [UV[i, j - 3], UV[i, j - 2], UV[i, j - 1], UV[i, j], UV[i, j + 1]]
                u_x = np.dot(filter[3], UV[i, j - 3:j + 2].detach().numpy())
            elif j == UV.shape[1] - 1:
                # [UV[i, j - 4], UV[i, j - 3], UV[i, j - 2], UV[i, j - 1], UV[i, j]]
                u_x = np.dot(filter[4], UV[i, j - 4:j + 1].detach().numpy())
            else:
                # [UV[i, j - 2], UV[i, j - 1], UV[i, j], UV[i, j + 1], UV[i, j + 2]]
                u_x = np.dot(filter[2], UV[i, j - 2:j + 3].detach().numpy())

            if i == 0:
                # [UV[i, j], UV[i + 1, j], UV[i + 2, j], UV[i + 3, j], UV[i + 4, j]]
                u_y = np.dot(filter[0], UV[i:i + 5, j].detach().numpy())
            elif i == 1:
                # [UV[i - 1, j], UV[i, j], UV[i + 1, j], UV[i + 2, j], UV[i + 3, j]]
                u_y = np.dot(filter[1], UV[i - 1:i + 4, j].detach().numpy())
            elif i == UV.shape[0] - 2:
                # [UV[i - 3, j], UV[i - 2, j], UV[i - 1, j], UV[i, j], UV[i + 1, j]]
                u_y = np.dot(filter[3], UV[i - 3:i + 2, j].detach().numpy())
            elif i == UV.shape[0] - 1:
                # [UV[i - 4, j], UV[i - 3, j], UV[i - 2, j], UV[i - 1, j], UV[i, j]]
                u_y = np.dot(filter[4], UV[i - 4:i + 1, j].detach().numpy())
            else:
                # [UV[i - 2, j], UV[i - 1, j], UV[i, j], UV[i + 1, j], UV[i + 2, j]]
                u_y = np.dot(filter[2], UV[i - 2:i + 3, j].detach().numpy())

            UV_xy[i, j, :] = torch.tensor([u_x, u_y])

    return UV_xy


# 通过filter计算导数生成物理损失
def generate_physics_loss_by_filter(last_state, pred_output, dx, dy, dt):
    u_last = last_state[:, 0:1].reshape(101, 101)
    v_last = last_state[:, 1:2].reshape(101, 101)

    u = pred_output[:, 0:1].reshape(101, 101)
    v = pred_output[:, 1:2].reshape(101, 101)

    dx_filter = 1 / (12 * dx) * np.array([
        [-25, 48, -36, 16, -3],
        [-3, -10, 18, -6, 1],
        [1, -8, 0, 8, -1],
        [-1, 6, -18, 10, 3],
        [3, -16, 36, -48, 25]
    ])

    dx2_filter = 1 / (12 * dx * dx) * np.array([
        [35, -104, 114, -56, 11],
        [11, -20, 6, 4, -1],
        [-1, 16, -30, 16, -1],
        [-1, 4, 6, -20, 11],
        [11, -56, 114, -104, 35]
    ])

    # generate_shape_function()
    # shape function方法也可以做 而且还成体系

    u_t = torch.sub(u, u_last) / dt
    v_t = torch.sub(v, v_last) / dt

    du_xy = cal_derivative_by_filter(dx_filter, u)
    dv_xy = cal_derivative_by_filter(dx_filter, v)

    laplace_u_x2y2 = cal_derivative_by_filter(dx2_filter, u)
    laplace_v_x2y2 = cal_derivative_by_filter(dx2_filter, v)

    u_x = du_xy[:, :, 0:1].reshape(101, 101)
    v_x = dv_xy[:, :, 0:1].reshape(101, 101)

    u_y = du_xy[:, :, 1:2].reshape(101, 101)
    v_y = dv_xy[:, :, 1:2].reshape(101, 101)

    laplace_u_x2 = laplace_u_x2y2[:, :, 0:1].reshape(101, 101)
    laplace_u_y2 = laplace_u_x2y2[:, :, 1:2].reshape(101, 101)
    laplace_v_x2 = laplace_v_x2y2[:, :, 0:1].reshape(101, 101)
    laplace_v_y2 = laplace_v_x2y2[:, :, 1:2].reshape(101, 101)

    laplace_u = torch.add(laplace_u_x2, laplace_u_y2)
    laplace_v = torch.add(laplace_v_x2, laplace_v_y2)

    # Burger's eqn
    nu = 1 / 200
    f_u = (u_t - nu * laplace_u + u * u_x + v * u_y) / 1
    f_v = (v_t - nu * laplace_v + u * v_x + v * v_y) / 1

    f_uv = torch.stack((f_u, f_v), dim=2)

    return f_uv


# 通过shape function计算生成一阶导数
def cal_one_order_derivative_by_shape_function(data, x, y):
    (u1, v1, x1, y1), (u2, v2, x2, y2), (u3, v3, x3, y3) = data[0], data[1], data[2]

    a1, a2, a3 = (x2 * y3 - x3 * y2), (x3 * y1 - x1 * y3), (x1 * y2 - x2 * y1)
    b1, b2, b3 = (y2 - y3), (y3 - y1), (y1 - y2)
    c1, c2, c3 = (x3 - x2), (x1 - x3), (x2 - x1)

    A_e = (a1 + a2 + a3) / 2

    L1 = (a1 + (b1) * x + (c1) * y) / (2 * A_e)
    L2 = (a2 + (b2) * x + (c2) * y) / (2 * A_e)
    L3 = (a3 + (b3) * x + (c3) * y) / (2 * A_e)
    # N1 = L1
    # N2 = L2
    # N3 = L3

    dN1_dx = b1 / (2 * A_e)
    dN2_dx = b2 / (2 * A_e)
    dN3_dx = b3 / (2 * A_e)

    dN1_dy = c1 / (2 * A_e)
    dN2_dy = c2 / (2 * A_e)
    dN3_dy = c3 / (2 * A_e)

    dU_dx = dN1_dx * u1 + dN2_dx * u2 + dN3_dx * u3
    dU_dy = dN1_dy * u1 + dN2_dy * u2 + dN3_dy * u3

    dV_dx = dN1_dx * v1 + dN2_dx * v2 + dN3_dx * v3
    dV_dy = dN1_dy * v1 + dN2_dy * v2 + dN3_dy * v3

    return dU_dx, dU_dy, dV_dx, dV_dy


def cal_two_order_derivative(uvxy, GraphConnectivity):
    # 待转成矩阵运算
    graphgrid_phy = torch.zeros([uvxy.shape[0], 12])
    for i in range(GraphConnectivity.shape[1]):
        [u1, v1, x1, y1] = uvxy[GraphConnectivity[0, i].type(torch.int64) - 1, :]
        [u2, v2, x2, y2] = uvxy[GraphConnectivity[1, i].type(torch.int64) - 1, :]
        [u3, v3, x3, y3] = uvxy[GraphConnectivity[2, i].type(torch.int64) - 1, :]
        [u4, v4, x4, y4] = uvxy[GraphConnectivity[3, i].type(torch.int64) - 1, :]
        [u5, v5, x5, y5] = uvxy[GraphConnectivity[4, i].type(torch.int64) - 1, :]
        [u6, v6, x6, y6] = uvxy[GraphConnectivity[5, i].type(torch.int64) - 1, :]
        data = [[u1, v1, x1, y1], [u2, v2, x2, y2], [u3, v3, x3, y3], [u4, v4, x4, y4], [u5, v5, x5, y5], [
            u6, v6, x6, y6]]

        graphgrid_phy[GraphConnectivity[0, i].type(torch.int64) - 1] += cal_two_order_derivative_by_shape_function(data,
                                                                                                                   x1,
                                                                                                                   y1)
        graphgrid_phy[GraphConnectivity[1, i].type(torch.int64) - 1] += cal_two_order_derivative_by_shape_function(data,
                                                                                                                   x2,
                                                                                                                   y2)
        graphgrid_phy[GraphConnectivity[2, i].type(torch.int64) - 1] += cal_two_order_derivative_by_shape_function(data,
                                                                                                                   x3,
                                                                                                                   y3)

    return graphgrid_phy


def cal_two_order_derivative_No_B_e(uvxy, GraphConnectivity):
    # 待转成矩阵运算
    graphgrid_phy = torch.zeros([uvxy.shape[0], 12])
    B = torch.zeros([uvxy.shape[0], 6, 6])
    for i in range(GraphConnectivity.shape[1]):
        [u1, v1, x1, y1] = uvxy[GraphConnectivity[0, i].type(torch.int64) - 1, :]
        [u2, v2, x2, y2] = uvxy[GraphConnectivity[1, i].type(torch.int64) - 1, :]
        [u3, v3, x3, y3] = uvxy[GraphConnectivity[2, i].type(torch.int64) - 1, :]
        [u4, v4, x4, y4] = uvxy[GraphConnectivity[3, i].type(torch.int64) - 1, :]
        [u5, v5, x5, y5] = uvxy[GraphConnectivity[4, i].type(torch.int64) - 1, :]
        [u6, v6, x6, y6] = uvxy[GraphConnectivity[5, i].type(torch.int64) - 1, :]
        data = [[u1, v1, x1, y1], [u2, v2, x2, y2], [u3, v3, x3, y3], [u4, v4, x4, y4], [u5, v5, x5, y5], [
            u6, v6, x6, y6]]

        d_e_u = torch.FloatTensor([[u1], [u2], [u3], [u4], [u5], [u6]])
        d_e_v = torch.FloatTensor([[v1], [v2], [v3], [v4], [v5], [v6]])
        B_e_1 = cal_two_order_derivative_of_shape_function_B_e(data, x1, y1)
        B_e_2 = cal_two_order_derivative_of_shape_function_B_e(data, x2, y2)
        B_e_3 = cal_two_order_derivative_of_shape_function_B_e(data, x3, y3)
        B_e_4 = cal_two_order_derivative_of_shape_function_B_e(data, x4, y4)
        B_e_5 = cal_two_order_derivative_of_shape_function_B_e(data, x5, y5)
        B_e_6 = cal_two_order_derivative_of_shape_function_B_e(data, x6, y6)

        B[GraphConnectivity[0, i].type(torch.int64) - 1] = B_e_1
        B[GraphConnectivity[1, i].type(torch.int64) - 1] = B_e_2
        B[GraphConnectivity[2, i].type(torch.int64) - 1] = B_e_3
        B[GraphConnectivity[3, i].type(torch.int64) - 1] = B_e_4
        B[GraphConnectivity[4, i].type(torch.int64) - 1] = B_e_5
        B[GraphConnectivity[5, i].type(torch.int64) - 1] = B_e_6

        # graphgrid_phy[GraphConnectivity[0, i].type(torch.int64) - 1] += torch.cat(
        #     [torch.mm(B_e_1, d_e_u), torch.mm(B_e_1, d_e_v)], dim=0).t().reshape(-1)
        # graphgrid_phy[GraphConnectivity[1, i].type(torch.int64) - 1] += torch.cat(
        #     [torch.mm(B_e_2, d_e_u), torch.mm(B_e_2, d_e_v)], dim=0).t().reshape(-1)
        # graphgrid_phy[GraphConnectivity[2, i].type(torch.int64) - 1] += torch.cat(
        #     [torch.mm(B_e_3, d_e_u), torch.mm(B_e_3, d_e_v)], dim=0).t().reshape(-1)

        graphgrid_phy[GraphConnectivity[0, i].type(torch.int64) - 1] += torch.cat(
            (torch.mm(B_e_1, d_e_u), torch.mm(B_e_1, d_e_v)), dim=0).t().reshape(-1)
        graphgrid_phy[GraphConnectivity[1, i].type(torch.int64) - 1] += torch.cat(
            (torch.mm(B_e_2, d_e_u), torch.mm(B_e_2, d_e_v)), dim=0).t().reshape(-1)
        graphgrid_phy[GraphConnectivity[2, i].type(torch.int64) - 1] += torch.cat(
            (torch.mm(B_e_3, d_e_u), torch.mm(B_e_3, d_e_v)), dim=0).t().reshape(-1)

    return B, graphgrid_phy


def cal_two_order_derivative_B_e(B, uvxy, GraphConnectivity):
    # 待转成矩阵运算
    graphgrid_phy = torch.zeros([uvxy.shape[0], 12])
    for i in range(GraphConnectivity.shape[1]):
        [u1, v1, x1, y1] = uvxy[GraphConnectivity[0, i].type(torch.int64) - 1, :]
        [u2, v2, x2, y2] = uvxy[GraphConnectivity[1, i].type(torch.int64) - 1, :]
        [u3, v3, x3, y3] = uvxy[GraphConnectivity[2, i].type(torch.int64) - 1, :]
        [u4, v4, x4, y4] = uvxy[GraphConnectivity[3, i].type(torch.int64) - 1, :]
        [u5, v5, x5, y5] = uvxy[GraphConnectivity[4, i].type(torch.int64) - 1, :]
        [u6, v6, x6, y6] = uvxy[GraphConnectivity[5, i].type(torch.int64) - 1, :]
        data = [[u1, v1, x1, y1], [u2, v2, x2, y2], [u3, v3, x3, y3], [u4, v4, x4, y4], [u5, v5, x5, y5], [
            u6, v6, x6, y6]]

        d_e_u = torch.FloatTensor([[u1], [u2], [u3], [u4], [u5], [u6]])
        d_e_v = torch.FloatTensor([[v1], [v2], [v3], [v4], [v5], [v6]])
        # B_e_1 = cal_two_order_derivative_of_shape_function_B_e(data, x1, y1)
        # B_e_2 = cal_two_order_derivative_of_shape_function_B_e(data, x2, y2)
        # B_e_3 = cal_two_order_derivative_of_shape_function_B_e(data, x3, y3)
        # B_e_4 = cal_two_order_derivative_of_shape_function_B_e(data, x4, y4)
        # B_e_5 = cal_two_order_derivative_of_shape_function_B_e(data, x5, y5)
        # B_e_6 = cal_two_order_derivative_of_shape_function_B_e(data, x6, y6)

        B_e_1 = B[GraphConnectivity[0, i].type(torch.int64) - 1]
        B_e_2 = B[GraphConnectivity[1, i].type(torch.int64) - 1]
        B_e_3 = B[GraphConnectivity[2, i].type(torch.int64) - 1]

        # graphgrid_phy[GraphConnectivity[0, i].type(torch.int64) - 1] += torch.cat(
        #     [torch.mm(B_e_1, d_e_u), torch.mm(B_e_1, d_e_v)], dim=0).t().reshape(-1)
        # graphgrid_phy[GraphConnectivity[1, i].type(torch.int64) - 1] += torch.cat(
        #     [torch.mm(B_e_2, d_e_u), torch.mm(B_e_2, d_e_v)], dim=0).t().reshape(-1)
        # graphgrid_phy[GraphConnectivity[2, i].type(torch.int64) - 1] += torch.cat(
        #     [torch.mm(B_e_3, d_e_u), torch.mm(B_e_3, d_e_v)], dim=0).t().reshape(-1)

        graphgrid_phy[GraphConnectivity[0, i].type(torch.int64) - 1] += torch.cat(
            (torch.mm(B_e_1, d_e_u), torch.mm(B_e_1, d_e_v)), dim=0).t().reshape(-1)
        graphgrid_phy[GraphConnectivity[1, i].type(torch.int64) - 1] += torch.cat(
            (torch.mm(B_e_2, d_e_u), torch.mm(B_e_2, d_e_v)), dim=0).t().reshape(-1)
        graphgrid_phy[GraphConnectivity[2, i].type(torch.int64) - 1] += torch.cat(
            (torch.mm(B_e_3, d_e_u), torch.mm(B_e_3, d_e_v)), dim=0).t().reshape(-1)

    return graphgrid_phy


# 通过shape function计算生成二阶导数
def cal_two_order_derivative_of_shape_function_B_e(data, x, y):
    # 待转成矩阵运算
    [u1, v1, x1, y1], [u2, v2, x2, y2], [u3, v3, x3, y3], \
        [u4, v4, x4, y4], [u5, v5, x5, y5], [u6, v6, x6, y6] = data[0], data[1], data[2], data[3], data[4], data[5]

    a1, a2, a3 = (x2 * y3 - x3 * y2), (x3 * y1 - x1 * y3), (x1 * y2 - x2 * y1)
    b1, b2, b3 = (y2 - y3), (y3 - y1), (y1 - y2)
    c1, c2, c3 = (x3 - x2), (x1 - x3), (x2 - x1)

    A_e = (a1 + a2 + a3) / 2

    L1 = (a1 + b1 * x + c1 * y) / (2 * A_e)
    L2 = (a2 + b2 * x + c2 * y) / (2 * A_e)
    L3 = (a3 + b3 * x + c3 * y) / (2 * A_e)
    L1_x = b1 / (2 * A_e)
    L2_x = b1 / (2 * A_e)
    L3_x = b2 / (2 * A_e)
    L1_y = c1 / (2 * A_e)
    L2_y = c2 / (2 * A_e)
    L3_y = c3 / (2 * A_e)
    L1_x2 = 0
    L2_x2 = 0
    L3_x2 = 0
    L1_y2 = 0
    L2_y2 = 0
    L3_y2 = 0
    L1_xy = 0
    L2_xy = 0
    L3_xy = 0
    L1_yx = 0
    L2_yx = 0
    L3_yx = 0

    # N1 = 2 * L1 * (L1 - 1 / 2)
    # N2 = 2 * L2 * (L2 - 1 / 2)
    # N3 = 2 * L3 * (L3 - 1 / 2)
    # N4 = 4 * L1 * L2
    # N5 = 4 * L3 * L2
    # N6 = 4 * L3 * L1

    dN1_dx = (4 * L1 - 1) * L1_x
    dN2_dx = (4 * L2 - 1) * L2_x
    dN3_dx = (4 * L3 - 1) * L3_x
    dN4_dx = 4 * (L2 * L1_x + L1 * L2_x)
    dN5_dx = 4 * (L2 * L3_x + L3 * L2_x)
    dN6_dx = 4 * (L3 * L1_x + L1 * L3_x)
    dN_dx = dN1_dx, dN2_dx, dN3_dx, dN4_dx, dN5_dx, dN6_dx

    dN1_dy = (4 * L1 - 1) * L1_y
    dN2_dy = (4 * L2 - 1) * L2_y
    dN3_dy = (4 * L3 - 1) * L3_y
    dN4_dy = 4 * (L2 * L1_y + L1 * L2_y)
    dN5_dy = 4 * (L2 * L3_y + L3 * L2_y)
    dN6_dy = 4 * (L2 * L3_y + L3 * L2_y)
    dN_dy = dN1_dy, dN2_dy, dN3_dy, dN4_dy, dN5_dy, dN6_dy

    dN1_dx2 = 4 * L1_x * L1_x + (4 * L1 - 1) * L1_x2
    dN2_dx2 = 4 * L2_x * L2_x + (4 * L2 - 1) * L2_x2
    dN3_dx2 = 4 * L3_x * L3_x + (4 * L3 - 1) * L3_x2
    dN4_dx2 = 4 * (L1_x2 * L2 + 2 * L1_x * L2_x + L1 * L2_x2)
    dN5_dx2 = 4 * (L3_x2 * L2 + 2 * L3_x * L2_x + L3 * L2_x2)
    dN6_dx2 = 4 * (L1_x2 * L3 + 2 * L1_x * L3_x + L1 * L3_x2)
    dN_dx2 = dN1_dx2, dN2_dx2, dN3_dx2, dN4_dx2, dN5_dx2, dN6_dx2

    dN1_dy2 = 4 * L1_y * L1_y + (4 * L1 - 1) * L1_y2
    dN2_dy2 = 4 * L2_y * L2_y + (4 * L2 - 1) * L2_y2
    dN3_dy2 = 4 * L3_y * L3_y + (4 * L3 - 1) * L3_y2
    dN4_dy2 = 4 * (L1_y2 * L2 + 2 * L1_y * L2_y + L1 * L2_y2)
    dN5_dy2 = 4 * (L3_y2 * L2 + 2 * L3_y * L2_y + L3 * L2_y2)
    dN6_dy2 = 4 * (L1_y2 * L3 + 2 * L1_y * L3_y + L1 * L3_y2)
    dN_dy2 = dN1_dy2, dN2_dy2, dN3_dy2, dN4_dy2, dN5_dy2, dN6_dy2

    dN1_dxdy = 4 * L1_y * L1_y + (4 * L1 - 1) * L1_xy
    dN2_dxdy = 4 * L2_y * L2_y + (4 * L2 - 1) * L2_xy
    dN3_dxdy = 4 * L3_y * L3_y + (4 * L3 - 1) * L3_xy
    dN4_dxdy = 4 * (L1_xy * L2 + L1_x * L2_y + L1_y * L2_x + L1 * L2_xy)
    dN5_dxdy = 4 * (L3_xy * L2 + L3_x * L2_y + L3_y * L2_x + L3 * L2_xy)
    dN6_dxdy = 4 * (L1_xy * L3 + L1_x * L3_y + L1_y * L3_x + L1 * L3_xy)
    dN_dxdy = dN1_dxdy, dN2_dxdy, dN3_dxdy, dN4_dxdy, dN5_dxdy, dN6_dxdy

    dN1_dydx = 4 * L1_y * L1_y + (4 * L1 - 1) * L1_yx
    dN2_dydx = 4 * L2_y * L2_y + (4 * L2 - 1) * L2_yx
    dN3_dydx = 4 * L3_y * L3_y + (4 * L3 - 1) * L3_yx
    dN4_dydx = 4 * (L1_yx * L2 + L1_x * L2_y + L1_y * L2_x + L1 * L2_yx)
    dN5_dydx = 4 * (L3_yx * L2 + L3_x * L2_y + L3_y * L2_x + L3 * L2_yx)
    dN6_dydx = 4 * (L1_yx * L3 + L1_x * L3_y + L1_y * L3_x + L1 * L3_yx)
    dN_dydx = dN1_dydx, dN2_dydx, dN3_dydx, dN4_dydx, dN5_dydx, dN6_dydx

    B_e = torch.FloatTensor([dN_dx, dN_dy, dN_dx2, dN_dy2, dN_dxdy, dN_dydx])

    # d_e = torch.FloatTensor([[u1], [v1], [u2], [v2], [u3], [v3], [u4], [v4], [u5], [v5], [u6], [v6]])
    # B_e_1 = cal_two_order_derivative_of_shape_function_B_e(data, x1, y1)
    # B_e_2 = cal_two_order_derivative_of_shape_function_B_e(data, x2, y2)
    # B_e_3 = cal_two_order_derivative_of_shape_function_B_e(data, x3, y3)
    # B_e_4 = cal_two_order_derivative_of_shape_function_B_e(data, x4, y4)
    # B_e_5 = cal_two_order_derivative_of_shape_function_B_e(data, x5, y5)
    # B_e_6 = cal_two_order_derivative_of_shape_function_B_e(data, x6, y6)
    # d_e = torch.FloatTensor([[u1,v1], [u2,v2], [u3,v3], [u4,v4], [u5,v5], [u6,v6]])
    # torch.mm(B_e, d_e)

    # graphgrid_phy=torch.FloatTensor(
    #     [dU_dx, dV_dx, dU_dy, dV_dy, dU_dx2, dV_dx2, dU_dy2, dV_dy2, dU_dxdy, dV_dxdy, dU_dydx, dV_dydx])
    return B_e


# 通过shape function计算生成二阶导数
def cal_two_order_derivative_by_shape_function(data, x, y):
    # 待转成矩阵运算
    [u1, v1, x1, y1], [u2, v2, x2, y2], [u3, v3, x3, y3], \
        [u4, v4, x4, y4], [u5, v5, x5, y5], [u6, v6, x6, y6] = data[0], data[1], data[2], data[3], data[4], data[5]

    a1, a2, a3 = (x2 * y3 - x3 * y2), (x3 * y1 - x1 * y3), (x1 * y2 - x2 * y1)
    b1, b2, b3 = (y2 - y3), (y3 - y1), (y1 - y2)
    c1, c2, c3 = (x3 - x2), (x1 - x3), (x2 - x1)

    A_e = (a1 + a2 + a3) / 2

    L1 = (a1 + b1 * x + c1 * y) / (2 * A_e)
    L2 = (a2 + b2 * x + c2 * y) / (2 * A_e)
    L3 = (a3 + b3 * x + c3 * y) / (2 * A_e)
    L1_x = b1 / (2 * A_e)
    L2_x = b1 / (2 * A_e)
    L3_x = b2 / (2 * A_e)
    L1_y = c1 / (2 * A_e)
    L2_y = c2 / (2 * A_e)
    L3_y = c3 / (2 * A_e)
    L1_x2 = 0
    L2_x2 = 0
    L3_x2 = 0
    L1_y2 = 0
    L2_y2 = 0
    L3_y2 = 0
    L1_xy = 0
    L2_xy = 0
    L3_xy = 0
    L1_yx = 0
    L2_yx = 0
    L3_yx = 0

    # N1 = 2 * L1 * (L1 - 1 / 2)
    # N2 = 2 * L2 * (L2 - 1 / 2)
    # N3 = 2 * L3 * (L3 - 1 / 2)
    # N4 = 4 * L1 * L2
    # N5 = 4 * L3 * L2
    # N6 = 4 * L3 * L1

    # B_e=torch.zeros([uvxy.shape[0], 12, 6])

    dN1_dx = (4 * L1 - 1) * L1_x
    dN2_dx = (4 * L2 - 1) * L2_x
    dN3_dx = (4 * L3 - 1) * L3_x
    dN4_dx = 4 * (L2 * L1_x + L1 * L2_x)
    dN5_dx = 4 * (L2 * L3_x + L3 * L2_x)
    dN6_dx = 4 * (L3 * L1_x + L1 * L3_x)

    dN1_dy = (4 * L1 - 1) * L1_y
    dN2_dy = (4 * L2 - 1) * L2_y
    dN3_dy = (4 * L3 - 1) * L3_y
    dN4_dy = 4 * (L2 * L1_y + L1 * L2_y)
    dN5_dy = 4 * (L2 * L3_y + L3 * L2_y)
    dN6_dy = 4 * (L2 * L3_y + L3 * L2_y)

    dN1_dx2 = 4 * L1_x * L1_x + (4 * L1 - 1) * L1_x2
    dN2_dx2 = 4 * L2_x * L2_x + (4 * L2 - 1) * L2_x2
    dN3_dx2 = 4 * L3_x * L3_x + (4 * L3 - 1) * L3_x2
    dN4_dx2 = 4 * (L1_x2 * L2 + 2 * L1_x * L2_x + L1 * L2_x2)
    dN5_dx2 = 4 * (L3_x2 * L2 + 2 * L3_x * L2_x + L3 * L2_x2)
    dN6_dx2 = 4 * (L1_x2 * L3 + 2 * L1_x * L3_x + L1 * L3_x2)

    dN1_dy2 = 4 * L1_y * L1_y + (4 * L1 - 1) * L1_y2
    dN2_dy2 = 4 * L2_y * L2_y + (4 * L2 - 1) * L2_y2
    dN3_dy2 = 4 * L3_y * L3_y + (4 * L3 - 1) * L3_y2
    dN4_dy2 = 4 * (L1_y2 * L2 + 2 * L1_y * L2_y + L1 * L2_y2)
    dN5_dy2 = 4 * (L3_y2 * L2 + 2 * L3_y * L2_y + L3 * L2_y2)
    dN6_dy2 = 4 * (L1_y2 * L3 + 2 * L1_y * L3_y + L1 * L3_y2)

    dN1_dxdy = 4 * L1_y * L1_y + (4 * L1 - 1) * L1_xy
    dN2_dxdy = 4 * L2_y * L2_y + (4 * L2 - 1) * L2_xy
    dN3_dxdy = 4 * L3_y * L3_y + (4 * L3 - 1) * L3_xy
    dN4_dxdy = 4 * (L1_xy * L2 + L1_x * L2_y + L1_y * L2_x + L1 * L2_xy)
    dN5_dxdy = 4 * (L3_xy * L2 + L3_x * L2_y + L3_y * L2_x + L3 * L2_xy)
    dN6_dxdy = 4 * (L1_xy * L3 + L1_x * L3_y + L1_y * L3_x + L1 * L3_xy)

    dN1_dydx = 4 * L1_y * L1_y + (4 * L1 - 1) * L1_yx
    dN2_dydx = 4 * L2_y * L2_y + (4 * L2 - 1) * L2_yx
    dN3_dydx = 4 * L3_y * L3_y + (4 * L3 - 1) * L3_yx
    dN4_dydx = 4 * (L1_yx * L2 + L1_x * L2_y + L1_y * L2_x + L1 * L2_yx)
    dN5_dydx = 4 * (L3_yx * L2 + L3_x * L2_y + L3_y * L2_x + L3 * L2_yx)
    dN6_dydx = 4 * (L1_yx * L3 + L1_x * L3_y + L1_y * L3_x + L1 * L3_yx)

    dU_dx = dN1_dx * u1 + dN2_dx * u2 + dN3_dx * u3 + dN4_dx * u4 + dN5_dx * u5 + dN6_dx * u6
    dV_dx = dN1_dx * v1 + dN2_dx * v2 + dN3_dx * v3 + dN4_dx * v4 + dN5_dx * v5 + dN6_dx * v6

    dU_dy = dN1_dy * u1 + dN2_dy * u2 + dN3_dy * u3 + dN4_dy * u4 + dN5_dy * u5 + dN6_dy * u6
    dV_dy = dN1_dy * v1 + dN2_dy * v2 + dN3_dy * v3 + dN4_dy * v4 + dN5_dy * v5 + dN6_dy * v6

    dU_dx2 = dN1_dx2 * u1 + dN2_dx2 * u2 + dN3_dx2 * u3 + dN4_dx2 * u4 + dN5_dx2 * u5 + dN6_dx2 * u6
    dV_dx2 = dN1_dx2 * v1 + dN2_dx2 * v2 + dN3_dx2 * v3 + dN4_dx2 * v4 + dN5_dx2 * v5 + dN6_dx2 * v6

    dU_dy2 = dN1_dy2 * u1 + dN2_dy2 * u2 + dN3_dy2 * u3 + dN4_dy2 * u4 + dN5_dy2 * u5 + dN6_dy2 * u6
    dV_dy2 = dN1_dy2 * v1 + dN2_dy2 * v2 + dN3_dy2 * v3 + dN4_dy2 * v4 + dN5_dy2 * v5 + dN6_dy2 * v6

    dU_dxdy = dN1_dxdy * u1 + dN2_dxdy * u2 + dN3_dxdy * u3 + dN4_dxdy * u4 + dN5_dxdy * u5 + dN6_dxdy * u6
    dV_dxdy = dN1_dxdy * v1 + dN2_dxdy * v2 + dN3_dxdy * v3 + dN4_dxdy * v4 + dN5_dxdy * v5 + dN6_dxdy * v6

    dU_dydx = dN1_dydx * u1 + dN2_dydx * u2 + dN3_dydx * u3 + dN4_dydx * u4 + dN5_dydx * u5 + dN6_dydx * u6
    dV_dydx = dN1_dydx * v1 + dN2_dydx * v2 + dN3_dydx * v3 + dN4_dydx * v4 + dN5_dydx * v5 + dN6_dydx * v6

    return torch.FloatTensor(
        [dU_dx, dV_dx, dU_dy, dV_dy, dU_dx2, dV_dx2, dU_dy2, dV_dy2, dU_dxdy, dV_dxdy, dU_dydx, dV_dydx])


# 通过shape function生成物理损失
def generate_physics_loss_by_shape_function(dt, encoded, processed, GraphConnectivity, GraphGrid):
    # u_last = last_state[:, 0:1].reshape(args.height, args.width)
    # v_last = last_state[:, 1:2].reshape(args.height, args.width)
    #
    # u = pred_output[:, 0:1].reshape(args.height, args.width)
    # v = pred_output[:, 1:2].reshape(args.height, args.width)
    #
    # u_t = torch.sub(u, u_last) / args.dt
    # v_t = torch.sub(v, v_last) / args.dt

    GraphGrid_T = GraphGrid[1:, :].t()
    process_uvxy = torch.cat((processed[:, 0:2], GraphGrid_T), axis=1)

    graphgrid_phy = cal_two_order_derivative(process_uvxy, GraphConnectivity)

    # dU_dx, dV_dx, dU_dy, dV_dy, dU_dx2, dV_dx2, dU_dy2, dV_dy2, dU_dxdy, dV_dxdy, dU_dydx, dV_dydx
    U = processed[:, 0:1]
    V = processed[:, 1:2]

    U_last = encoded[:, 0:1]
    V_last = encoded[:, 1:2]

    # U_t = torch.sub(U, U_last) / dt
    # V_t = torch.sub(V, V_last) / dt

    U_t = torch.sub(U, U_last)
    V_t = torch.sub(V, V_last)

    dU_dx = graphgrid_phy[:, 0:1]
    dV_dx = graphgrid_phy[:, 1:2]

    dU_dy = graphgrid_phy[:, 2:3]
    dV_dy = graphgrid_phy[:, 3:4]

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]

    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    # Burger's eqn
    nu = 1 / 200

    f_u = (U_t - nu * laplace_u + torch.mul(U, dU_dx) + torch.mul(V, dU_dy)) / 1
    f_v = (V_t - nu * laplace_v + torch.mul(U, dV_dx) + torch.mul(V, dV_dy)) / 1

    # f_u = ( - nu * laplace_u + torch.mul(U, dU_dx) + torch.mul(V, dU_dy)) / 1
    # f_v = ( - nu * laplace_v + torch.mul(U, dV_dx) + torch.mul(V, dV_dy)) / 1

    f_uv = torch.stack((f_u, f_v), dim=2)

    return f_uv


# 通过shape function生成物理损失
def generate_semiphysics_loss_by_shape_function_B_e(args, encoded, processed, GraphConnectivity, GraphGrid):
    nu = 1 / 120
    GraphGrid_T = GraphGrid[1:, :].t()
    # process_uvxy = torch.cat((processed[:, 0:2], GraphGrid_T), axis=1)
    encoder_uvxy = torch.cat((encoded, GraphGrid_T), axis=1)

    # 未固定B_e值
    # graphgrid_phy = cal_two_order_derivative(process_uvxy, GraphConnectivity)
    # 固定B_e值
    if os.path.exists(args.data_save_path + "B.npy"):
        B = torch.from_numpy(np.load(args.data_save_path + "B.npy"))
        graphgrid_phy = cal_two_order_derivative_B_e(B, encoder_uvxy, GraphConnectivity)
    else:
        B, graphgrid_phy = cal_two_order_derivative_No_B_e(encoder_uvxy, GraphConnectivity)
        np.save(args.data_save_path + "B.npy", B)

    """
    # 12个 dU_dx, dV_dx, dU_dy, dV_dy, dU_dx2, dV_dx2, dU_dy2, dV_dy2, dU_dxdy, dV_dxdy, dU_dydx, dV_dydx
    U = encoded[:, 0:1]
    V = encoded[:, 1:2]

    dU_dx = graphgrid_phy[:, 0:1]
    dU_dy = graphgrid_phy[:, 1:2]
    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]

    dV_dx = graphgrid_phy[:, 6:7]
    dV_dy = graphgrid_phy[:, 7:8]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    # Burger's eqn
    nu = 1 / 200

    # f_u = (U_t - nu * laplace_u + torch.mul(U, dU_dx) + torch.mul(V, dU_dy)) / 1
    # f_v = (V_t - nu * laplace_v + torch.mul(U, dV_dx) + torch.mul(V, dV_dy)) / 1

    # f_u = (nu * laplace_u - torch.mul(U, dU_dx) - torch.mul(V, dU_dy))* args.dt
    # f_v = (nu * laplace_v - torch.mul(U, dV_dx) - torch.mul(V, dV_dy))* args.dt

    # f_u = (nu * laplace_u) * args.dt
    # f_v = (nu * laplace_v) * args.dt

    # f_uv = torch.cat((f_u, f_v), dim=1)
    """
    # RK4
    ################################# stage 0 ############################
    U_0 = encoded[:, 0:1]
    V_0 = encoded[:, 1:2]

    dU_dx = graphgrid_phy[:, 0:1]
    dU_dy = graphgrid_phy[:, 1:2]
    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]

    dV_dx = graphgrid_phy[:, 6:7]
    dV_dy = graphgrid_phy[:, 7:8]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    u_stage_0 = (nu * laplace_u - torch.mul(U_0, dU_dx) - torch.mul(V_0, dU_dy)) / 1
    v_stage_0 = (nu * laplace_v - torch.mul(U_0, dV_dx) - torch.mul(V_0, dV_dy)) / 1
    ################################# stage 1 ############################
    U_1 = U_0 + u_stage_0 * args.dt / 2.0
    V_1 = V_0 + v_stage_0 * args.dt / 2.0

    encoded = torch.cat((U_1, V_1), axis=1)
    encoder_uvxy = torch.cat((encoded, GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative_B_e(B, encoder_uvxy, GraphConnectivity)

    dU_dx = graphgrid_phy[:, 0:1]
    dU_dy = graphgrid_phy[:, 1:2]
    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]

    dV_dx = graphgrid_phy[:, 6:7]
    dV_dy = graphgrid_phy[:, 7:8]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    u_stage_1 = (nu * laplace_u - torch.mul(U_1, dU_dx) - torch.mul(V_1, dU_dy)) / 1
    v_stage_1 = (nu * laplace_v - torch.mul(U_1, dV_dx) - torch.mul(V_1, dV_dy)) / 1
    ################################# stage 2 ############################
    U_2 = U_1 + u_stage_1 * args.dt / 2.0
    V_2 = V_1 + v_stage_1 * args.dt / 2.0

    encoded = torch.cat((U_2, V_2), axis=1)
    encoder_uvxy = torch.cat((encoded, GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative_B_e(B, encoder_uvxy, GraphConnectivity)

    dU_dx = graphgrid_phy[:, 0:1]
    dU_dy = graphgrid_phy[:, 1:2]
    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]

    dV_dx = graphgrid_phy[:, 6:7]
    dV_dy = graphgrid_phy[:, 7:8]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    u_stage_2 = (nu * laplace_u - torch.mul(U_2, dU_dx) - torch.mul(V_2, dU_dy)) / 1
    v_stage_2 = (nu * laplace_v - torch.mul(U_2, dV_dx) - torch.mul(V_2, dV_dy)) / 1
    ################################# stage 3 ############################
    U_3 = U_2 + u_stage_2 * args.dt / 2.0
    V_3 = V_2 + v_stage_2 * args.dt / 2.0

    encoded = torch.cat((U_3, V_3), axis=1)
    encoder_uvxy = torch.cat((encoded, GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative_B_e(B, encoder_uvxy, GraphConnectivity)

    dU_dx = graphgrid_phy[:, 0:1]
    dU_dy = graphgrid_phy[:, 1:2]
    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]

    dV_dx = graphgrid_phy[:, 6:7]
    dV_dy = graphgrid_phy[:, 7:8]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    u_stage_3 = (nu * laplace_u - torch.mul(U_3, dU_dx) - torch.mul(V_3, dU_dy)) / 1
    v_stage_3 = (nu * laplace_v - torch.mul(U_3, dV_dx) - torch.mul(V_3, dV_dy)) / 1
    ################################# stage 3 ############################
    f_u = args.dt * (u_stage_0 + 2 * u_stage_1 + 2 * u_stage_2 + u_stage_3) / 6.0
    f_v = args.dt * (v_stage_0 + 2 * v_stage_1 + 2 * v_stage_2 + v_stage_3) / 6.0

    f_uv = torch.cat((f_u, f_v), dim=1)

    return f_uv


# 通过shape function生成物理损失
def generate_semiphysics_loss_by_shape_function(args, encoded, processed, GraphConnectivity, GraphGrid):
    GraphGrid_T = GraphGrid[1:, :].t()
    process_uvxy = torch.cat((processed[:, 0:2], GraphGrid_T), axis=1)

    # 未固定B_e值
    # graphgrid_phy = cal_two_order_derivative(process_uvxy, GraphConnectivity)
    # 固定B_e值
    graphgrid_phy = cal_two_order_derivative_B_e(process_uvxy, GraphConnectivity)

    # 12个 dU_dx, dV_dx, dU_dy, dV_dy, dU_dx2, dV_dx2, dU_dy2, dV_dy2, dU_dxdy, dV_dxdy, dU_dydx, dV_dydx
    #
    U = processed[:, 0:1]
    V = processed[:, 1:2]

    # U_last = encoded[:, 0:1]
    # V_last = encoded[:, 1:2]

    # U_t = torch.sub(U, U_last)
    # V_t = torch.sub(V, V_last)

    dU_dx = graphgrid_phy[:, 0:1]
    dV_dx = graphgrid_phy[:, 1:2]

    dU_dy = graphgrid_phy[:, 2:3]
    dV_dy = graphgrid_phy[:, 3:4]

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]

    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    # Burger's eqn
    nu = 1 / 200

    # f_u = (U_t - nu * laplace_u + torch.mul(U, dU_dx) + torch.mul(V, dU_dy)) / 1
    # f_v = (V_t - nu * laplace_v + torch.mul(U, dV_dx) + torch.mul(V, dV_dy)) / 1

    # f_u = (nu * laplace_u - torch.mul(U, dU_dx) - torch.mul(V, dU_dy)) / 1
    # f_v = (nu * laplace_v - torch.mul(U, dV_dx) - torch.mul(V, dV_dy)) / 1

    # f_u = (nu * laplace_u - torch.mul(U, dU_dx) - torch.mul(V, dU_dy)) * args.dt
    # f_v = (nu * laplace_v - torch.mul(U, dV_dx) - torch.mul(V, dV_dy)) * args.dt

    # semi_physics
    f_u = (nu * laplace_u) * args.dt
    f_v = (nu * laplace_v) * args.dt

    f_uv = torch.stack((f_u, f_v), dim=2).reshape(-1, args.feature_dim)

    return f_uv


# 通过shape function生成物理损失
def generate_grayscott_physics_loss_by_shape_function_B_e(args, encoded, processed, GraphConnectivity, GraphGrid):
    # Gray-Scott's eqn
    Du = 0.16  # 2*10**-5
    Dv = 0.08  # DA/4
    F = 0.06  # 1/25
    K = 0.062  # 3/50
    GraphGrid_T = GraphGrid[1:, :].t()
    # encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)
    processed_uvxy = torch.cat((processed[:, 0:2], GraphGrid_T), axis=1)

    # 固定B_e值
    if os.path.exists(args.data_save_path + "B.npy"):
        B = torch.from_numpy(np.load(args.data_save_path + "B.npy"))
        graphgrid_phy = cal_two_order_derivative_B_e(B, processed_uvxy, GraphConnectivity)
    else:
        B, graphgrid_phy = cal_two_order_derivative_No_B_e(processed_uvxy, GraphConnectivity)
        np.save(args.data_save_path + "B.npy", B)

    U = processed[:, 0:1]
    V = processed[:, 1:2]

    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    V2 = torch.mul(V, V)

    # args.dt *
    # f_u = (Du * laplace_u - torch.mul(U, V2) + F * (torch.ones_like(U) - U))
    # f_v = (Dv * laplace_v + torch.mul(U, V2) - (F + K) * V)
    f_u = (Du * laplace_u)
    f_v = (Dv * laplace_v)

    f_uv = torch.cat((f_u, f_v), dim=1)

    return f_uv


# 通过shape function生成物理损失
def generate_grayscott_physics_loss_by_shape_function(dt, encoded, processed, GraphConnectivity, GraphGrid):
    # Gray-Scott's eqn
    Du = 0.16  # 2*10**-5
    Dv = 0.08  # DA/4
    F = 0.06  # 1/25
    K = 0.062  # 3/50
    GraphGrid_T = GraphGrid[1:, :].t()

    # RK4
    ################################# stage 0 ############################
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)

    graphgrid_phy = cal_two_order_derivative(encoder_uvxy, GraphConnectivity)

    # dU_dx, dV_dx, dU_dy, dV_dy, dU_dx2, dV_dx2, dU_dy2, dV_dy2, dU_dxdy, dV_dxdy, dU_dydx, dV_dydx
    U = encoded[:, 0:1]
    V = encoded[:, 1:2]

    dU_dx = graphgrid_phy[:, 0:1]
    dV_dx = graphgrid_phy[:, 1:2]

    dU_dy = graphgrid_phy[:, 2:3]
    dV_dy = graphgrid_phy[:, 3:4]

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]

    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    V2 = torch.mul(V, V)

    u_stage_0 = (Du * laplace_u - torch.mul(U, V2) + F * (torch.ones_like(U) - U)) / 1
    v_stage_0 = (Dv * laplace_v + torch.mul(U, V2) - (F + K) * V) / 1

    ################################# stage 1 ############################
    encoded = torch.cat((U, V), axis=1)
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)

    graphgrid_phy = cal_two_order_derivative(encoder_uvxy, GraphConnectivity)

    U = U + u_stage_0 * dt / 2.0
    V = V + v_stage_0 * dt / 2.0

    dU_dx = graphgrid_phy[:, 0:1]
    dV_dx = graphgrid_phy[:, 1:2]

    dU_dy = graphgrid_phy[:, 2:3]
    dV_dy = graphgrid_phy[:, 3:4]

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]

    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    V2 = torch.mul(V, V)

    u_stage_1 = (Du * laplace_u - torch.mul(U, V2) + F * (torch.ones_like(U) - U)) / 1
    v_stage_1 = (Dv * laplace_v + torch.mul(U, V2) - (F + K) * V) / 1
    ################################# stage 2 ############################

    encoded = torch.cat((U, V), axis=1)
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)

    graphgrid_phy = cal_two_order_derivative(encoder_uvxy, GraphConnectivity)

    U = U + u_stage_1 * dt / 2.0
    V = V + v_stage_1 * dt / 2.0

    dU_dx = graphgrid_phy[:, 0:1]
    dV_dx = graphgrid_phy[:, 1:2]

    dU_dy = graphgrid_phy[:, 2:3]
    dV_dy = graphgrid_phy[:, 3:4]

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]

    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    V2 = torch.mul(V, V)

    u_stage_2 = (Du * laplace_u - torch.mul(U, V2) + F * (torch.ones_like(U) - U)) / 1
    v_stage_2 = (Dv * laplace_v + torch.mul(U, V2) - (F + K) * V) / 1

    ################################# stage 3 ############################

    encoded = torch.cat((U, V), axis=1)
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)

    graphgrid_phy = cal_two_order_derivative(encoder_uvxy, GraphConnectivity)

    U = U + u_stage_2 * dt / 2.0
    V = V + v_stage_2 * dt / 2.0

    dU_dx = graphgrid_phy[:, 0:1]
    dV_dx = graphgrid_phy[:, 1:2]

    dU_dy = graphgrid_phy[:, 2:3]
    dV_dy = graphgrid_phy[:, 3:4]

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]

    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    V2 = torch.mul(V, V)

    u_stage_3 = (Du * laplace_u - torch.mul(U, V2) + F * (torch.ones_like(U) - U)) / 1
    v_stage_3 = (Dv * laplace_v + torch.mul(U, V2) - (F + K) * V) / 1

    U = encoded[:, 0:1] + dt * (u_stage_0 + 2 * u_stage_1 + 2 * u_stage_2 + u_stage_3) / 6.0
    V = encoded[:, 1:2] + dt * (v_stage_0 + 2 * v_stage_1 + 2 * v_stage_2 + v_stage_3) / 6.0

    f_u = processed[:, 0:1] - U
    f_v = processed[:, 1:2] - V

    # f_u = (U_t - Du * laplace_u + torch.mul(U, V2) - F * (torch.ones_like(U) - U)) / 1
    # f_v = (V_t - Dv * laplace_v - torch.mul(U, V2) + (F + K) * V) / 1
    f_uv = torch.stack((f_u, f_v), dim=2)

    return f_uv


# 通过shape function生成物理损失
def generate_FN_physics_loss_by_shape_function_B_e(args, encoded, processed, GraphConnectivity, GraphGrid):
    # FN's eqn
    Du = 1.0
    Dv = 100.0
    alpha = 0.01
    beta = 0.25
    GraphGrid_T = GraphGrid[1:, :].t()
    # encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)
    processed_uvxy = torch.cat((processed[:, 0:2], GraphGrid_T), axis=1)

    # 固定B_e值
    if os.path.exists(args.data_save_path + "B.npy"):
        B = torch.from_numpy(np.load(args.data_save_path + "B.npy"))
        graphgrid_phy = cal_two_order_derivative_B_e(B, processed_uvxy, GraphConnectivity)
    else:
        B, graphgrid_phy = cal_two_order_derivative_No_B_e(processed_uvxy, GraphConnectivity)
        np.save(args.data_save_path + "B.npy", B)

    U = processed[:, 0:1]
    V = processed[:, 1:2]

    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    U2 = torch.mul(U, U)
    U3 = torch.mul(U2, U)

    # args.dt *
    # f_u = processed[:, 0:1] - encoded[:, 0:1] + args.dt *(Du * laplace_u + U - U3 - V + alpha * torch.ones_like(U))
    # f_v = processed[:, 1:2] - encoded[:, 1:2] + args.dt *(Dv * laplace_v + (U - V) * beta)

    # f_u =  - encoded[:, 0:1] + args.dt * (Du * laplace_u)
    # f_v =  - encoded[:, 1:2] + args.dt * (Dv * laplace_v)

    f_u = args.dt * (Du * laplace_u + U - U3 - V + alpha * torch.ones_like(U))
    f_v = args.dt * (Dv * laplace_v + (U - V) * beta)

    #  args.dt *
    # f_u =(Du * laplace_u )
    # f_v =  (Dv * laplace_v )

    # f_uv = torch.stack((f_u, f_v), dim=2).reshape(-1, args.feature_dim)
    f_uv = torch.cat((f_u, f_v), dim=1)

    return f_uv


"""
# 通过shape function生成物理损失
def generate_FN_physics_loss_by_shape_function_B_e(args, encoded, GraphConnectivity, GraphGrid):
    # FN's eqn
    Du = 1.0
    Dv = 100.0
    alpha = 0.01
    beta = 0.25
    GraphGrid_T = GraphGrid[1:, :].t()
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)

    # 固定B_e值
    if os.path.exists(args.data_save_path + "B.npy"):
        B = torch.from_numpy(np.load(args.data_save_path + "B.npy"))
        graphgrid_phy = cal_two_order_derivative_B_e(B, encoder_uvxy, GraphConnectivity)
    else:
        B, graphgrid_phy = cal_two_order_derivative_No_B_e(encoder_uvxy, GraphConnectivity)
        np.save(args.data_save_path + "B.npy", B)

    U_0 = encoded[:, 0:1]
    V_0 = encoded[:, 1:2]
    # RK4
    ################################# stage 0 ############################

    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    u_stage_0 = Du * laplace_u
    v_stage_0 = Dv * laplace_v
    ################################# stage 1 ############################
    U_1 = U_0 + u_stage_0 * args.dt / 2.0
    V_1 = V_0 + v_stage_0 * args.dt / 2.0

    encoded = torch.cat((U_1, V_1), axis=1)
    encoder_uvxy = torch.cat((encoded, GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative_B_e(B, encoder_uvxy, GraphConnectivity)

    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    u_stage_1 = Du * laplace_u
    v_stage_1 = Dv * laplace_v
    ################################# stage 2 ############################
    U_2 = U_0 + u_stage_1 * args.dt / 2.0
    V_2 = V_0 + v_stage_1 * args.dt / 2.0

    encoded = torch.cat((U_2, V_2), axis=1)
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative_B_e(B, encoder_uvxy, GraphConnectivity)

    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    u_stage_2 = Du * laplace_u
    v_stage_2 = Dv * laplace_v
    ################################# stage 3 ############################
    U_3 = U_0 + u_stage_2 * args.dt / 2.0
    V_3 = V_0 + v_stage_2 * args.dt / 2.0

    encoded = torch.cat((U_3, V_3), axis=1)
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative_B_e(B, encoder_uvxy, GraphConnectivity)

    dU_dx2 = graphgrid_phy[:, 2:3]
    dU_dy2 = graphgrid_phy[:, 3:4]
    dV_dx2 = graphgrid_phy[:, 8:9]
    dV_dy2 = graphgrid_phy[:, 9:10]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    u_stage_3 = Du * laplace_u
    v_stage_3 = Dv * laplace_v
    ################################# stage 3 ############################

    f_u = args.dt * (u_stage_0 + 2 * u_stage_1 + 2 * u_stage_2 + u_stage_3) / 6.0
    f_v = args.dt * (v_stage_0 + 2 * v_stage_1 + 2 * v_stage_2 + v_stage_3) / 6.0

    # f_u = processed[:, 0:1] - U_last + U_t
    # f_v = processed[:, 1:2] - V_last + V_t

    # f_u = (U_t - Du * laplace_u + torch.mul(U, V2) - F * (torch.ones_like(U) - U)) / 1
    # f_v = (V_t - Dv * laplace_v - torch.mul(U, V2) + (F + K) * V) / 1

    f_uv = torch.cat((f_u, f_v), dim=1)

    return f_uv
"""


# 通过shape function生成物理损失
def generate_FN_physics_loss_by_shape_function(dt, encoded, processed, GraphConnectivity, GraphGrid):
    # FN's eqn
    Du = 1.0
    Dv = 100.0
    alpha = 0.01
    beta = 0.25
    GraphGrid_T = GraphGrid[1:, :].t()

    U_last = encoded[:, 0:1]
    V_last = encoded[:, 1:2]

    # RK4
    ################################# stage 0 ############################
    U = U_last
    V = V_last

    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative(encoder_uvxy, GraphConnectivity)

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]
    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    U2 = torch.mul(U, U)
    U3 = torch.mul(U2, U)

    u_stage_0 = Du * laplace_u + U - U3 - V + alpha * torch.ones_like(U)
    v_stage_0 = Dv * laplace_v + (U - V) * beta

    ################################# stage 1 ############################
    U = U_last + u_stage_0 * dt / 2.0
    V = V_last + v_stage_0 * dt / 2.0

    encoded = torch.cat((U, V), axis=1)
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative(encoder_uvxy, GraphConnectivity)

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]
    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    U2 = torch.mul(U, U)
    U3 = torch.mul(U2, U)

    u_stage_1 = Du * laplace_u + U - U3 - V + alpha * torch.ones_like(U)
    v_stage_1 = Dv * laplace_v + (U - V) * beta
    ################################# stage 2 ############################
    U = U_last + u_stage_1 * dt / 2.0
    V = V_last + v_stage_1 * dt / 2.0

    encoded = torch.cat((U, V), axis=1)
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative(encoder_uvxy, GraphConnectivity)

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]
    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    U2 = torch.mul(U, U)
    U3 = torch.mul(U2, U)

    u_stage_2 = Du * laplace_u + U - U3 - V + alpha * torch.ones_like(U)
    v_stage_2 = Dv * laplace_v + (U - V) * beta
    ################################# stage 3 ############################
    U = U_last + u_stage_2 * dt / 2.0
    V = V_last + v_stage_2 * dt / 2.0

    encoded = torch.cat((U, V), axis=1)
    encoder_uvxy = torch.cat((encoded[:, 0:2], GraphGrid_T), axis=1)
    graphgrid_phy = cal_two_order_derivative(encoder_uvxy, GraphConnectivity)

    dU_dx2 = graphgrid_phy[:, 4:5]
    dV_dx2 = graphgrid_phy[:, 5:6]
    dU_dy2 = graphgrid_phy[:, 6:7]
    dV_dy2 = graphgrid_phy[:, 7:8]

    laplace_u = torch.add(dU_dx2, dU_dy2)
    laplace_v = torch.add(dV_dx2, dV_dy2)

    U2 = torch.mul(U, U)
    U3 = torch.mul(U2, U)

    u_stage_3 = Du * laplace_u + U - U3 - V + alpha * torch.ones_like(U)
    v_stage_3 = Dv * laplace_v + (U - V) * beta
    ################################# stage 4 ############################
    U_t = dt * (u_stage_0 + 2 * u_stage_1 + 2 * u_stage_2 + u_stage_3) / 6.0
    V_t = dt * (v_stage_0 + 2 * v_stage_1 + 2 * v_stage_2 + v_stage_3) / 6.0

    f_u = processed[:, 0:1] - U_last + U_t
    f_v = processed[:, 1:2] - V_last + V_t

    # f_u = (U_t - Du * laplace_u + torch.mul(U, V2) - F * (torch.ones_like(U) - U)) / 1
    # f_v = (V_t - Dv * laplace_v - torch.mul(U, V2) + (F + K) * V) / 1
    f_uv = torch.stack((f_u, f_v), dim=2)

    return f_uv


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)  # parameters的封装使得变量可以容易访问到

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            # loss_sum += 0.5 * torch.exp(-log_vars[i]) * loss + self.params[i]
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            # +1避免了log 0的问题  log sigma部分对于整体loss的影响不大
        return loss_sum


class AutomaticWeight(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(AutomaticWeight, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


#####################################################
# 分割线 grid
#####################################################


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
    U = U.detach().numpy()
    V = V.detach().numpy()

    laplace_u = apply_laplacian(U, dx)
    laplace_v = apply_laplacian(V, dx)

    u_x = apply_dx(U, dx)
    v_x = apply_dx(V, dx)

    u_y = apply_dy(U, dx)
    v_y = apply_dy(V, dx)

    # governing equation
    # .R.detach().numpy()

    u_t = (1.0 / R) * torch.from_numpy(laplace_u) - torch.from_numpy(U) * torch.from_numpy(u_x) - torch.from_numpy(
        V) * torch.from_numpy(u_y)
    v_t = (1.0 / R) * torch.from_numpy(laplace_v) - torch.from_numpy(U) * torch.from_numpy(v_x) - torch.from_numpy(
        V) * torch.from_numpy(v_y)

    return u_t, v_t


def update(U0, V0, R=120.0, dt=0.05, dx=1.0):
    U0 = U0.detach()
    V0 = V0.detach()

    u_t, v_t = get_temporal_diff(U0, V0, R, dx)

    U = U0 + dt * u_t
    V = V0 + dt * v_t

    return U, V


def update_rk4(U0, V0, R=120.0, dt=0.05, dx=1.0):
    """Update with Runge-kutta-4 method
       See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    # U0 = U0.detach()
    # V0 = V0.detach()

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

    # print("dt", dt)
    # print("dx", dx)
    # print("R", R)

    return U, V


#####################################################
# 分割线
#####################################################
# 给数据补充pos
def get_pos():
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    x_star, y_star = np.meshgrid(x, y)
    print(x_star.shape)
    print(y_star.shape)
    xy_star = torch.stack((torch.from_numpy(x_star), torch.from_numpy(y_star)), axis=2)
    print(xy_star.shape)


def is_near(x, y, eps=1.0e-16):
    x = np.array(x)
    y = np.array(y)
    for yi in y:
        if np.linalg.norm(x - yi) < eps:
            return True
    return False


# import networkx as nx

# def draw(data):
#     G = to_networkx(data)
#     nx.draw(G)
#     plt.savefig("path.png")
#     plt.show()


def generate_PyG_data(feature, label):
    """
    :param feature: Data with keys uv. shape(101x101,2)
    :param label: Lable data with keys uv. shape(101x101,2)
    :return: dataset (tensor): Array of torchgeometric Data objects.
    """
    edge_index = knn_graph(feature, k=10, batch=None, loop=False)
    data = Data(x=feature, edge_index=edge_index, y=label)
    return data


# def generate_torchgeometric_dataset(data):
#     """
#     Generate dataset that can be used to train with PyG Graph.
#     """
#     """
#     Args:
#         data (dict): Data  with keys t, uv,x,y.
#     Returns:
#         dataset (list): Array of torchgeometric Data objects.
#     """
#
#     n_sims = data['uv'].shape[0]
#     dataset = []
#
#     for sim_ind in range(n_sims):
#         print("{} / {}".format(sim_ind + 1, n_sims))
#
#         x = data['x'][sim_ind]
#         tri = Delaunay(x)
#         neighbors = neighbors_from_delaunay(tri)
#
#         # Find periodic couples and merge their neighborhoods
#         origin_node = 0
#         corner_nodes = []
#         hor_couples = []
#         vert_couples = []
#         eps = 1.0e-6
#
#         b = x.ravel().max()  # domain size
#
#         for i in range(x.shape[0]):
#             if is_near(x[i], [[b, 0], [0, b], [b, b]]):
#                 corner_nodes.append(i)
#             elif is_near(x[i], [[0, 0]]):
#                 origin_node = i
#             elif abs(x[i, 0]) < eps:  # left boundary
#                 for j in range(x.shape[0]):
#                     if abs(x[j, 0] - b) < eps and abs(x[j, 1] - x[i, 1]) < eps:
#                         hor_couples.append([i, j])
#             elif abs(x[i, 1]) < eps:  # bottom boundary
#                 for j in range(x.shape[0]):
#                     if abs(x[j, 1] - b) < eps and abs(x[j, 0] - x[i, 0]) < eps:
#                         vert_couples.append([i, j])
#
#         remove_nodes = []
#
#         # Merge corners
#         for i in corner_nodes:
#             neighbors[origin_node].extend(neighbors[i])
#             remove_nodes.append(i)
#
#         # Merge horizontal couples
#         for i, j in hor_couples:
#             neighbors[i].extend(neighbors[j])
#             remove_nodes.append(j)
#
#         # Merge vertical couples
#         for i, j in vert_couples:
#             neighbors[i].extend(neighbors[j])
#             remove_nodes.append(j)
#
#         use_nodes = list(set(range(len(x))) - set(remove_nodes))
#
#         # Remove right and top boundaries
#         neighbors = np.array(neighbors, dtype=np.object)[use_nodes]
#
#         # Rewrite indices of the removed nodes
#         map_domain = corner_nodes + [x[1] for x in hor_couples] + [x[1] for x in vert_couples]
#         map_codomain = [origin_node] * 3 + [x[0] for x in hor_couples] + [x[0] for x in vert_couples]
#         map_inds = dict(zip(map_domain, map_codomain))
#
#         for i in range(len(neighbors)):
#             for j in range(len(neighbors[i])):
#                 if neighbors[i][j] in remove_nodes:
#                     neighbors[i][j] = map_inds[neighbors[i][j]]
#             neighbors[i] = list(set(neighbors[i]))  # remove duplicates
#
#         # Reset indices
#         map_inds = dict(zip(use_nodes, range(len(use_nodes))))
#
#         for i in range(len(neighbors)):
#             for j in range(len(neighbors[i])):
#                 neighbors[i][j] = map_inds[neighbors[i][j]]
#
#         # ...
#         edge_index = []
#         for i, _ in enumerate(neighbors):
#             for _, neighbor in enumerate(neighbors[i]):
#                 if i == neighbor:
#                     continue
#                 edge = [i, neighbor]
#                 edge_index.append(edge)
#         edge_index = np.array(edge_index).T
#
#         # coords_use = data['x'][sim_ind, use_nodes]
#         # coords_rem = data['x'][sim_ind, remove_nodes]
#         # plt.scatter(coords_use[:, 0], coords_use[:, 1], s=3)
#         # plt.scatter(coords_rem[:, 0], coords_rem[:, 1], s=3)
#         # plt.savefig("tmp.png")
#         # print(qwe)
#
#         n = None
#         print(f"generate_torchgeom_dataset() -> using {n} steps.")
#         tg_data = Data(
#             x=torch.Tensor(data['u'][sim_ind, 0, use_nodes, :]),
#             edge_index=torch.Tensor(edge_index).long(),
#             y=torch.Tensor(data['u'][sim_ind][0:n, use_nodes]).transpose(0, 1),
#             pos=torch.Tensor(data['x'][sim_ind, use_nodes]),
#             t=torch.Tensor(data['t'][sim_ind][0:n]),
#         )
#
#         dataset.append(tg_data)
#
#     return dataset


def summary_parameters(model):
    print(model)


def get_boundary_loss(input):
    mse_loss = nn.MSELoss()
    input = input.permute(0, 2, 1).reshape(-1, 2, 101, 101)
    input_lr = torch.sub(input[:, :, :, -1], input[:, :, :, 1])
    input_ul = torch.sub(input[:, :, -1, :], input[:, :, 1, :])
    loss = mse_loss(input_lr, torch.zeros_like(input_lr)) + mse_loss(input_ul, torch.zeros_like(input_ul))
    return loss


def weights_init(m):
    for sub_m in m.modules():
        if isinstance(sub_m, (nn.Conv2d, nn.Linear)):
            # nn.init.orthogonal_(sub_m.weight)
            nn.init.xavier_uniform_(sub_m.weight.data)  # Xavier初始化的基本思想是保持输入和输出的方差一致，这样就避免了所有输出值都趋向于0。这是通用的方法，适用于任何激活函数
            # nn.init.xavier_uniform_(sub_m.weight(), gain=nn.init.calculate_gain('relu')) #使用 gain 参数来自定义初始化的标准差来匹配特定的激活函数
            # nn.init.kaiming_normal_(sub_m.weight.data, mode='fan_in')  # 推荐在ReLU网络中使用
            # nn.init.xavier_uniform_(sub_m.weight)
            # nn.init.normal_(sub_m.weight.data, 0.0, 0.02)
            nn.init.zeros_(sub_m.bias.data)


import os


# import cv2


def csv2npy(dir_path, save_path, save_name):
    npy = []
    filenames = sorted((fn for fn in os.listdir(dir_path) if fn.endswith('.csv') and fn.startswith('diff2d')))
    for filename in filenames:
        print(filename)
        csv = pd.read_csv(dir_path + filename)
        npy.append(csv.values)
    np.save(save_path + save_name + ".npy", npy)


"""
# VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
# 'MJPG’意思是支持jpg格式图片
# fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
# (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
# resize方法是cv2库提供的更改像素大小的方法
# 释放资源：video.release()
def png2vedio():
    # 要被合成的多张图片所在文件夹
    # 路径分隔符最好使用“/”,而不是“\”,“\”本身有转义的意思；或者“\\”也可以。
    # 因为是文件夹，所以最后还要有一个“/”
    file_dir = 'C:/Users/xxx/Desktop/img/'
    list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            list.append(file)  # 获取目录下文件名列表

    # VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
    # 'MJPG'意思是支持jpg格式图片
    # fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
    # (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
    # 定义保存视频目录名称和压缩格式，像素为1280*720
    video = cv2.VideoWriter('C:/Users/xxx/Desktop/test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 5, (1280, 720))

    for i in range(1, len(list)):
        # 读取图片
        img = cv2.imread('C:/Users/xxx/Desktop/img/' + list[i - 1])
        # resize方法是cv2库提供的更改像素大小的方法
        # 将图片转换为1280*720像素大小
        img = cv2.resize(img, (1280, 720))
        # 写入视频
        video.write(img)

    # 释放资源
    video.release()


def draw_2d_points(points, title=None):
    x = points[:, 0]
    y = points[:, 1]
    triangle = tri.Triangulation(x, y)
    plt.figure()
    plt.triplot(triangle, 'ko-', lw=0.5, markersize=2, alpha=0.6)
    if title:
        plt.title(title)
"""
