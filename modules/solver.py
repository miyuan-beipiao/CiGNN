#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: solver_1.py
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
from torch_geometric.data.data import Data
from torch_cluster import knn_graph
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, inits, GATConv, GATv2Conv, InstanceNorm, global_add_pool as gap, \
    global_mean_pool as gmp
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter
import torch_sparse
from torch_geometric.utils import softmax, is_torch_sparse_tensor, is_undirected, remove_self_loops, add_self_loops, \
    to_torch_coo_tensor
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size, SparseTensor
import torch.nn.functional as F

from utils.dataUtils import fix_bc_in_solver, Normalizer, add_velocity_noise, fix_obst_in_solver, \
    NodeType, swish, LpLoss, copy_geometric_data


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, args):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.args = args
        self.add_dim = self.args.addition_feature_dim

        self.norm = nn.LayerNorm(self.hidden_dim)
        # self.norm = InstanceNorm(self.hidden_dim)
        # self.activation = nn.LeakyReLU(self.args.leaky_relu_alpha)
        # self.activation = nn.ReLU()
        # self.activation = swish()
        self.activation = nn.GELU()
        # if args.dataset_name == 'cf' or args.dataset_name == 'ns':
        #     self.activation = nn.ReLU()
        # elif args.dataset_name == 'burgers' or args.dataset_name == 'gs' or args.dataset_name == 'bs':
        #     self.activation = nn.GELU()

        node_dim = self.args.input_step * self.input_dim + self.args.pos_feature_dim + 1 + self.add_dim
        edge_dim = 2 * self.args.pos_feature_dim + 2 + 3

        self.node_embedding_mlp = nn.Sequential(
            nn.Linear(node_dim, self.hidden_dim), self.activation,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim), self.norm)

        self.edge_embedding_mlp = nn.Sequential(
            nn.Linear(edge_dim, self.hidden_dim), self.activation,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim), self.norm)

        self.node_normalizer = Normalizer(size=node_dim, name='node_normalizer')
        self.edge_normalizer = Normalizer(size=edge_dim, name='edge_normalizer')

    def encode_node_feature(self, feature, pos_feature, type_feature, v_feature):
        node_feature = torch.cat((feature, pos_feature, type_feature, v_feature), dim=-1)
        # node_feature = torch.cat((feature, pos_feature, type_feature, v_feature), dim=-1)
        node_feature = self.node_normalizer(node_feature, self.training)
        return self.node_embedding_mlp(node_feature)

    def encode_edge_feature(self, pos_feature, edge_index):
        # node_i, node_j = feature[edge_index[0]], feature[edge_index[1]]
        relative_pos_vector = pos_feature[edge_index[1]] - pos_feature[edge_index[0]]
        relative_pos_distance = torch.norm(relative_pos_vector, dim=1).reshape(-1, 1)
        weight = 1 / relative_pos_distance
        weight_2 = 1 / (relative_pos_distance ** 2)
        weight_3 = 1 / (relative_pos_distance ** 3)
        weight_4 = 1 / (relative_pos_distance ** 4)

        if self.args.dimension == 2:
            x, y = relative_pos_vector[:, 0], relative_pos_vector[:, 1]
            alpha_x, alpha_y = x / relative_pos_distance.view(-1), y / relative_pos_distance.view(-1)
            angel = torch.cat((alpha_x.view(-1, 1), alpha_y.view(-1, 1)), dim=-1)
        elif self.args.dimension == 3:
            x, y, z = relative_pos_vector[:, 0], relative_pos_vector[:, 1], relative_pos_vector[:, 2]
            alpha_x, alpha_y, alpha_z = x / relative_pos_distance.view(-1), y / relative_pos_distance.view(
                -1), z / relative_pos_distance.view(-1)
            angel = torch.cat((alpha_x.view(-1, 1), alpha_y.view(-1, 1), alpha_z.view(-1, 1)), dim=-1)

        # edge_feature = torch.cat((node_i, node_j, relative_pos_vector, relative_pos_distance, weight, angel), dim=-1)
        edge_feature = torch.cat(
            (relative_pos_vector, relative_pos_distance, weight, weight_2, weight_3, weight_4, angel), dim=-1)
        edge_feature = self.edge_normalizer(edge_feature, self.training)

        return self.edge_embedding_mlp(edge_feature)

    def forward(self, graph):
        node_, edge_index, pos_, type_, variables = graph.x, graph.edge_index, graph.pos, graph.type, graph.v

        encoded_node = self.encode_node_feature(node_, pos_, type_, variables)
        encoded_edge = self.encode_edge_feature(pos_, edge_index)

        return Data(x=encoded_node, edge_attr=encoded_edge, edge_index=edge_index, pos=pos_, v=variables)


class AttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input, mask=None):
        batch_size, sequence_length, _ = input.shape
        query = self.query(input)  # (batch_size, sequence_length, hidden_dim)
        key = self.key(input)  # (batch_size, sequence_length, hidden_dim)
        value = self.value(input)  # (batch_size, sequence_length, hidden_dim)

        # Calculate attention weights
        attention_weights = torch.matmul(query, key.transpose(1, 2))  # (batch_size, sequence_length, sequence_length)
        attention_weights = attention_weights / (self.hidden_dim ** 0.5)  # Scale attention weights
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)  # Mask padded positions
        attention_weights = torch.softmax(attention_weights, dim=-1)  # Normalize attention weights

        # Calculate attention-weighted output
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, sequence_length, hidden_dim)
        attention_output = self.fc(attention_output)  # (batch_size, sequence_length, hidden_dim)
        return attention_output, attention_weights


class GraphAttDiv_Layer(MessagePassing):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, args, aggr, negative_slope=0.2, bias=True,
                 concat=False, add_self_loops=True, edge_dim=True, fill_value='mean'):
        # target_to_source   source_to_target
        super(GraphAttDiv_Layer, self).__init__(node_dim=0, flow="target_to_source", aggr=aggr)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.args = args
        self.pos_dim = args.pos_feature_dim
        self.add_dim = args.addition_feature_dim
        self.heads = args.gat_head
        self.negative_slope = negative_slope
        self.dropout = args.drop_out
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # self.activation = nn.LeakyReLU(self.args.leaky_relu_alpha)
        # self.activation = nn.ReLU()
        # self.activation = swish()
        self.activation = nn.GELU()
        # if args.dataset_name == 'cf' or args.dataset_name == 'ns':
        #     self.activation = nn.ReLU()
        # elif args.dataset_name == 'burgers' or args.dataset_name == 'gs' or args.dataset_name == 'bs':
        #     self.activation = nn.GELU()

        self.norm = nn.LayerNorm(self.hidden_dim)
        # self.norm = InstanceNorm(self.hidden_dim)
        # self.norm = nn.BatchNorm1d(self.hidden_dim)

        self.drop = nn.Dropout(p=self.dropout)

        # self.linear_src = nn.Linear(self.hidden_dim, self.heads * self.hidden_dim, bias=False)
        # self.att_src = nn.Parameter(torch.Tensor(1, self.heads, self.hidden_dim))
        # #
        # if edge_dim is not None:
        #     self.linear_edge = nn.Linear(self.hidden_dim, self.heads * self.hidden_dim, bias=False)
        #     self.att_edge = nn.Parameter(torch.Tensor(1, self.heads, self.hidden_dim))
        # else:
        #     self.linear_edge = None
        #     self.register_parameter('att_edge', None)
        #
        # if bias and concat:
        #     self.bias = nn.Parameter(torch.Tensor(self.heads * self.hidden_dim))
        # elif bias and not concat:
        #     self.bias = nn.Parameter(torch.Tensor(self.hidden_dim))
        # else:
        #     self.register_parameter('bias', None)

        # self.alpha_net = nn.Sequential(nn.Linear(4 * self.hidden_dim, self.hidden_dim, bias=True),
        #                                nn.LeakyReLU(self.negative_slope))

        # self.flux_net = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation, self.norm)

        # self.weight_net = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim ** 2, bias=True), self.activation)

        # self.weight_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # self.weight_2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.flux_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim), self.norm)

        self.message_net = nn.Sequential(
            # nn.Linear(3 * self.heads * self.hidden_dim + self.args.addition_feature_dim,
            #           self.hidden_dim), self.activation,
            nn.Linear(3 * self.hidden_dim, self.hidden_dim), self.activation,
            # self.drop,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim), self.norm)

        self.update_net = nn.Sequential(
            # nn.Linear(2 * self.hidden_dim + self.args.addition_feature_dim, self.hidden_dim), self.activation,
            nn.Linear(2 * self.hidden_dim, self.hidden_dim), self.activation,
            # self.drop,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim), self.norm)

        # self.alpha = nn.Parameter(torch.tensor(100.))
        # self.belta = nn.Parameter(torch.tensor(0.))
        # self.att_edge_src = nn.Parameter(torch.Tensor(1, self.hidden_dim))
        # self.att_edge_rvs = nn.Parameter(torch.Tensor(1, self.hidden_dim))

        self.att_edge_src = nn.Parameter(torch.ones(1, self.hidden_dim))
        # self.bias_edge_src = nn.Parameter(torch.zeros(1, self.hidden_dim))

        # self.bn_1d = nn.BatchNorm1d(self.hidden_dim)

    #     self.reset_parameters()
    #
    # #
    # def reset_parameters(self):
    #     #     # super().reset_parameters()
    #     #     self.linear_src.reset_parameters()
    #     #     if self.linear_edge is not None:
    #     #         self.linear_edge.reset_parameters()
    #     #     glorot(self.att_src)
    #     glorot(self.att_edge_src)
    #     glorot(self.att_edge_rvs)

    #     zeros(self.bias)

    # def to_Sparse_inverse(self, edge_index, edge):
    #     adj_sparse_t = torch_sparse.tensor.SparseTensor.from_edge_index(edge_index, edge).t()
    #     return adj_sparse_t.to_torch_sparse_coo_tensor()._indices(), adj_sparse_t.to_torch_sparse_coo_tensor()._values()
    def to_Sparse_inverse(self, edge_index, edge):
        return torch_sparse.tensor.SparseTensor.from_edge_index(edge_index, edge).t().coo()
        # return torch_sparse.tensor.SparseTensor.from_edge_index(edge_index, edge).t().to_torch_sparse_coo_tensor()._values()

        # 在处理大规模张量时，对于转置和置换（permutation）操作，通常 permute 的运算速度会更快。但sparse_coo_tensor不存在permute方法
        # return to_torch_coo_tensor(edge_index, edge).transpose(0, 1)._values()

    def forward(self, graph):
        # graph_last = graph.clone()
        graph_last = copy_geometric_data(graph)
        # B, N, E = self.args.batch_size, graph.num_nodes, graph.num_edges
        # H, C = self.heads, self.hidden_dim
        # graph.x = self.linear_src(graph.x)

        # if self.add_self_loops:
        #     edge_index, edge_attr = remove_self_loops(graph.edge_index, graph.edge_attr)
        #     graph.edge_index, graph.edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value,
        #                                                        num_nodes=graph.num_nodes)

        # attention, edge_attr = self.edge_updater(graph.edge_index, node=graph.x, edge=graph.edge_attr)
        # _, edge_attr_inverse = self.to_Sparse_inverse(graph.edge_index, edge_attr)
        # node_attr = torch.cat((graph.x, graph.pos), dim=-1)
        edge_attr = self.edge_updater(edge_index=graph.edge_index, node=graph.x, edge=graph.edge_attr)
        _, _, edge_attr_inverse = self.to_Sparse_inverse(graph.edge_index, edge_attr)
        # edge_attr_inverse = self.to_Sparse_inverse(graph.edge_index, edge_attr)
        # To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
        # flux = self.flux_net(torch.cat((edge_attr, edge_attr_inverse), dim=-1))
        # flux = (edge_attr - edge_attr_inverse) / 2
        # flux = edge_attr
        # flux = (edge_attr - edge_attr_inverse) + self.flux_net(torch.cat((edge_attr, edge_attr_inverse), dim=-1))

        # symmetry = edge_attr - self.att_edge_src * edge_attr_inverse  # epoch:[900/1000]  0.078302285655505955
        # # inequality = edge_attr + edge_attr_inverse
        # flux = symmetry # epoch:[900/1000]  0.078302285655505955

        # best in current situation
        symmetry = edge_attr - edge_attr_inverse  # epoch:[900/1000]  0.06407302285655505955
        inequality = self.flux_net(edge_attr + edge_attr_inverse)
        flux = symmetry + inequality  # epoch:[900/1000]  0.06407302285655505955

        # inequality = edge_attr + edge_attr_inverse
        # inequality = torch.abs(edge_attr + edge_attr_inverse) # epoch:[72/1000]  0.24639285655505955

        # all_inequality = gmp(inequality, graph_last.batch)
        # all_symmetry = gmp(symmetry, graph_last.batch)

        # print('alpha:', self.alpha)
        # flux = (1 - self.alpha) * symmetry + self.alpha * inequality
        # flux = self.alpha * symmetry + self.belta * inequality
        # flux = self.weight_1(symmetry) + self.weight_2(inequality)

        # flux = inequality
        # flux = edge_attr - edge_attr_inverse
        # flux = self.att_edge_src * edge_attr - self.att_edge_rvs * edge_attr_inverse
        # flux = self.activation(edge_attr - edge_attr_inverse)
        # flux = self.flux_net(edge_attr - edge_attr_inverse)
        # flux = self.norm(edge_attr - edge_attr_inverse)
        # flux = self.att_edge * (edge_attr + edge_attr_inverse)
        # flux = filter(edge_attr - edge_attr_inverse)
        # flux = (edge_attr - edge_attr_inverse) / torch.sqrt(torch.tensor(self.hidden_dim))
        # flux = (edge_attr + edge_attr_inverse) / 2

        node = self.propagate(edge_index=graph.edge_index, node=graph.x, edge=flux)
        node = node + graph_last.x
        # _, edge_attr = remove_self_loops(graph.edge_index, edge_attr)
        # edge_attr = edge_attr + graph_last.edge_attr
        edge_attr = flux + graph_last.edge_attr
        # edge_attr = symmetry + graph_last.edge_attr
        # return Data(x=node, edge_attr=edge_attr, edge_index=graph_last.edge_index, pos=graph_last.pos,
        #             v=graph_last.v), message, all_symmetry

        return Data(x=node, edge_attr=edge_attr, edge_index=graph_last.edge_index, pos=graph_last.pos,
                    v=graph_last.v)

    def edge_update(self, node_i, node_j, edge):
        # H, C = self.heads, self.hidden_dim
        # if self.edge_dim is not None:
        #     edge = self.linear_edge(edge)
        #     att_edge = (edge.view(-1, H, C) * self.att_edge).sum(dim=-1)
        # # alpha = self.alpha_net(torch.cat((node_i, node_j), dim=-1)).sum(dim=-1)  # [-1,H,2*C]->[-1,H,C]->[-1,H]
        # # alpha = self.alpha_net(torch.cat((node_i, node_j, edge_ij), dim=-1)).squeeze(-1)  # [-1,H,3*C]->[-1,H,1]->[-1,H]
        #
        # att_node_i = (node_i.view(-1, H, C) * self.att_src).sum(dim=-1)
        # att_node_j = (node_j.view(-1, H, C) * self.att_src).sum(dim=-1)
        # alpha = 1 / torch.norm(pos_i - pos_j, dim=1).reshape(-1, 1)
        # alpha = torch.norm(pos_i - pos_j, dim=1).reshape(-1, 1)

        # alpha = self.alpha_net(torch.cat((node_i, node_j), dim=-1))  # [-1,3*C]->[-1,C]
        # alpha = self.alpha_net(torch.cat((node_i, node_j, edge), dim=-1))  # [-1,3*C]->[-1,C]
        # alpha = self.alpha_net(torch.cat((node_i, node_j, edge), dim=-1))  # [-1,3*C]->[-1,C]
        # sum(dim=-1).squeeze(-1)
        # alpha = F.leaky_relu(torch.cat((node_i, node_j), dim=-1), self.negative_slope)
        # alpha = self.alpha_net(alpha).sum(dim=-1)  # [-1,H,3*C]->[-1,H,1]C
        # alpha = self.alpha_net(torch.cat((node_i, node_j, edge), dim=-1))
        # alpha = (self.att_*alpha).sum(dim=-1)
        # alpha = self.alpha_net(edge)  # [-1,3*C]->[-1,C]
        # alpha = att_node_i + att_node_j + att_edge
        # Masked Attention
        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # alpha = alpha - torch.max(alpha)
        # attention = softmax(alpha, index, ptr, size_i)  # [-1,H,1]
        # attention = F.dropout(attention, p=self.dropout, training=self.training)
        # return attention * self.message_net(torch.cat((node_i, node_j), dim=-1))
        # return torch.mul(attention, self.message_net(torch.cat((node_i, node_j, edge), dim=-1)))
        # relative_pos_distance = torch.norm(pos_j - pos_i, dim=-1)
        # weight = 1 / relative_pos_distance
        # return self.message_net(
        #     torch.cat(((node_i + node_j) / 2, torch.abs(node_i - node_j) / 2, edge), dim=-1))
        # return attention, self.message_net(torch.cat((node_i, node_j, edge, pos_i - pos_j, v_i), dim=-1))
        # return edge * self.message_net(torch.cat((node_i, node_j), dim=-1)) # rmse 0.39
        # return attention
        # return attention * self.message_net(torch.cat((node_i, node_j, edge, v_i), dim=-1))
        # return self.message_net(torch.cat((node_i, node_j, edge, v_i), dim=-1))
        return self.message_net(torch.cat((node_i, node_j, edge), dim=-1))

    def message(self, edge):
        return edge
        # weight_diag = torch.diag_embed(edge).view(-1, self.hidden_dim, self.hidden_dim)
        # weight_diag = torch.diag_embed(self.weight_net(edge)).view(-1, self.hidden_dim, self.hidden_dim)
        # weight_diag = self.weight_net(edge).view(-1, self.hidden_dim, self.hidden_dim)
        # return torch.matmul(flux.unsqueeze(1), weight_diag).squeeze(1)

    # def aggregate(self, inputs, index, ptr, dim_size):
    #     # scatter()在CUDA上运行，会有一定概率导致模型结果不可reproducibility
    #     # return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    #     return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def update(self, message, node):
        # return self.update_net(torch.cat((node, pos), dim=-1)) + message
        # return self.update_net(node) + message
        # return self.update_net(torch.cat((node, message), dim=-1)) + message
        # return self.update_net(torch.cat((node, message, v), dim=-1))
        return self.update_net(torch.cat((node, message), dim=-1))
        # return node + message


class SmoothGraphAttDiv_Layer(MessagePassing):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, args, aggr, negative_slope=0.2, bias=True,
                 concat=False, add_self_loops=True, edge_dim=None):
        # target_to_source   source_to_target
        super(SmoothGraphAttDiv_Layer, self).__init__(node_dim=0, flow="source_to_target", aggr=aggr)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.args = args
        self.pos_dim = args.pos_feature_dim
        self.add_dim = args.addition_feature_dim
        self.heads = args.gat_head
        self.negative_slope = negative_slope
        self.dropout = args.drop_out
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim

    def forward(self, graph):
        # graph_last = graph.clone()
        graph_last = copy_geometric_data(graph)
        edge_attr = self.edge_updater(graph.edge_index, node=graph.x, edge=graph.edge_attr, pos=graph.pos)
        node = self.propagate(graph.edge_index, node=graph.x, edge=edge_attr)
        # node = node + graph_last.x
        # edge = edge_attr + graph_last.edge_attr
        return Data(x=node, edge_attr=edge_attr, edge_index=graph.edge_index, pos=graph_last.pos, v=graph_last.v)

    def edge_update(self, node_i, node_j):
        return (node_i + node_j) / 2

    def message(self, edge):
        return edge

    def aggregate(self, inputs, index, ptr, dim_size):
        # scatter()在CUDA上运行，会有一定概率导致模型结果不可reproducibility
        # return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def update(self, message):
        return message


class Processor(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, args):
        super(Processor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hidden_layer = args.hidden_layer
        self.v_layer = args.v_layer
        self.args = args

        self.mpnn_layers = nn.ModuleList()
        for _ in range(self.hidden_layer):
            self.mpnn_layers.append(
                # GATConv(self.hidden_dim, self.hidden_dim, heads=8, concat=False),
                GraphAttDiv_Layer(input_dim=self.input_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                                  args=self.args, aggr='mean')
            )
        # self.smooth_mpnn_layers = nn.ModuleList()
        # for _ in range(self.hidden_layer):
        #     self.smooth_mpnn_layers.append(
        #         # GATConv(self.hidden_dim, self.hidden_dim, heads=8, concat=False),
        #         SmoothGraphAttDiv_Layer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, out_dim=self.hidden_dim,
        #                                 args=self.args, aggr='mean')
        #     )
        #
        # self.filter_net = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.ReLU())
        # self.mpnn_layer = GraphAttDiv_Layer(input_dim=self.input_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
        #                           args=self.args, aggr='mean')

        # self.weight_net = nn.Sequential(
        #     nn.Linear((self.hidden_layer + 1) * self.hidden_dim, self.hidden_dim),nn.LayerNorm())
        # self.weight_net = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        # self.weight_net = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # self.weight_net = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
        # self.hidden_dim - self.out_dim + 1

        # self.bn_1d = nn.BatchNorm1d(self.hidden_dim)
        # self.bn_1d = nn.BatchNorm1d((self.hidden_layer + 1))
        # self.bn_2d = nn.BatchNorm2d(self.hidden_dim)

        # self.activation = nn.ReLU()
        self.activation = nn.GELU()
        self.output_cnn = nn.Sequential(
            nn.Conv1d(in_channels=(self.hidden_layer + 1), out_channels=4, kernel_size=1, padding=0, stride=1),
            self.activation,  # nn.RELU(), # nn.RELU(), nn.GELU()
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, padding=0, stride=1))

        # self.temporal_weight_ = nn.Parameter(torch.ones(self.hidden_layer + 1, 1, 1))
        # self.smx = nn.Softmax(dim=0)

    # temporal gs
    # def forward(self, graph):
    #     # init_x = graph.x.detach()
    #     init_x = graph.x
    #     init_x_list = []
    #     for i in range(self.hidden_layer - 1):
    #         graph = self.mpnn_layers[i](graph)
    #         # graph = self.mpnn_layer(graph)
    #         # graph = self.smooth_mpnn_layers[i](graph)
    #         init_x_list.append(graph.x)
    #         graph.x = init_x + graph.x
    #
    #     graph = self.mpnn_layers[self.hidden_layer - 1](graph)
    #     init_x_list.append(graph.x)
    #
    #     weight = self.smx(self.temporal_weight_)
    #     increment = weight * torch.stack(init_x_list)
    #     # weight = torch.softmax(increment.sum(dim=-1).sum(dim=-1), dim=-1).unsqueeze(-1).unsqueeze(-1)
    #     graph.x = init_x + increment.sum(dim=0)
    #     # graph.x = (weight * increment).sum(dim=0)
    #
    #     # graph.x = self.weight_net(torch.cat(init_x_list, dim=-1))
    #     return graph

    # temporal burgers
    # def forward(self, graph):
    #     # init_x = graph.x.detach()
    #     init_x = graph.x
    #     init_x_list = [init_x]
    #     for i in range(self.hidden_layer):
    #         graph = self.mpnn_layers[i](graph)
    #         init_x_list.append(graph.x)
    #         graph.x = init_x + graph.x
    #     # graph.x = torch.stack(init_x_list,dim=0).sum(dim=0).squeeze(dim=0)
    #     graph.x = self.weight_net(torch.stack(init_x_list, dim=0).sum(dim=0).squeeze(dim=0))
    #     # graph.x = self.weight_net(torch.cat(init_x_list, dim=-1))
    #     # graph.x = torch.stack(init_x_list, dim=0).sum(dim=0).squeeze(dim=0)
    #     return graph

    # cf bs
    # def forward(self, graph):
    #     init_x = graph.x
    #     for i in range(self.hidden_layer):
    #         graph = self.mpnn_layers[i](graph)
    #         graph.x = init_x + graph.x
    #     return graph

    # cf bs
    def forward(self, graph):
        if self.args.dataset_name == 'bs' or self.args.dataset_name == 'gs':
            for i in range(self.hidden_layer):
                graph = self.mpnn_layers[i](graph)
            return graph
        else:
            init_x = graph.x
            init_x_list = [init_x]
            # a_list_ = []
            # b_list_ = []
            for i in range(self.hidden_layer):
                # graph, all_inequality, all_symmetry = self.mpnn_layers[i](graph)
                graph = self.mpnn_layers[i](graph)
                init_x_list.append(graph.x)
                # a_list_.append(all_inequality)
                # b_list_.append(all_symmetry)
            # a = torch.sum(torch.stack(a_list_), dim=0)
            # b = torch.sum(torch.stack(b_list_), dim=0)
            # inter_x = torch.sum(torch.stack(init_x_list, dim=0), dim=0)
            inter_x = torch.stack(init_x_list, dim=0).permute(1, 0, 2)
            # graph.x = self.weight_net(inter_x)
            # weight = self.smx(self.temporal_weight_)
            # graph.x = torch.sum(self.temporal_weight_ * inter_x, dim=0)

            # Decoder CNN, maps to different outputs (temporal bundling)
            # self.output_mlp = nn.Sequential(
            #     nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2, stride=1), nn.ReLU(),
            #     nn.Conv1d(in_channels=8, out_channels=1, kernel_size=5, padding=2, stride=1))

            # Decoder (formula 10 in the paper)
            # dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
            # dt = torch.cumsum(dt, dim=1)
            # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
            # diff = self.output_mlp(graph.x[:, None]).squeeze(1)
            # out = u[:, (-1) * self.args.feature_dim:].repeat(self.time_window, 1) + diff
            # graph.x = self.bn_1d(self.output_cnn(inter_x).squeeze(1))
            # graph.x = self.output_cnn(self.bn_1d(inter_x)).squeeze(1)
            graph.x = self.output_cnn(inter_x).squeeze(1)

            # i_x = graph.x.reshape(-1, self.args.width + 2 * self.args.pad_len,
            #                       self.args.height + 2 * self.args.pad_len, self.hidden_dim)
            # i_x = self.bn_2d(i_x.permute(0, 3, 1, 2))
            # graph.x = i_x.permute(0, 2, 3, 1).reshape(-1, self.hidden_dim)

            # return graph, a, b
            return graph


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, args):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.args = args

        self.norm = nn.LayerNorm(self.out_dim)
        # self.norm = InstanceNorm(self.hidden_dim)
        # self.activation = nn.LeakyReLU(self.args.leaky_relu_alpha)
        # self.activation = nn.ReLU()
        # self.activation = swish()
        self.activation = nn.GELU()
        # if args.dataset_name == 'cf' or args.dataset_name == 'ns':
        #     self.activation = nn.ReLU()
        # elif args.dataset_name == 'burgers' or args.dataset_name == 'gs' or args.dataset_name == 'bs':
        #     self.activation = nn.GELU()

        self.decode_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
                                        # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
                                        # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
                                        nn.Linear(self.hidden_dim, self.out_dim))

        # self.edge_decode_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
        #                                      # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
        #                                      # nn.Linear(self.hidden_dim, self.hidden_dim), self.activation,
        #                                      nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, graph):
        return self.decode_mlp(graph.x)

    # def edge_forward(self, all_inequality, all_symmetry):
    #     return self.edge_decode_mlp(all_symmetry), self.edge_decode_mlp(all_inequality)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        self.encoder = Encoder(self.args.feature_dim, self.args.hidden_dim, self.args.num_classes, self.args)
        self.processor = Processor(self.args.feature_dim, self.args.hidden_dim, self.args.num_classes, self.args)
        self.decoder = Decoder(self.args.feature_dim, self.args.hidden_dim, self.args.num_classes, self.args)

        self.output_normalizer = Normalizer(size=self.args.num_classes, name='output_normalizer')

    def forward(self, graph):
        # generalization to different domain size
        # self.args.mesh_node = 16384
        # self.args.width = 128
        # self.args.height = 128

        # self.args.mesh_node = 8000
        # self.args.width = 20
        # self.args.height = 20
        # self.args.length = 20

        # self.args.mesh_node = 1024
        # self.args.width = 32
        # self.args.height = 32

        # train
        if self.training:
            graph = add_velocity_noise(graph, self.args, noise_std=1e-4)
            # return self.train_forward_no_grad(graph)
            return self.train_forward(graph)
        # test
        else:
            return self.test_forward(graph)

    def train_forward(self, graph):
        pred_list, label_list, reconstruct_list, perceptual_list = [], [], [], []
        # torch.logical_or 只能拼2个
        mask = (graph.type[:, 0] == NodeType.NORMAL) | (graph.type[:, 0] == NodeType.PERIODIC_BOUNDARY) | (
                graph.type[:, 0] == NodeType.OUTFLOW)
        # mask = (graph.type[:, 0] == NodeType.NORMAL) | (graph.type[:, 0] == NodeType.PERIODIC_BOUNDARY)
        # mask = torch.logical_or(graph.type[:, 0] == NodeType.NORMAL, graph.type[:, 0] == NodeType.OUTFLOW)
        # mask = (graph.type[:, 0] == NodeType.NORMAL)

        for i in range(self.args.roll_step):
            # type.reshape(-1, self.args.input_step, 1)[:, :self.args.input_step, 0]
            output = graph.y[:, (self.args.feature_dim * i):(self.args.feature_dim * (i + 1))]

            if self.args.predict_type == 'increment':
                norm_label = output[:, :self.args.num_classes] - graph.x[:, -self.args.num_classes:]
                # norm_label = norm_label / (self.args.dt * self.args.time_step)
                label = self.output_normalizer(norm_label, accumulate=self.training)
            elif self.args.predict_type == 'state':
                norm_label = output[:, :self.args.num_classes]
                label = self.output_normalizer(norm_label, accumulate=self.training)

            # perceptual_target = self.encoder.encode_node_feature(label, graph.pos, graph.type, graph.v)

            # pred, reconstruct, perceptual_input = self.forward_once(graph)
            # pred, zero, incre = self.forward_once(graph)
            pred = self.forward_once(graph)
            # zero_list.append(zero)
            # pred = self.forward_once(graph)
            # perceptual_list.append(perceptual_target - perceptual_input)
            pred = fix_bc_in_solver(self.args, pred)
            pred_list.append(pred[mask])
            # reconstruct = fix_bc_in_solver(self.args, reconstruct)
            # reconstruct = self.encoder.node_normalizer.inverse(
            #     torch.cat((reconstruct, graph.pos, graph.type, graph.v), dim=-1))[..., :self.args.feature_dim]
            # reconstruct_list.append(reconstruct)

            if self.args.predict_type == 'increment':
                input = self.output_normalizer.inverse(pred) + graph.x[:, -self.args.num_classes:]
            elif self.args.predict_type == 'state':
                input = self.output_normalizer.inverse(pred)

            # input = torch.cat((graph.x[:, self.args.num_classes:], input), dim=-1)
            # .detach()
            graph.x = input
            label_list.append(label[mask])

            # incre_label = torch.mean(label[mask], dim=0)
            # incre_list.append(incre[mask])

        # return torch.stack(pred_list, dim=-1), torch.stack(label_list, dim=-1), torch.stack(reconstruct_list,
        #                                                                                     dim=-1), torch.stack(
        #     perceptual_list, dim=-1)
        return torch.stack(pred_list, dim=-1), torch.stack(label_list, dim=-1)
        # return torch.stack(pred_list, dim=-1), torch.stack(label_list, dim=-1), torch.stack(zero_list,
        #                                                                                     dim=-1), torch.stack(
        #     incre_list, dim=-1)

    def train_forward_no_grad(self, graph):
        # type.reshape(-1, self.args.input_step, 1)[:, :self.args.input_step, 0]
        # torch.logical_or 只能拼2个
        mask = (graph.type[:, 0] == NodeType.NORMAL) | (graph.type[:, 0] == NodeType.PERIODIC_BOUNDARY) | (
                graph.type[:, 0] == NodeType.OUTFLOW)
        with torch.no_grad():
            for i in range(self.args.roll_step - 1):
                # type.reshape(-1, self.args.input_step, 1)[:, :self.args.input_step, 0]
                output = graph.y[:, (self.args.feature_dim * i):(self.args.feature_dim * (i + 1))]

                pred = self.forward_once(graph)
                pred = fix_bc_in_solver(self.args, pred)

                if self.args.predict_type == 'increment':
                    norm_label = output[:, :self.args.num_classes] - graph.x[:, -self.args.num_classes:]
                    label = self.output_normalizer(norm_label, accumulate=self.training)
                    input = self.output_normalizer.inverse(pred) + graph.x[:, -self.args.num_classes:]
                elif self.args.predict_type == 'state':
                    label = self.output_normalizer(output[:, :self.args.num_classes], accumulate=self.training)
                    input = self.output_normalizer.inverse(pred)

                input = torch.cat((graph.x[:, self.args.num_classes:], input), dim=-1)
                # .detach()
                graph.x = input
                # pred[torch.logical_not(mask)] = output[torch.logical_not(mask)]

        output = graph.y[:, -self.args.feature_dim:]

        pred = self.forward_once(graph)
        # pred = fix_bc_in_solver(self.args, pred)

        if self.args.predict_type == 'increment':
            if self.args.roll_step == 1:
                norm_label = output[:, :self.args.num_classes] - graph.x[:, -self.args.num_classes:]
            else:
                last_output = graph.y[:, -2 * self.args.feature_dim:-self.args.feature_dim]
                norm_label = output[:, :self.args.num_classes] - last_output[:, :self.args.num_classes]

            label = self.output_normalizer(norm_label, accumulate=self.training)
        elif self.args.predict_type == 'state':
            label = self.output_normalizer(output[:, :self.args.num_classes], accumulate=self.training)

        return pred[mask], label[mask]

    def test_forward(self, graph):
        pred_list, label_list = [], []
        # torch.logical_or 只能拼2个
        # mask = (graph.type[:, 0] == NodeType.NORMAL) | (graph.type[:, 0] == NodeType.PERIODIC_BOUNDARY) | (
        #             graph.type[:, 0] == NodeType.OUTFLOW)
        # mask = (graph.type[:, 0] == NodeType.NORMAL) | (graph.type[:, 0] == NodeType.PERIODIC_BOUNDARY)
        mask = torch.logical_or(graph.type[:, 0] == NodeType.NORMAL, graph.type[:, 0] == NodeType.OUTFLOW)
        # mask = (graph.type[:, 0] == NodeType.NORMAL)

        n_roll = 1
        if self.args.dataset_name == 'bs':
            n_roll = self.args.roll_step
        for i in range(n_roll):
            # type.reshape(-1, self.args.input_step, 1)[:, :self.args.input_step, 0]
            output = graph.y[:, (self.args.feature_dim * i):(self.args.feature_dim * (i + 1))]

            # pred, _, _ = self.forward_once(graph)
            pred = self.forward_once(graph)
            pred = fix_bc_in_solver(self.args, pred)

            if self.args.predict_type == 'increment':
                # pred = pred * (self.args.dt * self.args.time_step)
                pred = self.output_normalizer.inverse(pred) + graph.x[:, -self.args.num_classes:]
            elif self.args.predict_type == 'state':
                pred = self.output_normalizer.inverse(pred)

            # pred = torch.cat((graph.x[:, self.args.num_classes:], pred), dim=-1)
            pred[torch.logical_not(mask)] = output[torch.logical_not(mask)]
            # pred[mask] = output[mask]

            pred_list.append(pred)
            graph.x = pred
            label_list.append(output)

        if n_roll == 1:
            return pred, output
        else:
            return torch.stack(pred_list, dim=0), torch.stack(label_list, dim=0)

    def forward_once(self, graph):
        return self.forward_euler(graph)

    def forward_euler(self, graph):
        # dynamic loss
        graph = self.encoder(graph)
        # edge_index = to_undirected(edge_index)
        # graph.edge_index, graph.edge_attr = add_self_loops(graph.edge_index, graph.edge_attr)
        # reconstruct loss
        # restrcut = self.decoder(graph)
        # graph, all_inequality, all_symmetry = self.processor(graph)
        graph = self.processor(graph)
        # perceptual loss
        # perceptual_input = graph.x
        pred = self.decoder(graph)
        # zero, incre = self.edgedecoder(all_inequality, all_symmetry)
        # zero, incre = self.decoder.edge_forward(all_inequality, all_symmetry)
        # return pred, restrcut, perceptual_input
        # return pred, zero, incre
        return pred
