#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: test.py 
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
import os
import sys
import time

# [2,3,4,5,6,7]
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6,7,8'
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# device_ids = [0]
# device_ids = [0,1,2,3,4,5,6,7]

import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
# from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
# from torch.Utils.tensorboard import SummaryWriter
# from torch_cluster import knn_graph, radius_graph
from torch_geometric.nn import knn_graph, radius_graph, fps
from tqdm import tqdm
from torch.utils.data.dataset import ConcatDataset
from torch_geometric.loader import DataLoader

sys.path.append("..")
# device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")
from utils import dataUtils, imageUtils

torch.set_printoptions(precision=8)

from scripts.test_scipt import main, gen_test_dataset


def plot_full_dataset(dataset, dimension, seed_list):
    args = dataUtils.get_config(dataset, dimension)
    dataUtils.check_path(args)
    dataUtils.set_random_seed(args.seed)
    for i, test_item in enumerate(seed_list):
        test_loader, test_u_x, test_pos = gen_test_dataset(test_item, args)
        truth_data = torch.unsqueeze(test_u_x[args.input_step:, ...], dim=1)
        test_pos = torch.unsqueeze(test_pos[args.input_step:, ...], dim=1)

        # [t,1, 841, 2]
        truth_data = dataUtils.minus_bc_in_loss(args, truth_data)
        test_pos = dataUtils.minus_bc_in_pos(args, test_pos)
        test_pos = torch.squeeze(test_pos, dim=1)

        imageUtils.plot_customed_data(test_pos.cpu(), truth_data.cpu(), truth_data.cpu(), args, test_item)


def plot_one_piece(dataset, dimension, seed_list):
    args = dataUtils.get_config(dataset, dimension)
    dataUtils.check_path(args)
    dataUtils.set_random_seed(args.seed)
    for i, test_item in enumerate(seed_list):
        test_loader, test_u_x, test_pos = gen_test_dataset(test_item, args)
        truth_data = torch.unsqueeze(test_u_x[args.input_step:, ...], dim=1)
        test_pos = torch.unsqueeze(test_pos[args.input_step:, ...], dim=1)

        # [t, 1, 841, 2]
        truth_data = dataUtils.minus_bc_in_loss(args, truth_data)
        test_pos = dataUtils.minus_bc_in_pos(args, test_pos)
        test_pos = torch.squeeze(test_pos, dim=1)

        one_truth_data = truth_data[0:1]

        imageUtils.plot_customed_data(test_pos.cpu(), one_truth_data.cpu(), one_truth_data.cpu(), args, test_item)


if __name__ == '__main__':
    """ 
        初始化参数 
    """
    dataset = 'cf'
    dimension = 2
    # seed_list = np.linspace(220, 224, 5, dtype=int)
    seed_list = [0, 8, 155, 224]  # 0, 8, 20, 155, 221, 224
    # seed_list = np.linspace(222, 222, 1, dtype=int)
    # 212, 214, 217, 219
    # seed_list = np.linspace(8, 8, 1, dtype=int)

    main(dataset, dimension, seed_list)
    # plot_full_dataset(dataset, dimension, seed_list)

    """ 
            绘制训练集数据曲线 
        """
    # print('绘制 train curve')
    # args = dataUtils.get_config(dataset, dimension)
    # train_loss_list = torch.load(args.train_loss_path)
    # imageUtils.plot_loss_curve(os.path.join(args.fig_save_path, "train_loss_curve.jpg"), train_loss_list, 0)

    """ 
        绘制验证集数据曲线 
    """
    # print('绘制 valid curve')
    # valid_loss_list = torch.load(args.valid_loss_path)
    # imageUtils.plot_loss_curve(os.path.join(args.fig_save_path, "valid_loss_curve.jpg"), valid_loss_list, 2)
