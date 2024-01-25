#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: test_scipt.py 
@time: 2023/02/09
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
# os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'
device_ids = [0]
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
from torch_geometric.loader import DataLoader, DataListLoader

sys.path.append("..")
# device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")
from gdo.modules.solver import Net
from gdo.utils import dataUtils, imageUtils

torch.set_printoptions(precision=8)
torch.set_num_threads(5)


def test(net, loader, criterion, test_loss_list, args):
    """ 测试模型 """
    test_out = []
    loss_list = []

    y_list = []
    pred_list = []

    batch_pred = None

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            # source = [data.x for data in batch_data]
            # batch_source = torch.cat(source).cuda()

            # target = [data.y for data in batch_data]
            # batch_label = torch.cat(target)[:, :args.num_classes].cuda()
            #
            # node_type = [data.type for data in batch_data]
            # node_type = torch.cat(node_type)[:, 0].cuda()
            # # 通过mask方式去除BC
            # mask = (node_type == dataUtils.NodeType.NORMAL)

            if args.dataset_name != 'bs':
                if batch_pred is not None:
                    batch_data[0].x = batch_pred.detach()

            batch_pred, batch_label = net(batch_data)

            y_list.append(batch_label)
            pred_list.append(batch_pred)

            loss = torch.sqrt(criterion(batch_pred, batch_label))

            test_loss_list.append(loss.item())
            loss_list.append(loss.item())

            test_out.append(batch_pred.reshape(-1, batch_pred.shape[-2], batch_pred.shape[-1]))

            if (i + 1) % 10 == 0:
                print('Step:[{}/{}]'.format(i + 1, len(loader)), 'roll loss:', loss.item())

    y_list = torch.stack(y_list, dim=0)
    pred_list = torch.stack(pred_list, dim=0)

    print('total mse loss:', F.mse_loss(y_list, pred_list))

    avg_loss = sum(loss_list) / len(loss_list)
    print('avg test loss:', avg_loss)

    return test_out


# def lr_func(e):
#     return min((e + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(e / args.num_epochs * math.pi) + 1))


def run(test_loader, args):
    test_start = time.time()
    """ 初始化参数并构建模型 """
    net = Net(args)

    """ GPU """
    # 判断是否存在多个GPU
    if torch.cuda.device_count() > 1 and args.is_dp:
        # print("Let's use", torch.cuda.device_count(), "GPUS")
        # 将模型部署到多个GPU上
        net = nn.DataParallel(net, device_ids=device_ids)

    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, mode='min', patience=100,
                                                           min_lr=1e-5)
    criterion = nn.MSELoss()
    # criterion = dataUtils.LpLoss(size_average=True)
    """ 加载模型 """
    path = args.checkpoint_path
    # path = args.pt_save_path + 'every_{}_checkpoint.pt'.format(10)
    if os.path.exists(path):
        print('读取 model, optimizer, scheduler, best_valid_loss')
        net, optimizer, scheduler, best_valid_loss = dataUtils.load_model(path)

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    total_params = sum(p.numel() for p in net.parameters())
    print('参数量为：', f'{total_params:,} total parameters.')
    print('best_valid_loss :', best_valid_loss)

    test_loss_list = []

    """ 预测模型 """
    # net.eval()
    test_out = test(net, test_loader, criterion, test_loss_list, args)

    test_end = time.time()
    print('The all test time is: ', (test_end - test_start))
    return test_out, test_loss_list


def gen_test_dataset(test_item, args):
    test_u_x, test_u_y, test_pos, test_edge_index, test_type, test_v = dataUtils.process_ux(args, test_item)
    train_test_idx = int(test_u_x.shape[0] * args.split_train_test)
    # train_valid_idx = int(test_u_x.shape[0] * args.split_train_valid)
    train_valid_idx = 0

    test_u_x = test_u_x[train_valid_idx:train_test_idx]
    test_u_y = test_u_y[train_valid_idx:train_test_idx]
    test_pos = test_pos[train_valid_idx:train_test_idx]
    test_edge_index = test_edge_index[train_valid_idx:train_test_idx]
    test_type = test_type[train_valid_idx:train_test_idx]
    test_v = test_v[train_valid_idx:train_test_idx]

    if args.dataset_name == 'bs':
        TGD = dataUtils.TimeseriesGraphDataset(test_u_x, test_u_y, input_len=args.input_step,
                                               roll_len=args.roll_step, pos=test_pos, edge_index=test_edge_index,
                                               type=test_type, v=test_v, args=args)
    else:
        TGD = dataUtils.TimeseriesGraphDataset(test_u_x, test_u_y, input_len=args.input_step,
                                               roll_len=1, pos=test_pos, edge_index=test_edge_index,
                                               type=test_type, v=test_v, args=args)

    test_dataset = TGD.generate_torchgeometric_dataset()
    test_loader = DataListLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # , num_workers=10,pin_memory=True)

    return test_loader, test_u_x, test_pos


def main(dataset, dimension, test_seed_list):
    args = dataUtils.get_config(dataset, dimension)

    dataUtils.check_path(args)
    dataUtils.set_random_seed(args.seed)

    test_out_list = []
    test_loss_list = []

    print('开始 test model')
    avg_loss = 0

    for _, test_item in enumerate(test_seed_list):
        print('读取 Test Seed [{}]'.format(test_item))
        test_loader, test_u_x, test_pos = gen_test_dataset(test_item, args)
        test_out, test_loss = run(test_loader, args)

        test_out_list.append(test_out)
        test_loss_list.append(test_loss)

        avg_loss += sum(test_loss) / len(test_loss)

        torch.save(test_loss_list, args.test_loss_path)
        torch.save(test_out_list, args.test_out_path)

    print('avg test loss:', avg_loss/len(test_seed_list))

    """ 绘制结果 """
    test_loss_list = torch.load(args.test_loss_path)
    test_out_list = torch.load(args.test_out_path)

    print('绘制 test image')
    for i, test_item in enumerate(test_seed_list):
        path = os.path.join(args.fig_save_path, "test_loss_curve_{}.jpg".format(test_item))
        imageUtils.plot_loss_curve(path, test_loss_list[i], 1)

        test_loader, test_u_x, test_pos = gen_test_dataset(test_item, args)

        test_out = torch.unsqueeze(torch.cat(test_out_list[i], dim=0), dim=1)
        truth_data = torch.unsqueeze(test_u_x[args.input_step:, ...], dim=1)
        test_pos = torch.unsqueeze(test_pos[args.input_step:, ...], dim=1)

        # test_out = torch.unsqueeze(
        #     torch.cat((test_u_x[:args.input_step, ...].cpu(), torch.cat(test_out_list[i], dim=0).cpu()), dim=0), dim=1)
        # truth_data = torch.unsqueeze(test_u_x[:, ...], dim=1)
        # test_pos = torch.unsqueeze(test_pos[:, ...], dim=1)

        # [t,1, 841, 2]
        test_out = dataUtils.minus_bc_in_loss(args, test_out)
        truth_data = dataUtils.minus_bc_in_loss(args, truth_data)
        test_pos = dataUtils.minus_bc_in_pos(args, test_pos)
        test_pos = torch.squeeze(test_pos, dim=1)

        imageUtils.plot_customed_data(test_pos.cpu(), test_out.cpu(), truth_data.cpu(), args, test_item)

        # imageUtils.plot_customed_data(test_pos.cpu(), truth_data.cpu(), truth_data.cpu(), args, test_item)

        # imageUtils.plot_psnr_score(test_out.cpu(), truth_data.cpu(), args, test_item)
        # imageUtils.plot_pccs_score(test_out.cpu(), truth_data.cpu(), args, test_item)
