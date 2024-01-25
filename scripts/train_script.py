#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: train_script.py 
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

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8'
# os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7,8'
# device_ids = [0, 1, 2, 4]
# device_ids = [0, 1, 3, 4]
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data.dataset import ConcatDataset
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from tqdm import tqdm, trange

sys.path.append("..")
# device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")
from modules.solver import Net
from utils import dataUtils, imageUtils

# import torch.distributed as distributed
# localrank = int(os.environ.get('LOCAL_RANK'))
# # print(localrank)
# device = torch.device('cuda', localrank)
# # device = torch.device('cuda',0)
# torch.cuda.set_device(device)
# distributed.init_process_group(backend='gloo')

torch.set_printoptions(precision=8)
torch.set_num_threads(5)

import logging
from datetime import datetime
import logging.config


# current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# current_file = os.path.basename(__file__)

def train(net, loader, optimizer, criterion, epoch, train_loss_list, args):
    """
        训练模型
    """
    loss_list = []
    # with torch.autograd.set_detect_anomaly(True):

    for i, batch_data in enumerate(loader):
        # source = [data.x for data in batch_data]
        # batch_source = torch.cat(source).cuda()
        #
        # target = [data.y for data in batch_data]
        # batch_target = torch.cat(target).cuda()
        #
        # batch_input = torch.cat((batch_source, batch_target), dim=-1)[..., :args.roll_step * args.feature_dim]

        # batch_pred, batch_label, batch_reconstruct, batch_perceptual = net(batch_data)
        batch_pred, batch_label = net(batch_data)
        # batch_pred, batch_label, zero, incre = net(batch_data)
        # inc_a = torch.mean(batch_pred, dim=0)
        # inc_b = torch.mean(batch_label, dim=0)
        dynamic_loss = torch.sqrt(criterion(batch_pred, batch_label))
        # zero_loss = torch.sqrt(criterion(zero, torch.zeros_like(zero)))
        # incre_loss = torch.sqrt(criterion(incre, batch_label))
        # reconstruct_loss = torch.sqrt(
        #     criterion(batch_input, batch_reconstruct.reshape(-1, args.roll_step * args.feature_dim)))
        # #
        # # perceptual_loss 高维特征相似
        # perceptual_loss = torch.sqrt(criterion(batch_perceptual, torch.zeros_like(batch_perceptual)))

        loss = dynamic_loss
        # global_logger.info(f'当前训练 {i+1} dynamic loss: {dynamic_loss}, zero loss: {zero_loss}, inequality loss: {incre_loss}')
        # loss = dynamic_loss + zero_loss
        # loss = dynamic_loss + zero_loss + incre_loss
        # loss = zero_loss + incre_loss
        # loss = 10 * dynamic_loss + reconstruct_loss + perceptual_loss

        # loss = loss.mean()
        loss_list.append(loss.item())

        optimizer.zero_grad()

        # grads = []
        # for param in net.parameters():
        #     grads.append(param.register_hook(lambda grad: grads.append(grad)))

        loss.backward()

        # 输出梯度大小
        # for grad in grads:
        # print(grad.abs().mean())
        # print(grad.mean())

        # 打印梯度范数
        # for name, param in net.named_parameters():
        #     if 'weight' in name:
        #         grad_norm = param.grad.norm()
        #         if grad_norm >= 100 or grad_norm <= 0.01:
        #             print(name, grad_norm)

        # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        optimizer.step()

        # if (i + 1) % 1 == 0:
        #     print('epoch:[{}/{}]'.format(epoch + 1, args.num_epochs), ',step:[{}/{}]'.format(i + 1, len(loader)),
        #           ',train loss:', dynamic_loss.item(), ',reconstruct loss:', reconstruct_loss.item(),
        #           ',perceptual loss:', perceptual_loss.item())

    avg_loss = sum(loss_list) / len(loss_list)
    train_loss_list.append(avg_loss)

    return avg_loss


def valid(net, loader, criterion, epoch, args):
    """ 测试模型 """
    # test_out = []
    loss_list = []
    batch_pred = None

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            """外显处理mask"""
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

            # loss = torch.sqrt(criterion(batch_pred[..., -args.num_classes:], batch_label))
            loss = torch.sqrt(criterion(batch_pred, batch_label))

            loss_list.append(loss.item())

            # if (i + 1) % 1 == 0:
            #     print('epoch:[{}/{}]'.format(epoch + 1, args.num_epochs), ',step:[{}/{}]'.format(i + 1, len(loader)),
            #           ',valid loss:', loss.item())

    avg_loss = sum(loss_list) / len(loss_list)

    return avg_loss


def gen_train_dataset(item, args):
    u_x, u_y, pos, edge_index, type, v = dataUtils.process_ux(args, item)

    train_valid_idx = int(u_x.shape[0] * args.split_train_valid)

    train_u_x = u_x[:train_valid_idx]
    train_u_y = u_y[:train_valid_idx]
    train_pos = pos[:train_valid_idx]
    train_edge_index = edge_index[:train_valid_idx]
    train_type = type[:train_valid_idx]
    train_v = v[:train_valid_idx]

    TGD = dataUtils.TimeseriesGraphDataset(train_u_x, train_u_y, input_len=args.input_step,
                                           roll_len=args.roll_step, pos=train_pos, edge_index=train_edge_index,
                                           type=train_type, v=train_v, args=args)

    return TGD.generate_torchgeometric_dataset()


def gen_valid_dataset(item, args):
    u_x, u_y, pos, edge_index, type, v = dataUtils.process_ux(args, item)

    train_valid_idx = int(u_x.shape[0] * args.split_train_valid)
    valid_u_x = u_x[:train_valid_idx]
    valid_u_y = u_y[:train_valid_idx]
    valid_pos = pos[:train_valid_idx]
    valid_edge_index = edge_index[:train_valid_idx]
    valid_type = type[:train_valid_idx]
    valid_v = v[:train_valid_idx]

    # if args.split_train_valid == 1:
    #     train_valid_idx = int(u_x.shape[0] * args.split_train_valid)
    #     valid_u_x = u_x[:train_valid_idx]
    #     valid_u_y = u_y[:train_valid_idx]
    #     valid_pos = pos[:train_valid_idx]
    #     valid_edge_index = edge_index[:train_valid_idx]
    #     valid_type = type[:train_valid_idx]
    #     valid_time = all_time[:train_valid_idx]
    # else:
    #     train_valid_idx = int(u_x.shape[0] * args.split_train_valid)
    #     valid_u_x = u_x[train_valid_idx:]
    #     valid_u_y = u_y[train_valid_idx:]
    #     valid_pos = pos[train_valid_idx:]
    #     valid_edge_index = edge_index[train_valid_idx:]
    #     valid_type = type[train_valid_idx:]
    #     valid_time = all_time[train_valid_idx:]

    if args.dataset_name == 'bs':
        TGD = dataUtils.TimeseriesGraphDataset(valid_u_x, valid_u_y, input_len=args.input_step,
                                               roll_len=args.roll_step, pos=valid_pos, edge_index=valid_edge_index,
                                               type=valid_type, v=valid_v, args=args)
    else:
        TGD = dataUtils.TimeseriesGraphDataset(valid_u_x, valid_u_y, input_len=args.input_step,
                                               roll_len=1, pos=valid_pos, edge_index=valid_edge_index,
                                               type=valid_type, v=valid_v, args=args)

    return TGD.generate_torchgeometric_dataset()


# def lr_func(e):
#     return min((e + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(e / args.num_epochs * math.pi) + 1))


# def collate_fn(data_list):
#     # 获取每个图数据的形状
#     shapes = [data.num_nodes for data in data_list]
#
#     # 确定batch中的最大节点数
#     max_nodes = max(shapes)
#
#     # 将每个图数据的节点数填充到最大节点数
#     for data in data_list:
#         num_nodes = data.num_nodes
#         if num_nodes < max_nodes:
#             # x=x, edge_index=edge_idx, y=y, pos=pos, type=type, v=v
#             data.x = torch.cat([data.x, torch.zeros(max_nodes - num_nodes, data.x.size(1))], dim=0)
#             data.y = torch.cat([data.y, torch.zeros(max_nodes - num_nodes, data.y.size(1))], dim=0)
#             data.pos = torch.cat([data.pos, torch.zeros(max_nodes - num_nodes, data.pos.size(1))], dim=0)
#             data.type = torch.cat([data.type, torch.zeros(max_nodes - num_nodes, data.type.size(1))], dim=0)
#             data.v = torch.cat([data.v, torch.zeros(max_nodes - num_nodes, data.v.size(1))], dim=0)
#
#     # batch_tensor = torch.zeros((len(batch),
#     #                             max_len))  # 将样本填充到batch_tensor中
#     # for i, item in enumerate(batch):
#     #     batch_tensor[i, :len(item)] = torch.Tensor(item)
#     # return batch_tensor, lengths
#
#     # 返回填充后的图数据
#     return data_list

def init_weights(module):
    # if isinstance(module, nn.Linear):
    #     nn.init.xavier_uniform_(module.weight)
    #     if module.bias is not None:
    #         nn.init.zeros_(module.bias)
    #
    #     # 对权重进行缩小1000倍
    #     module.weight.data /= 1000.0
    #
    # if type(module) == nn.Linear:
    #     module.weight.fill_(1.0)
    if isinstance(module, nn.Linear):
        # module.weight.data /= (module.in_features * module.out_features)
        module.weight.data /= 10


def run(train_datasets, valid_datasets, args):
    """
        初始化参数并构建模型
    """
    best_valid_loss = args.best_valid_loss
    net = Net(args)
    # net.apply(init_weights)
    # writer = SummaryWriter('runs/graph_example')
    # writer.add_graph(net, input_to_model=None, verbose=False)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate * args.batch_size,
    #                              weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay_steps=4e4, min_lr=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, mode='min', patience=100, min_lr=1e-8)

    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, eta_min=1e-8)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100, min_lr=1e-8)
    # lr_func = lambda e: min((e + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(e / args.num_epochs * math.pi) + 1))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    # torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,steps_per_epoch=len(train_loader), epochs=args.num_epochs)
    criterion = nn.MSELoss()
    # criterion = dataUtils.LpLoss(size_average=True)
    # criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.MSELoss(reduce=False)
    # criterion = nn.KLDivLoss(reduction='batchmean')
    # criterion = nn.KLDivLoss()
    # criterion = nn.CrossEntropyLoss()
    # L1 = nn.L1Loss()

    # 判断是否存在多个GPU
    if torch.cuda.device_count() >= 1 and args.is_dp:
        # print("Let's use", torch.cuda.device_count(), "GPUS")
        # 将模型部署到多个GPU上
        # net = nn.DataParallel(net, device_ids=args.device_ids)
        net = DataParallel(net, device_ids=args.device_ids)
        # net = DataParallel(net, device_ids=[localrank])

    net = net.cuda()

    train_loss_list = []
    valid_loss_list = []

    """ 
        加载模型 
    """
    path = args.pt_save_path + 'every_{}_checkpoint.pt'.format(10)
    if os.path.exists(path):
        global_logger.info(f'读取 model,optimizer, scheduler, best_valid_loss')
        net, optimizer, scheduler, best_valid_loss = dataUtils.load_model(path)
    if os.path.exists(args.train_loss_path):
        global_logger.info(f'读取 train_loss_list')
        train_loss_list = torch.load(args.train_loss_path)
    if os.path.exists(args.valid_loss_path):
        global_logger.info(f'读取 valid_loss_list')
        valid_loss_list = torch.load(args.valid_loss_path)

    used_epoch = len(train_loss_list)
    global_logger.info(f'当前训练 epoch: {len(train_loss_list)}')
    global_logger.info(f'当前训练 best_valid_loss: {best_valid_loss}')
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    global_logger.info(f'当前训练 lr: {current_lr}')
    total_params = sum(p.numel() for p in net.parameters())
    global_logger.info(f'当前训练 total parameters：{total_params}')

    # 遍历各个模块，打印每个模块的名称和参数数量
    for name, param in net.named_parameters():
        print(f"{name}: {param.numel()}")

    """ 
        训练模型 
    """
    # all_start = time.time()
    # with tqdm(total=args.num_epochs) as pbar:
    #     print("\n")
    # pbar.set_description("Processing %s" % c)
    # pbar.update(1)
    global_logger.info(f'开始 train model')
    # global_logger.info(f'torch.cuda.memory_stats(): {torch.cuda.memory_stats()}')

    # DataListLoader DataLoader
    # pin_memory=True num_workers=0
    train_loader = DataListLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True)
    # train_loader = DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True,
    #                               collate_fn=collate_fn)

    # for epoch in range(args.num_epochs):
    for epoch in trange(args.num_epochs - used_epoch, desc="Training", unit="Epoch"):
        # train_start = time.time()
        net.train()
        avg_train_loss = train(net, train_loader, optimizer, criterion, epoch, train_loss_list, args)
        # train_end = time.time()
        # if (epoch + 1) % 10 == 0:
        #     print('The now 10 epoch train time is: ', (train_end - train_start), ',The rest train time is: ',
        #           (train_end - train_start) * (args.num_epochs - epoch - 1))
        # scheduler.step()

        """验证模型"""
        # valid_start = time.time()
        net.eval()
        loss_valid = 0
        for _ in range(len(valid_datasets)):
            valid_loader = DataListLoader(dataset=valid_datasets[_], batch_size=1, shuffle=False)
            # valid_loader = DataLoader(dataset=valid_datasets[_], batch_size=1, shuffle=False)
            # ,num_workers=10,pin_memory=True)
            avg_valid_loss = valid(net, valid_loader, criterion, epoch, args)
            loss_valid += avg_valid_loss
        avg_loss_valid = loss_valid / len(valid_datasets)
        valid_loss_list.append(avg_loss_valid)

        # print('alpha:',net.module.processor.mpnn_layers._modules['0'].alpha,'belta:',net.module.processor.mpnn_layers._modules['0'].belta)

        # valid_end = time.time()
        # if (epoch + 1) % 10 == 0:
        #     print('The one epoch valid time is: ', (valid_end - valid_start), ',The rest valid time is: ',
        #           (valid_end - valid_start) * (args.num_epochs - epoch - 1))

        if avg_loss_valid < best_valid_loss:
            best_valid_loss = avg_loss_valid
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            global_logger.info(f'saving the current best valid model!!! [the current lr]: {current_lr}')
            state = {'model': net, 'optimizer': optimizer, 'scheduler': scheduler, 'best_valid_loss': best_valid_loss,
                     "train_loss_list": train_loss_list, "valid_loss_list": valid_loss_list}
            torch.save(state, args.checkpoint_path)

        scheduler.step(avg_loss_valid)
        # scheduler.step()
        # time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        global_logger.info(
            f'epoch:[{epoch + 1}/{args.num_epochs}] avg train loss: {avg_train_loss},avg valid loss: {avg_loss_valid},the current lr: {current_lr}')
        if (epoch + 1) % 10 == 0:
            state = {'model': net, 'optimizer': optimizer, 'scheduler': scheduler, 'best_valid_loss': best_valid_loss,
                     "train_loss_list": train_loss_list, "valid_loss_list": valid_loss_list}
            # torch.save(state, args.model_save_path + 'checkpoint_epoch{}.pt'.format(epoch + 1))
            torch.save(state, args.pt_save_path + 'every_{}_checkpoint.pt'.format(10))

        torch.save(train_loss_list, args.train_loss_path)
        torch.save(valid_loss_list, args.valid_loss_path)

    # state = {'model': net, 'optimizer': optimizer, 'scheduler': scheduler, 'best_valid_loss': best_valid_loss}
    # torch.save(state, args.pt_save_path + 'final_checkpoint.pt')
    # torch.save(state, args.checkpoint_path)

    # all_end = time.time()
    # print('The all train_valid time is: ', (all_end - all_start))


def main(dataset, dimension, train_seed_list, valid_seed_list):
    args = dataUtils.get_config(dataset, dimension)

    # {current_date}
    log_file_name = f'{args.log_save_path}{dimension}D-{dataset}-meshnode{args.mesh_node}-batchsize{args.batch_size}-' \
                    f'inputstep{args.input_step}-rollstep{args.roll_step}-' \
                    f'learningrate{args.learning_rate}-weightdecay{args.weight_decay}-poollayer{args.pool_layer}-' \
                    f'hiddenlayer{args.hidden_layer}-hiddendim{args.hidden_dim}'

    log_filename = f'{log_file_name}.log'
    logging.config.fileConfig('../logging.conf', defaults={'filename': log_filename})
    # 创建 logger
    global global_logger
    logger = logging.getLogger('sampleLogger')
    global_logger = logger

    # if local_rank == 0:
    #     global_logger.info('Only output from worker 0')
    # else:
    #     global_logger.disabled = True  # 禁用其他工作线程的日志输出

    args_dict = vars(args)
    # 输出参数及其值
    # for arg, value in args_dict.items():
    #     print(f'{arg}: {value}')
    output = ', '.join([f'{arg}: {value}' for arg, value in args_dict.items()])
    global_logger.info(f'args parameters:{output}')

    dataUtils.check_path(args)
    dataUtils.set_random_seed(args.seed)

    train_datasets = []
    valid_datasets = []

    with tqdm(total=train_seed_list.size, desc='读取 Train Dataset', leave=True, ncols=100, unit='Dataset',
              unit_scale=True) as pbar:
        for _, item in enumerate(train_seed_list):
            train_dataset = gen_train_dataset(item, args)
            train_datasets.extend(train_dataset)
            # 更新发呆进度
            pbar.update(1)

    with tqdm(total=valid_seed_list.size, desc='读取 Valid Dataset', leave=True, ncols=100, unit='Dataset',
              unit_scale=True) as pbar:
        for _, item in enumerate(valid_seed_list):
            valid_dataset = gen_valid_dataset(item, args)
            valid_datasets.append(valid_dataset)
            # 更新发呆进度
            pbar.update(1)

    global_logger.info(f'读取 datasets over')

    run(train_datasets, valid_datasets, args)
