#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: cf_args_2d.py 
@time: 2023/02/16
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
import argparse


def get_config_cf(dataset='cf'):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dimension', type=int, default=2, help='2d dimension.')
    parser.add_argument('--space-step', type=int, default=1, help='space-step.')
    parser.add_argument('--time-step', type=int, default=5, help='time-step.')

    parser.add_argument('--addition-feature-dim', type=int, default=4, help='addition input feature dimension.')
    parser.add_argument('--num-knn', type=int, default=8, help='Number of knn nodes.')
    # parser.add_argument('--mode', type=int, default=1, help='1:train; 2:valid; 3:test')
    # parser.add_argument('--multi-scale', type=int, default=3, help='multi scale')
    parser.add_argument('--mask-ratio', type=float, default=0.9, help='mask ratio')
    parser.add_argument('--device_ids', type=list, default=[0], help='device ids')

    parser.add_argument('--predict-type', type=str, default='increment', help='output type: state increment or state')

    parser.add_argument('--num-epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--warmup-epoch', type=int, default=10, help='Warm up number of epochs to train.')
    parser.add_argument('--feature-dim', type=int, default=3, help='input feature dimension.')
    parser.add_argument('--num-classes', type=int, default=3, help='output feature dimension.')
    parser.add_argument('--hidden-dim', type=int, default=128, help='hidden units.')
    parser.add_argument('--hidden-layer', type=int, default=4, help='hidden layers.')
    parser.add_argument('--pool-layer', type=int, default=1, help='pool layers.')
    parser.add_argument('--mesh-layer', type=int, default=0, help='mesh layers.')
    parser.add_argument('--v-layer', type=int, default=0, help='v layers.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay ~ L2 loss.')
    parser.add_argument('--gat-head', type=int, default=1, help='GAT heads')
    parser.add_argument('--input-step', type=int, default=1, help='input time steps.')
    parser.add_argument('--roll-step', type=int, default=10, help='roll time steps.')
    parser.add_argument('--batch-size', type=int, default=10, help='batch size.')
    parser.add_argument('--drop-out', type=int, default=0.5, help='drop out.')
    parser.add_argument('--leaky-relu-alpha', type=int, default=0.01, help='leaky relu alpha.')
    parser.add_argument('--split-train-valid', type=int, default=1, help='split train valid dataset')
    parser.add_argument('--split-train-test', type=int, default=1, help='split train test dataset')
    parser.add_argument('--all-time-steps', type=int, default=1001, help='All time steps.')
    parser.add_argument('--mesh-node', type=int, default=3400, help='mesh nodes.')
    # parser.add_argument('--is-pool', type=bool, default=False, help='whether to pool the model.')
    parser.add_argument('--is-dp', type=bool, default=True, help='whether to DataParallel the model.')
    # parser.add_argument('--is-drop', type=bool, default=False, help='whether to dropout the model.')
    parser.add_argument('--is-euler', type=bool, default=True, help='whether to euler the model.')
    # parser.add_argument('--is-rk4', type=bool, default=False, help='whether to rk4 the model.')
    parser.add_argument('--pad-len', type=int, default=0, help='padding length.')
    parser.add_argument('--seed', type=int, default=44, help='Random seed.')
    parser.add_argument('--best-valid-loss', type=int, default=1, help='best valid loss')
    parser.add_argument('--pos-feature-dim', type=int, default=2, help='pos feature dimension.')
    parser.add_argument('--dataset-name', type=str, default='{}'.format(dataset), help='dataset name.')
    parser.add_argument('--pt-save-path', type=str, default='pt/{}/'.format(dataset), help='pt save path.')
    parser.add_argument('--log-save-path', type=str, default='log/{}/'.format(dataset), help='log save path.')
    parser.add_argument('--fig-save-path', type=str, default='figures/{}/'.format(dataset), help='fig save path.')
    parser.add_argument('--data-save-path', type=str, default='/mnt/miyuan/AI4Physics/Data/{}/'.format(dataset),
                        help='data save path.')
    parser.add_argument('--gif-save-name', type=str, default='{}_graph_phy'.format(dataset), help='gif save name.')
    parser.add_argument('--checkpoint-path', type=str, default='pt/{}/checkpoint.pt'.format(dataset),
                        help='last checkpoint path.')
    parser.add_argument('--train-loss-path', type=str, default='pt/{}/trainloss.pt'.format(dataset),
                        help='train loss path.')
    parser.add_argument('--valid-loss-path', type=str, default='pt/{}/validloss.pt'.format(dataset),
                        help='valid loss path.')
    parser.add_argument('--test-loss-path', type=str, default='pt/{}/testloss.pt'.format(dataset),
                        help='test loss path.')
    parser.add_argument('--test-out-path', type=str, default='pt/{}/testout.pt'.format(dataset),
                        help='test out path.')

    args = parser.parse_args()
    return args
