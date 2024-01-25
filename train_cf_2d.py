#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: train.py 
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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# device_ids = [0, 1, 2, 4]
# device_ids = [0, 1, 3, 4]
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data.dataset import ConcatDataset
from torch_geometric.loader import DataLoader

sys.path.append("..")
# device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

torch.set_printoptions(precision=8)

from scripts.train_script import main

def get_rxy():
    import numpy as np
    from decimal import Decimal

    R = np.arange(0.04, 0.081, 0.005)  # 9
    cylinder_x = np.arange(0.1, 0.31, 0.05)  # 5
    cylinder_y = np.arange(0.1, 0.31, 0.05)  # 5

    list = []
    count = 0
    for r in R:
        for cx in cylinder_x:
            for cy in cylinder_y:
                list.append([r, cx, cy])
                # if count in [0, 8, 20, 155, 222, 224]:
                #     print(Decimal(r), cx, cy)
                count += 1

    # for i in [0, 8, 155, 222, 224]:
    #     print(list[i])

    np.save('rxy.npy', list)


if __name__ == '__main__':
    """ 
        初始化参数 
    """
    dataset = 'cf'
    dimension = 2
    train_seed_list = np.linspace(0, 199, 200, dtype=int)
    valid_seed_list = np.linspace(200, 219, 20, dtype=int)

    # train_seed_list = np.linspace(0, 4, 5, dtype=int)
    # valid_seed_list = np.linspace(6, 9, 5, dtype=int)

    # get_rxy()

    main(dataset, dimension, train_seed_list, valid_seed_list)
