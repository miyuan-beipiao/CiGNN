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

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# device_ids = [0, 1, 2, 4] 0,1,2,3,
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

if __name__ == '__main__':
    """ 
        初始化参数 
    """
    dataset = 'gs'
    dimension = 3
    # train_seed_list = np.linspace(1, 50, 50, dtype=int)
    # valid_seed_list = np.linspace(51, 55, 5, dtype=int)
    # train_seed_list = np.linspace(7, 36, 30, dtype=int)
    # valid_seed_list = np.linspace(2, 6, 5, dtype=int)

    train_seed_list = np.linspace(2, 6, 5, dtype=int)
    valid_seed_list = np.linspace(51, 52, 2, dtype=int)

    main(dataset, dimension, train_seed_list, valid_seed_list)
