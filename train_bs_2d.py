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
# os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'
import math
import torch
import numpy as np

sys.path.append("..")
# device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

torch.set_printoptions(precision=8)

from scripts.train_script import main

if __name__ == '__main__':
    """ 
        初始化参数 
    """
    dataset = 'bs'
    dimension = 2
    # train_seed_list = np.linspace(1994, 1994, 1, dtype=int)
    # valid_seed_list = np.linspace(1994, 1994, 1, dtype=int)
    train_seed_list = np.linspace(1994, 2018, 25, dtype=int)
    valid_seed_list = np.linspace(2019, 2020, 2, dtype=int)
    # train_seed_list = np.linspace(1993, 2021, 29, dtype=int)
    main(dataset, dimension, train_seed_list, valid_seed_list)
