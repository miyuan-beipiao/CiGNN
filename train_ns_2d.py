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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '8'
import math
import torch
import numpy as np
import sys

sys.path.append("..")
# device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

torch.set_printoptions(precision=8)

from scripts.train_script import main
import multiprocessing as mp

if __name__ == '__main__':
    """ 
        初始化参数 
    """
    # print(f"num of CPU: {mp.cpu_count()}")

    dataset = 'ns'
    dimension = 2
    train_seed_list = np.linspace(1, 50, 50, dtype=int)
    valid_seed_list = np.linspace(51, 55, 5, dtype=int)

    # train_seed_list = np.linspace(1, 10, 10, dtype=int)
    # valid_seed_list = np.linspace(11, 15, 5, dtype=int)

    main(dataset, dimension, train_seed_list, valid_seed_list)
