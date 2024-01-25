#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:miyuan 
@license: Apache Licence 
@file: imageUtils.py 
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
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import imageio
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams['backend'] = 'SVG'
import seaborn as sns
from decimal import Decimal
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
from scipy.interpolate import griddata
from tqdm import tqdm
from piqa import SSIM, PSNR, TV, MS_SSIM, LPIPS, GMSD, MS_GMSD, MDSI, HaarPSI, VSI, FSIM
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from utils import dataUtils


# 查看数据集
def plot_all_data(item, args):
    u_x, u_y, pos, edge_index, type, all_time = dataUtils.process_ux(args, item)
    # u_y = u_y.reshape(-1,(args.height+1)*(args.width+1),2)
    test_out = torch.unsqueeze(u_y, dim=1)
    truth_data = torch.unsqueeze(u_y, dim=1)
    plot_customed_data(pos.cpu(), test_out.cpu(), truth_data.cpu(), args, item)


# 绘制损失函数曲线
def plot_loss_curve(img_path, loss, stage):
    plt.switch_backend('Agg')

    plt.figure()
    x = range(0, len(loss))
    if stage == 0:
        plt.plot(x, loss, 'b', label='TrainLoss')
        plt.ylabel('train_loss')
        plt.xlabel('iter_num')
        plt.legend()
        plt.savefig(img_path)
    elif stage == 1:
        plt.plot(x, loss, 'r', label='TestLoss')
        plt.ylabel('test_loss')
        plt.xlabel('iter_num')
        plt.legend()
        plt.savefig(img_path)
    elif stage == 2:
        plt.plot(x, loss, 'r', label='ValidLoss')
        plt.ylabel('valid_loss')
        plt.xlabel('iter_num')
        plt.legend()
        plt.savefig(img_path)
    plt.close('all')


def subplot_reg_plt(data, x, ax_):
    # 计算均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    relative_std = std / mean
    # 计算浮动区间
    lower_bound = mean - std
    upper_bound = mean + std

    truth = np.ones_like(mean)

    ax_.plot(x, truth, color='red', label='Reference')
    ax_.plot(x, mean, color='blue', label='Mean pccs score')
    # ax_.plot(x, lower_bound, label='lower pccs score', color='none')
    # ax_.plot(x, upper_bound, label='upper pccs score', color='none')
    ax_.plot(x, lower_bound, color='none')
    ax_.plot(x, upper_bound, color='none')

    # plt.fill_betweenx([lower_bound, upper_bound], 0, 1, color='yellow', alpha=0.1)
    ax_.fill_between(x, lower_bound, upper_bound, color='cyan', alpha=0.1)

    # sns.set_style(style="darkgrid")
    # data = np.array(pccs_loss_list[i]).T
    # sns.lineplot(ax=ax_, data=data,style="darkgrid")
    # sns.lineplot(ax=ax_, data=data, )
    # sns.regplot(ax=ax_, data=data)
    # sns.lmplot(data=data)
    # plt.show()

    xticks = [0, 10, 20]
    ax_.set_xlim(0, 20)
    ax_.set_xticks(xticks)
    # ax_.set_ylim(0, 1)

    # ax_.spines['bottom'].set_position('zero')

    # 去除上方和右侧边框线
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)

    ax_.set_xlabel('timestamp')
    ax_.set_ylabel('correlation')

    ax_.legend(loc='best')

    # ax_.set_aspect('equal')
    # ax_.set_title(title)


def subplot_reg_sns(data, x, ax_, cmaps):
    sns.set_style(style="darkgrid")
    # ax_.set_aspect('equal')
    # sns.lineplot(ax=ax_, data=data,style="darkgrid")
    # sns.lineplot(ax=ax_, data=data.T, linewidth=4,
    #              linestyles=['solid', 'dashed', 'dotted', 'dashdot', 'solid', 'dashed', 'dotted', 'dashdot', 'solid'],
    #              color=['red','blue','green','yellow','black','gray','orange','cyan','magenta'])
    # linestyles = ["-", "--", "-."]

    sns.lineplot(ax=ax_, data=data.T, linewidth=2,
                 # style=['-', '--', ':', '-.', '-', '--', ':', '-.', '-'],
                 palette=cmaps)
    # 设置图例
    # handles, labels = ax_.get_legend_handles_labels()
    # ax_.legend(handles, labels, loc='best')
    # ax_.set_prop_cycle(plt.cycler('linestyle', linestyles))
    # sns.regplot(ax=ax_, data=data)
    # sns.lmplot(data=data)
    # plt.show()
    # ax_.set_xlabel('timestamp', fontsize=14)
    # ax_.set_ylabel('correlation', fontsize=14)

    # plt.legend(title='Legend', labels=['Legend 1', 'Legend 2'])
    # xticks = [0, 21, 41, 61, 81, 101,121,141,161,180]
    # ax_.set_xticks(xticks)
    # 设置x轴刻度和标签的字体大小
    ax_.tick_params(axis='x', labelsize=18)
    # 设置y轴刻度和标签的字体大小
    ax_.tick_params(axis='y', labelsize=18)
    # 设置x轴刻度的范围和间隔
    # ax_.set_xlim(1, 20)
    ax_.set_xticks(np.arange(0, 21, 1))
    # 设置y轴刻度的范围和间隔
    ax_.set_ylim(0, 1)
    ax_.set_yticks(np.arange(0, 1.1, 0.1))
    # ax_.set_ylim(0, 100)
    # ax_.set_yticks(range(0, 101, 20))
    # 设置图形标题
    # ax_.set_title('示例图形', fontsize=16)
    # for i in range(y.shape[1]):
    #     line, = sns.lineplot(ax=ax, x=x, y=y[:, i])
    #     lines.append(line)
    # legend_markers = [plt.Line2D([0], [0], color=line.get_color(), marker='o', linestyle='') for line in lines]

    # cbar = plt.colorbar(cmaps, ax=ax_)

    legend_markers = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=5, color=color) for color in cmaps]
    ax_.legend(handles=legend_markers,  # fontsize='small', # title='time',
               labels=['1-20', '21-40', '41-60', '61-80', '81-100', '101-120', '121-140', '141-160', '161-180'],
               loc='upper right', bbox_to_anchor=(1.0, 1.0))

    ax_.legend().remove()
    # ax_.set_title(title)


def plot_pccs_score(predict_data, truth_data, args, item):
    # 判断图片存储路径是否存在，不存在则新建
    if not os.path.exists(args.fig_save_path):
        print("图片存储路径不存在，已新建")
        os.makedirs(args.fig_save_path)
    time_steps = predict_data.shape[0]

    # list = [predict_data, truth_data]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    interval_step = 1
    fig_save_path = args.fig_save_path

    # (3001, 100* 100, 2)
    #####################################
    # test_out_ = torch.clamp(test_out.cpu(), 0, 1)
    # truth_data_ = torch.clamp(truth_data.cpu(), 0, 1)

    # (score, diff) = ssim_loss(grayA, grayB, win_size=7, full=True, data_range=1)
    # diff = (diff * 255).astype("uint8")

    # ssim = SSIM()
    # print('ssim loss:', ssim(test_out_, truth_data_))
    # ms_ssim = MS_SSIM()
    # print(ms_ssim(test_out_, truth_data_))

    # cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    # print(cos_loss(test_out_.view(-1), truth_data_.view(-1)))
    #####################################

    # pccs = np.corrcoef(test_out.cpu(), truth_data.cpu())
    pccs_loss_list = []
    # print('psnr loss:', psnr(test_out_, truth_data_))
    for i in range(args.num_classes):
        f_list = []
        output = predict_data[..., i:i + 1]
        truth = truth_data[..., i:i + 1]
        # [1000,1,1000,2]
        output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
        truth = truth.permute(0, 3, 1, 2)
        for num in range(0, time_steps, args.roll_step):
            list = [1]
            for idx in range(num, num + args.roll_step, 1):
                star = truth[num:idx + 1, 0, ...].contiguous().view(-1)  # [1,1,1000]
                pred = output[num:idx + 1, 0, ...].contiguous().view(-1)
                # star_ = torch.clamp(star, 0, 1)
                # pred_ = torch.clamp(pred, 0, 1)
                star_ = star
                pred_ = pred

                from scipy.stats import pearsonr
                pccs_loss, _ = pearsonr(star_, pred_)
                # pccs_loss = np.corrcoef(star_, pred_) #[2,2]
                list.append(pccs_loss)
            f_list.append(list)
        pccs_loss_list.append(f_list)

    fig, ax = plt.subplots(nrows=args.num_classes, ncols=1, figsize=(3, 4 * args.num_classes))

    cmaps = ['red', 'blue', 'green', 'purple', 'black', 'gray', 'brown', 'indigo', 'magenta']
    for i in range(args.num_classes):
        line_num = len(pccs_loss_list[i])
        x = range(0, len(pccs_loss_list[i][0]))
        ax_ = ax if args.num_classes == 1 else ax[i]
        # ax_.plot(x, pccs_loss_list[i], label='pccs score')
        data = np.array(pccs_loss_list[i])

        subplot_reg_plt(data, x, ax_)
        # subplot_reg_sns(data, x, ax_, cmaps)

    # plt.legend()
    plt.tight_layout()
    plt.savefig(fig_save_path + '{}d_{}_pccs_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    print("已成功存储所有图片->{}".format(args.fig_save_path))


def plot_psnr_score(predict_data, truth_data, args, item):
    # 判断图片存储路径是否存在，不存在则新建
    if not os.path.exists(args.fig_save_path):
        print("图片存储路径不存在，已新建")
        os.makedirs(args.fig_save_path)
    time_steps = predict_data.shape[0]

    # list = [predict_data, truth_data]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    interval_step = 1
    fig_save_path = args.fig_save_path

    # (3001, 100* 100, 2)
    #####################################
    # test_out_ = torch.clamp(test_out.cpu(), 0, 1)
    # truth_data_ = torch.clamp(truth_data.cpu(), 0, 1)

    # (score, diff) = ssim_loss(grayA, grayB, win_size=7, full=True, data_range=1)
    # diff = (diff * 255).astype("uint8")

    # ssim = SSIM()
    # print('ssim loss:', ssim(test_out_, truth_data_))
    # ms_ssim = MS_SSIM()
    # print(ms_ssim(test_out_, truth_data_))

    # cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    # print(cos_loss(test_out_.view(-1), truth_data_.view(-1)))
    #####################################

    psnr = PSNR()
    psnr_loss_list = []
    # print('psnr loss:', psnr(test_out_, truth_data_))
    for i in range(args.num_classes):
        list = []
        output = predict_data[..., i:i + 1]
        truth_data = truth_data[..., i:i + 1]
        # [1000,1,1000,2]
        output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
        truth_data = truth_data.permute(0, 3, 1, 2)
        for num in range(time_steps):
            star = truth_data[num, 0:1, ...]  # [1,1,1000]
            pred = output[num, 0:1, ...]
            star_ = torch.clamp(star, 0, 1)
            pred_ = torch.clamp(pred, 0, 1)
            # star_ = star
            # pred_ = pred

            psnr_loss = psnr(star_, pred_)
            list.append(psnr_loss)
        psnr_loss_list.append(list)

    fig, ax = plt.subplots(nrows=args.num_classes, ncols=1, figsize=(3 * args.num_classes, 4))

    for i in range(args.num_classes):
        x = range(0, len(psnr_loss_list[i]))
        ax_ = ax if args.num_classes == 1 else ax[i]
        ax_.plot(x, psnr_loss_list[i], label='psnr score')
        # ax_.set_aspect('equal')
        # ax_.set_title(title)

    # plt.ylabel('psnr score')
    # plt.xlabel('iter_num')
    # plt.legend()

    plt.savefig(fig_save_path + '{}d_{}_psnr_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    print("已成功存储所有图片->{}".format(args.fig_save_path))


def plot_3D_slices(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    # uv: [t,c,h,w], [1000,2,10,10,10]
    output = output.permute(0, 1, 3, 2)  # [1000,1,2,1000]
    output = output.reshape(output.shape[0], output.shape[1], output.shape[2], args.length, args.width,
                            args.height).cpu().numpy()

    truth_data = truth_data.permute(0, 1, 3, 2)  # [1000,1,2,1000]
    truth_data = truth_data.reshape(truth_data.shape[0], truth_data.shape[1], truth_data.shape[2], args.length,
                                    args.width, args.height).cpu().numpy()

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0, 0:1, ...]
    u_pred = output[num, 0, 0:1, ...]
    v_star = truth_data[num, 0, 1:2, ...]
    v_pred = output[num, 0, 1:2, ...]
    u_error = np.abs(u_star - u_pred)
    v_error = np.abs(v_star - v_pred)

    x_star, y_star, z_star = position[num, :, 0], position[num, :, 1], position[num, :, 2]

    # 创建三个视角的切片
    u_pred_slices = [
        {'name': 'U-XY-plane', 'slice': u_pred[0, :, :, args.height // 2]},  # XY平面切片
        {'name': 'U-YZ-plane', 'slice': u_pred[0, args.length // 2, :, :]},  # YZ平面切片
        {'name': 'U-XZ-plane', 'slice': u_pred[0, :, args.width // 2, :]}  # XZ平面切片
    ]
    v_pred_slices = [
        {'name': 'V-XY-plane', 'slice': v_pred[0, :, :, args.height // 2]},  # XY平面切片
        {'name': 'V-YZ-plane', 'slice': v_pred[0, args.length // 2, :, :]},  # YZ平面切片
        {'name': 'V-XZ-plane', 'slice': v_pred[0, :, args.width // 2, :]}  # XZ平面切片
    ]

    u_star_slices = [
        {'name': 'U-XY-plane', 'slice': u_star[0, :, :, args.height // 2]},  # XY平面切片
        {'name': 'U-YZ-plane', 'slice': u_star[0, args.length // 2, :, :]},  # YZ平面切片
        {'name': 'U-XZ-plane', 'slice': u_star[0, :, args.width // 2, :]}  # XZ平面切片
    ]
    v_star_slices = [
        {'name': 'V-XY-plane', 'slice': v_star[0, :, :, args.height // 2]},  # XY平面切片
        {'name': 'V-YZ-plane', 'slice': v_star[0, args.length // 2, :, :]},  # YZ平面切片
        {'name': 'V-XZ-plane', 'slice': v_star[0, :, args.width // 2, :]}  # XZ平面切片
    ]

    # 创建图形对象
    # fig = go.Figure()
    # fig = make_subplots(rows=2, cols=3, subplot_titles=['u_truth', 'u_pred', 'u_error', 'v_truth', 'v_pred', 'v_error'],
    #                     horizontal_spacing=0.2, vertical_spacing=0.1,  # 2行 3列
    #                     specs=[[{'type': type}, {'type': type}, {'type': type}],
    #                            [{'type': type}, {'type': type}, {'type': type}]])
    type = 'Heatmap'
    fig = make_subplots(rows=4, cols=3, horizontal_spacing=0.2, vertical_spacing=0.1,  # 4行 3列
                        specs=[[{'type': type}, {'type': type}, {'type': type}],
                               [{'type': type}, {'type': type}, {'type': type}],
                               [{'type': type}, {'type': type}, {'type': type}],
                               [{'type': type}, {'type': type}, {'type': type}]])
    fig.update_layout(autosize=True, width=1600, height=1600, title_font_family="Times New Roman",
                      title_text='{}d_{}_{}_切片'.format(args.dimension, args.dataset_name, num))

    # 添加三个切片
    count = 1
    for slice_data in v_pred_slices:
        fig.add_trace(go.Heatmap(z=slice_data['slice'], name=slice_data['name'], showscale=False, showlegend=False,
                                 autocolorscale=False), row=1, col=count)
        count += 1
    count = 1
    for slice_data in v_star_slices:
        fig.add_trace(go.Heatmap(z=slice_data['slice'], name=slice_data['name'], showscale=False, showlegend=False,
                                 autocolorscale=False), row=2, col=count)
        count += 1
    count = 1
    for slice_data in u_pred_slices:
        fig.add_trace(go.Heatmap(z=slice_data['slice'], name=slice_data['name'], showscale=False, showlegend=False,
                                 autocolorscale=False), row=3, col=count)
        count += 1
    count = 1
    for slice_data in u_star_slices:
        fig.add_trace(go.Heatmap(z=slice_data['slice'], name=slice_data['name'], showscale=False, showlegend=False,
                                 autocolorscale=False), row=4, col=count)
        count += 1

    # 设置图形布局和轴标签
    # sc_dic = dict(xaxis_title='X轴', yaxis_title='Y轴', zaxis_title='Z轴')
    # fig.update_layout(scene=sc_dic, scene1=sc_dic, scene2=sc_dic, scene3=sc_dic, scene4=sc_dic, scene5=sc_dic,
    #                   scene6=sc_dic)

    # fig.update_layout(coloraxis=dict(colorscale='Viridis'), coloraxis_colorbar=dict(x=1.1, y=0.5, len=0.6))

    # 显示图形
    # fig.show()

    fig.write_image(fig_save_path + '{}d_{}_uv_slices_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item))


def get_lim(datas):
    # data = list(datas)
    min_lim = float(torch.min(datas))
    max_lim = float(torch.max(datas))

    min_lim_round = float(Decimal(min_lim).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP"))
    max_lim_round = float(Decimal(max_lim).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP"))

    lim = []
    if (min_lim_round - min_lim) > 0:
        lim.append(min_lim_round - 0.01)
    else:
        lim.append(min_lim_round)
    if (max_lim_round - max_lim) < 0:
        lim.append(max_lim_round + 0.01)
    else:
        lim.append(max_lim_round)

    return lim


# plt.contour
# plt.contourf
# sharex='col',sharey='row'

def plot_customed_data(position, predict_data, truth_data, args, item):
    # 判断图片存储路径是否存在，不存在则新建
    if not os.path.exists(args.fig_save_path):
        print("图片存储路径不存在，已新建")
        os.makedirs(args.fig_save_path)
    time_steps = predict_data.shape[0]

    # list = [predict_data, truth_data]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    if args.dataset_name == 'bs':
        interval_step = time_steps // 50
    else:
        interval_step = time_steps // 20

    with tqdm(total=predict_data.size(0), desc='生成图片中', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        if args.num_classes == 1:
            for i in range(0, time_steps, interval_step):
                if args.dimension == 2:
                    if args.dataset_name == 'bs':
                        plotR1C3_bs_D1(args.num_classes, position, predict_data, truth_data, num=i,
                                       fig_save_path=args.fig_save_path, item=item, args=args)
                    else:
                        plotR1C3(args.num_classes, position, predict_data, truth_data, num=i,
                                 fig_save_path=args.fig_save_path, item=item, args=args)
                    # 更新发呆进度
                    pbar.update(interval_step)
                elif args.dimension == 3:
                    print('plot 3 dimension')

                pbar.update(interval_step)
        elif args.num_classes == 2:
            for i in range(0, time_steps, interval_step):
                if args.dimension == 2:
                    if args.dataset_name == 'cf':
                        # u,v
                        plotR2C3_cf(args.num_classes, position, predict_data, truth_data, num=i,
                                    fig_save_path=args.fig_save_path, item=item, args=args)
                        # u**2+v**2
                        # plotR1C3_cf(args.num_classes, position, predict_data, truth_data, num=i,
                        #             fig_save_path=args.fig_save_path, item=item, args=args)
                    elif args.dataset_name == 'bs':
                        plotR1C3_bs_D2(args.num_classes, position, predict_data, truth_data, num=i,
                                       fig_save_path=args.fig_save_path, item=item, args=args)
                    else:
                        plotR2C3(args.num_classes, position, predict_data, truth_data, num=i,
                                 fig_save_path=args.fig_save_path, item=item, args=args)
                elif args.dimension == 3:
                    # plotR2C3_GO(args.num_classes, position, predict_data, truth_data, i, args.fig_save_path, item, args)
                    plotR2C3_GO_GS(args.num_classes, position, predict_data, truth_data, i, args.fig_save_path, item,
                                   args)
                    if i == 300:
                        plot_3D_slices(args.num_classes, position, predict_data, truth_data, i, args.fig_save_path,
                                       item, args)
                # 更新发呆进度
                pbar.update(interval_step)
        elif args.num_classes == 3:
            for i in range(0, time_steps, interval_step):
                if args.dimension == 2:
                    if args.dataset_name == 'bs':
                        # streamplot
                        plotR2C3_bs(args.num_classes, position, predict_data, truth_data, num=i,
                                    fig_save_path=args.fig_save_path, item=item, args=args)
                    elif args.dataset_name == 'cf':
                        plotR3C3_cf(args.num_classes, position, predict_data, truth_data, num=i,
                                    fig_save_path=args.fig_save_path, item=item, args=args)
                    else:
                        plotR3C3(args.num_classes, position, predict_data, truth_data, num=i,
                                 fig_save_path=args.fig_save_path, item=item, args=args)
                elif args.dimension == 3:
                    print('plot 3 dimension')
                # 更新发呆进度
                pbar.update(interval_step)

    print("已成功存储所有图片->{}".format(args.fig_save_path))

    processImage2gif(gif_save_path=args.fig_save_path, gif_save_name=args.gif_save_name, item=item, args=args)
    print("已成功生成gif图片->{}/{}_{}.gif".format(args.fig_save_path, args.gif_save_name, item))

    # processImage2mp4(gif_save_path=args.fig_save_path, gif_save_name=args.gif_save_name, item=item, args=args)
    # print("已成功生成mp4视频->{}/{}_{}.mp4".format(args.fig_save_path, args.gif_save_name, item))


# 将时间切片png处理生成gif文件
def processImage2gif(gif_save_path, gif_save_name, item, args):
    images = []
    filenames = sorted(
        (fn for fn in os.listdir(gif_save_path) if
         fn.endswith('_{}.png'.format(item)) and fn.startswith(
             '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name))))
    # filenames = sorted(
    #     (fn.replace('svg', 'png') for fn in os.listdir(gif_save_path) if
    #      fn.endswith('_{}.svg'.format(item)) and fn.startswith(
    #          '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name))))

    # print(filenames)
    with tqdm(total=len(filenames), desc='提取图片中', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for filename in filenames:
            images.append(imageio.imread(gif_save_path + filename))
            # images.append(cv2.imread(gif_save_path + filename))
            # 更新发呆进度
            pbar.update(1)
    # imageio.mimsave(gif_save_path + gif_save_name + '_{}.gif'.format(item), images, duration=0.1)
    # fps越大越快 duration越小越快  duration = 1 / fps
    imageio.mimsave(gif_save_path + gif_save_name + '_{}.gif'.format(item), images, fps=10)

    # anim = animation.FuncAnimation(fig, animate_func, frames=len(rollout_data), blit=False, repeat=False, interval=200,
    #                                cache_frame_data=False)
    # anim.save(output_file, writer=writer, fps=fps)

# 将时间切片png处理生成gif文件
def processImage2mp4(gif_save_path, gif_save_name, item, args):
    filenames = sorted(
        (fn for fn in os.listdir(gif_save_path) if
         fn.endswith('_{}.png'.format(item)) and fn.startswith(
             '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name))))

    output_file = gif_save_path + gif_save_name + '_{}.mp4'.format(item)

    with tqdm(total=len(filenames), desc='生成视频中', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        with imageio.get_writer(output_file, fps=30) as writer:  # 设置帧率（fps）
            for filename in filenames:
                image = imageio.imread(gif_save_path + filename)
                writer.append_data(image)
                # 更新发呆进度
                pbar.update(1)

    print(f'视频已创建并保存为{output_file}')


def rmv_tick():
    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)


def _createColorBarVertical(fig, im, ax, ax_list, c_min, c_max, label_format="{:02.2f}", cmap='viridis',
                            label_title='velocity'):
    """Util method for plotting a colorbar next to a subplot
    """
    # get_position(): x0,y0,w,h
    # bbox = ax.get_position()
    # x0 = bbox.x0
    # y0 = bbox.y0
    # width = bbox.width
    # height = bbox.height
    # # Point 1: The bottom-left corner of the Axes.
    # # Point 2: The top-right corner of the Axes.
    # p = bbox.get_points()
    # # 左下右上
    # p0 = p.flatten()
    #
    # ax_cbar = fig.add_axes([p0[2] + 0.005, p0[1], 0.0075, p0[3] - p0[1]])
    # cax = plt.axes([p0[2] + 0.005, p0[1], 0.0075, p0[3] - p0[1]])
    # cf_cb = plt.colorbar(im, orientation='vertical', cax=cax, format='%.2f',ticks=[c_min, (c_max + c_min) / 2, c_max])

    # p1 = ax0[1, -1].get_position().get_points().flatten()
    # ax_cbar = fig.add_axes([p1[2] + 0.005, p1[1], 0.0075, p0[3] - p1[1]])
    # ax_cbar = fig.add_axes([p0[0], p0[1]-0.075, p0[2]-p0[0], 0.02])

    # ticks = np.linspace(0, 1, 5)
    # tickLabels = np.linspace(c_min, c_max, 5)
    # tickLabels = [label_format.format(t0) for t0 in tickLabels]
    # cbar = plt.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    # cbar.set_ticklabels(tickLabels)
    # horizontal
    # cf_cb = fig.colorbar(im, orientation='vertical', ax=ax_cbar, fraction=0.046, pad=0.1, shrink=0.9, format='%.2f',
    #                      ticks=[c_min, (c_max + c_min) / 2, c_max])  # vertical
    # cf_cb.set_label(label=label_title, loc='center')

    # cf_cb = fig.colorbar(im, orientation='vertical', ax=ax_cbar, format='%.2f',
    #                      ticks=[c_min, (c_max + c_min) / 2, c_max])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # ticks=[c_min, (c_max + c_min) / 2, c_max]
    #
    cb = plt.colorbar(im, cax=cax, ax=ax_list, orientation='vertical', extend='both', spacing='uniform')
    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 20, }
    cb.set_label(label_title, fontdict=font)  # 设置colorbar的标签字体及其大小

    # cb = plt.colorbar(im, ax=ax, orientation='vertical', extend='both', ticks=[-1, 0, 1])
    # cb.ax.arrow(0.5, 0.05, 0, 0.4, width=0.03, head_width=0.08, head_length=0.1, fc='k', ec='k')
    # cb.ax.arrow(0.5, 0.95, 0, -0.4, width=0.03, head_width=0.08, head_length=0.1, fc='k', ec='k')


def _createColorBarHorizontal(fig, im, ax, ax_list, c_min, c_max, label_format="{:02.2f}", cmap='viridis',
                              label_title='velocity'):
    """Util method for plotting a colorbar next to a subplot
    """
    # get_position(): x0,y0,w,h
    # bbox = ax.get_position()
    # x0 = bbox.x0
    # y0 = bbox.y0
    # width = bbox.width
    # height = bbox.height
    # # Point 1: The bottom-left corner of the Axes.
    # # Point 2: The top-right corner of the Axes.
    # p = bbox.get_points()
    # # 左下右上
    # p0 = p.flatten()
    #
    # ax_cbar = fig.add_axes([p0[2] + 0.005, p0[1], 0.0075, p0[3] - p0[1]])
    # cax = plt.axes([p0[2] + 0.005, p0[1], 0.0075, p0[3] - p0[1]])
    # cf_cb = plt.colorbar(im, orientation='vertical', cax=cax, format='%.2f',ticks=[c_min, (c_max + c_min) / 2, c_max])

    # p1 = ax0[1, -1].get_position().get_points().flatten()
    # ax_cbar = fig.add_axes([p1[2] + 0.005, p1[1], 0.0075, p0[3] - p1[1]])
    # ax_cbar = fig.add_axes([p0[0], p0[1]-0.075, p0[2]-p0[0], 0.02])

    # ticks = np.linspace(0, 1, 5)
    # tickLabels = np.linspace(c_min, c_max, 5)
    # tickLabels = [label_format.format(t0) for t0 in tickLabels]
    # cbar = plt.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    # cbar.set_ticklabels(tickLabels)
    # horizontal
    # cf_cb = fig.colorbar(im, orientation='vertical', ax=ax_cbar, fraction=0.046, pad=0.1, shrink=0.9, format='%.2f',
    #                      ticks=[c_min, (c_max + c_min) / 2, c_max])  # vertical
    # cf_cb.set_label(label=label_title, loc='center')

    # cf_cb = fig.colorbar(im, orientation='vertical', ax=ax_cbar, format='%.2f',
    #                      ticks=[c_min, (c_max + c_min) / 2, c_max])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    # ticks=[c_min, (c_max + c_min) / 2, c_max]
    #
    cb = plt.colorbar(im, cax=cax, ax=ax_list, orientation='horizontal', extend='both', spacing='uniform',
                      ticks=[c_min, (c_max + c_min) / 2, c_max])
    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 20, }
    cb.set_label(label_title, fontdict=font)  # 设置colorbar的标签字体及其大小

    # cb = plt.colorbar(im, ax=ax, orientation='vertical', extend='both', ticks=[-1, 0, 1])
    # cb.ax.arrow(0.5, 0.05, 0, 0.4, width=0.03, head_width=0.08, head_length=0.1, fc='k', ec='k')
    # cb.ax.arrow(0.5, 0.95, 0, -0.4, width=0.03, head_width=0.08, head_length=0.1, fc='k', ec='k')


def subplot_mesh(x_star, y_star):
    # x_star [10000,1] y_star [10000,1] z_star [10000,1] c [1,1,1000]
    # viridis
    # ax.imshow(im, aspect='equal')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax.set_aspect('equal')
    # ax.set_title(title, fontsize=20)
    # ax.set_title(title)
    # x = [[1, 3], [2, 5]]  # 要连接的两个点的坐标
    # y = [[4, 7], [6, 3]]

    for i in range(len(x_star)):
        print(i)
        # plt.plot(x_star[i], y_star[i], color='r')
        # plt.scatter(x_star[i], y_star[i], color='b')
        ax.plot([x_star[i][0], y_star[i][0]], [x_star[i][1], y_star[i][1]], 'ko-', color='red', markersize=0.05,
                linewidth=0.1, markerfacecolor='black')
        # ax.scatter([x_star[i][0], y_star[i][0]], [x_star[i][1], y_star[i][1]], color='b')

    # plt.plot(x_star, y_star, color='r')
    # plt.scatter(x_star, y_star, color='b')

    plt.savefig('mesh_{}.png'.format(len(x_star)))


def subplot_ax(ax, x_star, y_star, c, title, fig, args, u_min, u_max, label_title):
    # x_star [10000,1] y_star [10000,1] z_star [10000,1] c [1,1,1000]

    w = (int)((args.width) / args.space_step)
    h = (int)((args.height) / args.space_step)
    arr = c.flatten().reshape(w, h)
    # viridis icefire rainbow
    ax = sns.heatmap(arr, annot=False, ax=ax, cmap="rainbow", xticklabels=False, yticklabels=False, vmin=u_min,
                     vmax=u_max,
                     cbar=True, cbar_kws={'format': '%.2f', 'ticks': [u_min, (u_max + u_min) / 2, u_max],
                                          # "shrink": 0.9, vertical, horizontal
                                          'fraction': 0.146, 'orientation': 'horizontal', 'pad': 0.01, 'shrink': 0.7})
    # ax.imshow(im, aspect='equal')
    ax.set_aspect('equal')
    # ax.set_title(title, fontsize=20)
    ax.set_title(title)

    # rmv_tick()
    cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)

    cbar.set_label(label=label_title, loc='center')
    # sns.despine(top=False, right=False, left=False, bottom=False)
    """
    # 不指定三角形，根据坐标按照Delaunay 三角剖分确定三角形
    triangulation = mtri.Triangulation(x_star, y_star)
    cf = ax.tripcolor(triangulation, c.flatten(), cmap='rainbow', zorder=1, vmin=u_min, vmax=u_max)
    cf_cb = fig.colorbar(cf, orientation='horizontal', ax=ax, fraction=0.046, pad=0.1, shrink=0.7, format='%.2f',
                         ticks=[u_min, (u_max + u_min) / 2, u_max])  # vertical

    # ax.triplot(triangulation, 'ko-', ms=0.05, lw=0.05)
    ax.triplot(triangulation, linewidth=0.1, color='black')
    """


def subplot_ax_tri(ax, triangulation, x_star, y_star, c, title, fig, args, u_min, u_max, label_title):
    ax.set_aspect('equal')
    # ax.set_title(title, fontsize=20)
    ax.set_title(title)
    ax.set_axis_off()
    # ax.set_position([0, 0, 1, 1])

    # RdBu rainbow viridis seismic bwr
    cf = ax.tripcolor(triangulation, c.flatten(), cmap='rainbow', zorder=1, vmin=u_min, vmax=u_max,shading='gouraud')
    # cf = ax.tripcolor(triangulation, c.flatten(), cmap='rainbow', zorder=1, vmin=-5, vmax=5)
    # cf = ax.tricontourf(triangulation, c.flatten(), cmap='rainbow', zorder=1, vmin=u_min, vmax=u_max)
    cf_cb = fig.colorbar(cf, orientation='horizontal', ax=ax, fraction=0.046, pad=0.1, shrink=0.9, format='%.2f',
                         # ticks=[u_min, (u_max + u_min) / 2, u_max])  # vertical
                         ticks=[u_min, 0, u_max])  # vertical
    cf_cb.set_label(label=label_title, loc='center')

    # ax.triplot(triangulation, 'ko-', ms=0.05, lw=0.05)
    # ax.triplot(triangulation, linewidth=0.1, color='black')

    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)
    return cf


def subplot_ax_quiver(ax, triangulation, x_star, y_star, u, v, title, fig, args, u_min, u_max, label_title):
    ax.set_aspect('equal')
    # ax.set_title(title, fontsize=20)
    ax.set_title(title)
    ax.set_axis_off()
    # X, Y, U, V 确定位置和对应的风速
    # width = 0.003,  # 箭杆箭身宽度
    # scale = 100, # 箭杆长度, 参数scale越小箭头越长
    # 画出风场，和箭头箭轴后，得说明 箭轴长度与风速的对应关系
    # 调用quiver可以生成 参考箭头 + label。
    # 传入quiver句柄 X=0.09, Y = 0.051, #确定 label 所在位置，都限制在[0,1]之间 U = 5, 参考箭头长度 表示风速为5m/s。
    # angle = 0, #参考箭头摆放角度。默认为0，即水平摆放 label='v:5m/s',
    # #箭头的补充：label的内容 + labelpos='S', #label在参考箭头的哪个方向; S表示南边
    # color = 'b',labelcolor = 'b', #箭头颜色 + label的颜色
    # fontproperties = font, #label 的字体设置：大小，样式，weight )
    # #由于风有U\V两个方向，最好设置两个方向的参考箭头 + label
    # normalized_u = (u.flatten() - torch.mean(u.flatten())) / torch.std(u.flatten())
    # normalized_v = (v.flatten() - torch.mean(v.flatten())) / torch.std(v.flatten())
    # values = torch.sqrt(normalized_u ** 2 + normalized_v ** 2)
    # # normalized_data = (values - torch.min(values)) / (torch.max(values) - torch.min(values))
    # normalized_data = (values - torch.mean(values)) / torch.std(values)
    normalized_u, normalized_v, normalized_data = u.flatten(), v.flatten(), torch.sqrt(
        u.flatten() ** 2 + v.flatten() ** 2)
    # rainbow viridis # color='green',  # scale=0.5,
    im = ax.quiver(x_star.flatten(), y_star.flatten(), normalized_u, normalized_v, normalized_data, cmap='viridis',
                   angles="xy", scale=None, scale_units="xy", zorder=2, width=0.005)
    # im_cb = fig.colorbar(im, orientation='horizontal', ax=ax, fraction=0.046, pad=0.1, shrink=0.7, format='%.2f',
    #                      ticks=[u_min, (u_max + u_min) / 2, u_max], extend='both')
    # im_cb.set_label(label=label_title, loc='center')

    # colorbar 带箭头
    # cbar = plt.colorbar(im, ax=ax, orientation='vertical', extend='both', ticks=[-1, 0, 1])
    # cbar.ax.arrow(0.5, 0.05, 0, 0.4, width=0.03, head_width=0.08, head_length=0.1, fc='k', ec='k')
    # cbar.ax.arrow(0.5, 0.95, 0, -0.4, width=0.03, head_width=0.08, head_length=0.1, fc='k', ec='k')

    # ax.triplot(triangulation, 'ko-', ms=0.05, lw=0.05)
    ax.triplot(triangulation, linewidth=0.1, color='black')
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)
    return im


def subplot_ax_stream(ax, triangulation, x_star, y_star, u, v, title, fig, args, u_min, u_max, label_title):
    ax.set_aspect('equal')
    # ax.set_title(title, fontsize=20)
    ax.set_title(title)
    ax.set_axis_off()
    # X, Y, U, V 确定位置和对应的风速
    # width = 0.003,  # 箭杆箭身宽度
    # scale = 100, # 箭杆长度, 参数scale越小箭头越长
    # 画出风场，和箭头箭轴后，得说明 箭轴长度与风速的对应关系
    # 调用quiver可以生成 参考箭头 + label。
    # 传入quiver句柄 X=0.09, Y = 0.051, #确定 label 所在位置，都限制在[0,1]之间 U = 5, 参考箭头长度 表示风速为5m/s。
    # angle = 0, #参考箭头摆放角度。默认为0，即水平摆放 label='v:5m/s',
    # #箭头的补充：label的内容 + labelpos='S', #label在参考箭头的哪个方向; S表示南边
    # color = 'b',labelcolor = 'b', #箭头颜色 + label的颜色
    # fontproperties = font, #label 的字体设置：大小，样式，weight )
    # #由于风有U\V两个方向，最好设置两个方向的参考箭头 + label
    im = ax.streamplot(x_star.flatten(), y_star.flatten(), u.flatten(), v.flatten(), density=[0.5, 1])
    im_cb = fig.colorbar(im, orientation='horizontal', ax=ax, fraction=0.046, pad=0.1, shrink=0.7, format='%.2f',
                         ticks=[u_min, (u_max + u_min) / 2, u_max])
    im_cb.set_label(label=label_title, loc='center')
    # ax.triplot(triangulation, 'ko-', ms=0.05, lw=0.05)
    ax.triplot(triangulation, linewidth=0.1, color='black')
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)


def subplot_ax_quiver_tri(ax, triangulation, x_star, y_star, u, v, c, title, fig, args, u_min, u_max, p_min, p_max):
    ax.set_aspect('equal')
    # ax.set_aspect('auto')
    # ax.set_title(title, fontsize=20)
    ax.set_title(title)
    ax.set_axis_off()
    # X, Y, U, V 确定位置和对应的风速
    # width = 0.003,  # 箭杆箭身宽度
    # scale = 100, # 箭杆长度, 参数scale越小箭头越长
    # 画出风场，和箭头箭轴后，得说明 箭轴长度与风速的对应关系
    # 调用quiver可以生成 参考箭头 + label。
    # 传入quiver句柄 X=0.09, Y = 0.051, #确定 label 所在位置，都限制在[0,1]之间 U = 5, 参考箭头长度 表示风速为5m/s。
    # angle = 0, #参考箭头摆放角度。默认为0，即水平摆放 label='v:5m/s',
    # #箭头的补充：label的内容 + labelpos='S', #label在参考箭头的哪个方向; S表示南边
    # color = 'b',labelcolor = 'b', #箭头颜色 + label的颜色
    # fontproperties = font, #label 的字体设置：大小，样式，weight )
    # #由于风有U\V两个方向，最好设置两个方向的参考箭头 + label
    im = ax.quiver(x_star.flatten(), y_star.flatten(), u.flatten(), v.flatten(), angles="xy",
                   scale=None,  # color = 'b', # scale=0.5,
                   scale_units="xy", zorder=None, width=0.005)
    im_cb = fig.colorbar(im, orientation='horizontal', ax=ax, fraction=0.046, pad=0.1, shrink=0.7, format='%.2f',
                         ticks=[u_min, (u_max + u_min) / 2, u_max])
    im_cb.set_label(label='wind speed', loc='center')

    cf = ax.tripcolor(triangulation, c.flatten(), cmap='rainbow', zorder=None, vmin=p_min, vmax=p_max)
    cf_cb = fig.colorbar(cf, orientation='vertical', ax=ax, fraction=0.046, pad=0.1, shrink=0.7, format='%.2f',
                         ticks=[p_min, (p_max + p_min) / 2, p_max])
    cf_cb.set_label(label='temperature', loc='center')
    # ax.triplot(triangulation, 'ko-', ms=0.05, lw=0.05)

    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)


def subplot_ax_scatter(ax, x_star, y_star, c, title, fig, args, u_min, u_max):
    # ax.set_aspect('auto')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=20)
    # x_star [10000,1] y_star [10000,1] z_star [10000,1] c [1,1,1000]
    cf = ax.scatter(x_star.flatten(), y_star.flatten(), c=c.flatten(), marker='s', alpha=0.4, s=4, cmap='rainbow',
                    vmin=u_min, vmax=u_max)
    fig.colorbar(cf, orientation='horizontal', ax=ax, fraction=0.046, pad=0.1, shrink=0.7)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)


# 生成1行3列png
def plotR1C3(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # (3001, 1,1000, 1)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0:1, ...]
    u_pred = output[num, 0:1, ...]

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    list = [u_star]
    u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 U
    subplot_ax(ax[0], x_star, y_star, u_star, 'u (Groud Truth.)' + str(num), fig, args, u_min, u_max,
               label_title='vorticity')

    # list = [u_pred]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 U
    subplot_ax(ax[1], x_star, y_star, u_pred, 'u (Pred.)' + str(num), fig, args, u_min, u_max, label_title='vorticity')

    # list = [torch.abs(u_pred - u_star)]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 U
    subplot_ax(ax[2], x_star, y_star, torch.abs(u_pred - u_star), 'u (Abs Err.)' + str(num), fig, args, u_min, u_max,
               label_title='vorticity')

    # 存储图片
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    plt.close('all')


# 生成2行3列png
def plotR2C3(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # (3001, 100* 100, 2)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0:1, ...]  # [1,1,1000]
    u_pred = output[num, 0:1, ...]
    v_star = truth_data[num, 1:2, ...]
    v_pred = output[num, 1:2, ...]

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    list = [u_star]
    u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 U
    subplot_ax(ax[0, 0], x_star, y_star, u_star, 'u (Groud Truth.)' + str(num), fig, args, u_min, u_max,
               label_title='velocity')

    # list = [u_pred]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 U
    subplot_ax(ax[0, 1], x_star, y_star, u_pred, 'u (Pred.)' + str(num), fig, args, u_min, u_max,
               label_title='velocity')

    # list = [torch.abs(u_pred - u_star)]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 U
    subplot_ax(ax[0, 2], x_star, y_star, torch.abs(u_pred - u_star), 'u (Abs Err.)' + str(num), fig, args, u_min, u_max,
               label_title='velocity')

    list = [v_star]
    v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 V
    subplot_ax(ax[1, 0], x_star, y_star, v_star, 'v (Groud Truth.)' + str(num), fig, args, v_min, v_max,
               label_title='velocity')

    # list = [v_pred]
    # v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 V
    subplot_ax(ax[1, 1], x_star, y_star, v_pred, 'v (Pred.)' + str(num), fig, args, v_min, v_max,
               label_title='velocity')

    # list = [torch.abs(v_pred - v_star)]
    # v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 V
    subplot_ax(ax[1, 2], x_star, y_star, torch.abs(v_pred - v_star), 'v (Abs Err.)' + str(num), fig, args, v_min, v_max,
               label_title='velocity')

    # 存储图片
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    plt.close('all')


# 生成3行3列png
def plotR3C3(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # (3001, 100* 100, 2)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0:1, ...]
    u_pred = output[num, 0:1, ...]
    v_star = truth_data[num, 1:2, ...]
    v_pred = output[num, 1:2, ...]
    p_star = truth_data[num, 2:3, ...]
    p_pred = output[num, 2:3, ...]

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

    list = [u_star]
    u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 U
    subplot_ax(ax[0, 0], x_star, y_star, u_star, 'u (Groud Truth.)' + str(num), fig, args, u_min, u_max,
               label_title='velocity')

    # list = [u_pred]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 U
    subplot_ax(ax[0, 1], x_star, y_star, u_pred, 'u (Pred.)' + str(num), fig, args, u_min, u_max,
               label_title='velocity')

    # list = [torch.abs(u_pred - u_star)]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 U
    subplot_ax(ax[0, 2], x_star, y_star, torch.abs(u_pred - u_star), 'u (Abs Err.)' + str(num), fig, args, u_min, u_max,
               label_title='velocity')

    list = [v_star]
    v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 V
    subplot_ax(ax[1, 0], x_star, y_star, v_star, 'v (Groud Truth.)' + str(num), fig, args, v_min, v_max,
               label_title='velocity')

    # list = [v_pred]
    # v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 V
    subplot_ax(ax[1, 1], x_star, y_star, v_pred, 'v (Pred.)' + str(num), fig, args, v_min, v_max,
               label_title='velocity')

    # list = [torch.abs(v_pred - v_star)]
    # v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 V
    subplot_ax(ax[1, 2], x_star, y_star, torch.abs(v_pred - v_star), 'v (Abs Err.)' + str(num), fig, args, v_min, v_max,
               label_title='velocity')

    list = [p_star]
    p_min, p_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 P
    subplot_ax(ax[2, 0], x_star, y_star, p_star, 'p (Groud Truth.)' + str(num), fig, args, p_min, p_max,
               label_title='pressure')

    # list = [p_pred]
    # p_min, p_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据  P
    subplot_ax(ax[2, 1], x_star, y_star, p_pred, 'p (Pred.)' + str(num), fig, args, p_min, p_max,
               label_title='pressure')

    # list = [torch.abs(p_pred - p_star)]
    # p_min, p_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 P
    subplot_ax(ax[2, 2], x_star, y_star, torch.abs(p_pred - p_star), 'p (Abs Err.)' + str(num), fig, args, p_min, p_max,
               label_title='pressure')

    # 存储图片
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    plt.close('all')


# 生成3行3列png
def plotR3C3_cf(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # predicted_v = np.linalg.norm(predicted, axis=-1)
    # for ax in axes:
    #     ax.cla()
    #     ax.triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)
    #
    # handle1 = axes[0].tripcolor(triang, target_v, vmax=v_max, vmin=v_min, edgecolors='k')
    # axes[1].tripcolor(triang, predicted_v, vmax=v_max, vmin=v_min, edgecolors='k')
    # # handle2 = axes[2].tripcolor(triang, diff, vmax=1, vmin=0)
    #
    # axes[0].set_title('Target\nTime @ %.2f s' % (step * 0.01))
    # axes[1].set_title('Prediction\nTime @ %.2f s' % (step * 0.01))
    # # axes[2].set_title('Difference\nTime @ %.2f s'%(step*0.01))
    # colorbar1 = fig.colorbar(handle1, ax=[axes[0], axes[1]])

    # (3001, 100* 100, 2)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # if args.dataset_name == 'cf':
    #     face_path = os.path.join('{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
    #                                                                     args.dataset_name, args.all_time_steps,
    #                                                                     args.mesh_node, item))
    #     face = np.load(face_path)
    #     face = torch.from_numpy(face).float()

    # 不指定三角形，根据坐标按照Delaunay 三角剖分确定三角形
    # triangulation = mtri.Triangulation(x_star, y_star)
    # 指定三角形，comsol生成的mesh elements
    face_path = os.path.join(
        '{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                               args.all_time_steps, args.mesh_node, item))
    face = np.load(face_path, allow_pickle=True)
    triangulation = mtri.Triangulation(x_star, y_star, face)
    # triangulation = face

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0:1, ...]
    u_pred = output[num, 0:1, ...]
    v_star = truth_data[num, 1:2, ...]
    v_pred = output[num, 1:2, ...]
    p_star = truth_data[num, 2:3, ...]
    p_pred = output[num, 2:3, ...]

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7))

    # fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12 * args.num_classes, 12))
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(4 * args.num_classes, 8))
    # fig.subplots_adjust(right=0.9)
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # fig.tight_layout()

    list = [u_star]
    u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 U
    _ = subplot_ax_tri(ax[0, 0], triangulation, x_star, y_star, u_star, 'u (Groud Truth.)' + str(num), fig, args,
                       u_min, u_max, label_title='velocity')

    # list = [u_pred]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 U
    _ = subplot_ax_tri(ax[0, 1], triangulation, x_star, y_star, u_pred, 'u (Pred.)' + str(num), fig, args, u_min,
                       u_max, label_title='velocity')

    # list = [torch.abs(u_pred - u_star)]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 U
    im = subplot_ax_tri(ax[0, 2], triangulation, x_star, y_star, torch.abs(u_pred - u_star), 'u (Abs Err.)' + str(num),
                        fig, args, u_min, u_max, label_title='velocity')
    """
    list = [u_pred, u_star, torch.abs(u_pred - u_star)]
    c_min, c_max = get_lim(torch.cat(list, dim=0))
    # _createColorBarVertical(fig, im, ax[0, 2], [ax[0, 0], ax[0, 1], ax[0, 2]], c_min, c_max, label_format="{:02.2f}",
    #                         cmap='rainbow', label_title='velocity')
    _createColorBarHorizontal(fig, im, ax[0, 0], [ax[0, 0], ax[0, 1], ax[0, 2]], c_min, c_max,
                              label_format="{:02.2f}", cmap='rainbow', label_title='velocity')
    """
    list = [v_star]
    v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 V
    _ = subplot_ax_tri(ax[1, 0], triangulation, x_star, y_star, v_star, 'v (Groud Truth.)' + str(num), fig, args,
                       v_min, v_max, label_title='velocity')

    # list = [v_pred]
    # v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 V
    _ = subplot_ax_tri(ax[1, 1], triangulation, x_star, y_star, v_pred, 'v (Pred.)' + str(num), fig, args, v_min,
                       v_max, label_title='velocity')

    # list = [torch.abs(v_pred - v_star)]
    # v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 V
    im = subplot_ax_tri(ax[1, 2], triangulation, x_star, y_star, torch.abs(v_pred - v_star), 'v (Abs Err.)' + str(num),
                        fig, args, v_min, v_max, label_title='velocity')
    """
    list = [v_pred, v_star, torch.abs(v_pred - v_star)]
    c_min, c_max = get_lim(torch.cat(list, dim=0))
    # _createColorBarVertical(fig, im, ax[1, 2], [ax[1, 0], ax[1, 1], ax[1, 2]], c_min, c_max, label_format="{:02.2f}",
    #                         cmap='rainbow', label_title='velocity')
    _createColorBarHorizontal(fig, im, ax[1, 0], [ax[1, 0], ax[1, 1], ax[1, 2]], c_min, c_max,
                              label_format="{:02.2f}",
                              cmap='rainbow', label_title='velocity')
    """
    list = [p_star]
    p_min, p_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 P
    _ = subplot_ax_tri(ax[2, 0], triangulation, x_star, y_star, p_star, 'p (Groud Truth.)' + str(num), fig, args,
                       p_min, p_max, label_title='pressure')

    # list = [p_pred]
    # p_min, p_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据  P
    _ = subplot_ax_tri(ax[2, 1], triangulation, x_star, y_star, p_pred, 'p (Pred.)' + str(num), fig, args, p_min,
                       p_max, label_title='pressure')

    # list = [torch.abs(p_pred - p_star)]
    # p_min, p_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 P
    im = subplot_ax_tri(ax[2, 2], triangulation, x_star, y_star, torch.abs(p_pred - p_star), 'p (Abs Err.)' + str(num),
                        fig, args, p_min, p_max, label_title='pressure')

    """
    list = [p_pred, p_star, torch.abs(p_pred - p_star)]
    c_min, c_max = get_lim(torch.cat(list, dim=0))
    # _createColorBarVertical(fig, im, ax[2, 2], [ax[2, 0], ax[2, 1], ax[2, 2]], c_min, c_max, label_format="{:02.2f}",
    #                         cmap='rainbow', label_title='pressure')
    _createColorBarHorizontal(fig, im, ax[2, 0], [ax[2, 0], ax[2, 1], ax[2, 2]], c_min, c_max,
                              label_format="{:02.2f}",
                              cmap='rainbow', label_title='pressure')
    """
    # 存储图片
    # plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
    #     5) + '_{}.svg'.format(item), dpi=300, format="svg")
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300, format="png")

    plt.close('all')


# 生成2行3列png
def plotR2C3_cf(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # (3001, 100* 100, 2)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # if args.dataset_name == 'cf':
    #     face_path = os.path.join('{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
    #                                                                     args.dataset_name, args.all_time_steps,
    #                                                                     args.mesh_node, item))
    #     face = np.load(face_path)
    #     face = torch.from_numpy(face).float()

    # 不指定三角形，根据坐标按照Delaunay 三角剖分确定三角形
    # triangulation = mtri.Triangulation(x_star, y_star)
    # 指定三角形，comsol生成的mesh elements
    face_path = os.path.join(
        '{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                               args.all_time_steps, args.mesh_node, item))
    face = np.load(face_path, allow_pickle=True)
    triangulation = mtri.Triangulation(x_star, y_star, face)

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0:1, ...]
    u_pred = output[num, 0:1, ...]
    v_star = truth_data[num, 1:2, ...]
    v_pred = output[num, 1:2, ...]

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    list = [u_star]
    u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 U
    subplot_ax_tri(ax[0, 0], triangulation, x_star, y_star, u_star, 'u (Groud Truth.)' + str(num), fig, args, u_min,
                   u_max, label_title='velocity')

    # list = [u_pred]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 U
    subplot_ax_tri(ax[0, 1], triangulation, x_star, y_star, u_pred, 'u (Pred.)' + str(num), fig, args, u_min, u_max,
                   label_title='velocity')

    # list = [torch.abs(u_pred - u_star)]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 U
    subplot_ax_tri(ax[0, 2], triangulation, x_star, y_star, torch.abs(u_pred - u_star), 'u (Abs Err.)' + str(num), fig,
                   args, u_min, u_max, label_title='velocity')

    list = [v_star]
    v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 V
    subplot_ax_tri(ax[1, 0], triangulation, x_star, y_star, v_star, 'v (Groud Truth.)' + str(num), fig, args, v_min,
                   v_max, label_title='velocity')

    # list = [v_pred]
    # v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 V
    subplot_ax_tri(ax[1, 1], triangulation, x_star, y_star, v_pred, 'v (Pred.)' + str(num), fig, args, v_min, v_max,
                   label_title='velocity')

    # list = [torch.abs(v_pred - v_star)]
    # v_min, v_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 V
    subplot_ax_tri(ax[1, 2], triangulation, x_star, y_star, torch.abs(v_pred - v_star), 'v (Abs Err.)' + str(num), fig,
                   args, v_min, v_max, label_title='velocity')

    # 存储图片
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    plt.close('all')


# 生成1行3列png
def plotR1C3_cf(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # (3001, 100* 100, 2)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # if args.dataset_name == 'cf':
    #     face_path = os.path.join('{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
    #                                                                     args.dataset_name, args.all_time_steps,
    #                                                                     args.mesh_node, item))
    #     face = np.load(face_path)
    #     face = torch.from_numpy(face).float()

    # 不指定三角形，根据坐标按照Delaunay 三角剖分确定三角形
    # triangulation = mtri.Triangulation(x_star, y_star)
    # 指定三角形，comsol生成的mesh elements
    face_path = os.path.join(
        '{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                               args.all_time_steps, args.mesh_node, item))
    face = np.load(face_path, allow_pickle=True)
    triangulation = mtri.Triangulation(x_star, y_star, face)

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0:1, ...]
    u_pred = output[num, 0:1, ...]
    v_star = truth_data[num, 1:2, ...]
    v_pred = output[num, 1:2, ...]

    # np.linalg.norm(predicted, axis=-1)
    norm_uv_star = u_star ** 2 + v_star ** 2
    norm_uv_pred = u_pred ** 2 + v_pred ** 2

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    list = [norm_uv_star]
    u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 未加噪音数据前的真实模拟数据 U
    subplot_ax_tri(ax[0], triangulation, x_star, y_star, norm_uv_star, 'norm(u,v) (Groud Truth.)' + str(num), fig, args,
                   u_min, u_max, label_title='maybe vorticity')

    # list = [norm_uv_pred]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 U
    subplot_ax_tri(ax[1], triangulation, x_star, y_star, norm_uv_pred, 'norm(u,v) (Pred.)' + str(num), fig, args, u_min,
                   u_max, label_title='maybe vorticity')

    # list = [torch.abs(norm_uv_star - norm_uv_pred)]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 误差数据 U
    subplot_ax_tri(ax[2], triangulation, x_star, y_star, torch.abs(norm_uv_star - norm_uv_pred),
                   'norm(u,v) (Abs Err.)' + str(num), fig, args, u_min, u_max, label_title='maybe vorticity')

    # 存储图片
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    plt.close('all')


def plotR2C3_GO(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    # uv: [t,c,h,w], [1000,2,10,10,10]
    output = output.permute(0, 1, 3, 2)  # [1000,1,2,1000]
    output = output.reshape(output.shape[0], output.shape[1], output.shape[2], args.length, args.width,
                            args.height).cpu().numpy()

    truth_data = truth_data.permute(0, 1, 3, 2)  # [1000,1,2,1000]
    truth_data = truth_data.reshape(truth_data.shape[0], truth_data.shape[1], truth_data.shape[2], args.length,
                                    args.width, args.height).cpu().numpy()

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0, 0:1, ...]
    u_pred = output[num, 0, 0:1, ...]
    v_star = truth_data[num, 0, 1:2, ...]
    v_pred = output[num, 0, 1:2, ...]

    x_star, y_star, z_star = position[num, :, 0], position[num, :, 1], position[num, :, 2]
    # if lres_tag == True:
    #     u = np.kron(u, np.ones((4, 4, 4)))
    type = 'Isosurface'  # 'volume'

    fig = make_subplots(rows=2, cols=3, subplot_titles=['u_truth', 'u_pred', 'u_error', 'v_truth', 'v_pred', 'v_error'],
                        specs=[[{'type': type}, {'type': type}, {'type': type}],
                               [{'type': type}, {'type': type}, {'type': type}]], horizontal_spacing=0,
                        vertical_spacing=0, )  # 2行 3列
    fig.update_layout(autosize=True, width=1000, height=800, title_font_family="Times New Roman",
                      title_text='{}d_{}_{}'.format(args.dimension, args.dataset_name, num))

    u_error = np.abs(u_star - u_pred)
    v_error = np.abs(v_star - v_pred)

    # go.Volume
    # go.Isosurface
    # 'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
    #              'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
    #              'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
    #              'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
    #              'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
    #              'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
    #              'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
    #              'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
    #              'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
    #              'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
    #              'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
    #              'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
    #              'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
    #              'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
    #              'ylorrd'].
    trace_1 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=u_star.flatten(),
                            opacity=0.5, colorscale='rdylbu', surface=dict(count=1), showscale=False,
                            showlegend=False,
                            # colorbar={'orientation': 'h', 'fraction': 0.046, 'pad': 0.1, 'shrink': 0.7})
                            colorbar={'orientation': 'h'}, autocolorscale=False)
    trace_2 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=u_pred.flatten(),
                            opacity=0.5, colorscale='rdylbu', surface=dict(count=1), showscale=False,
                            showlegend=False,
                            # colorbar={'orientation': 'h', 'fraction': 0.046, 'pad': 0.1, 'shrink': 0.7})
                            colorbar={'orientation': 'h'}, autocolorscale=False)
    trace_3 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=u_error.flatten(),
                            opacity=0.5, colorscale='rdylbu', surface=dict(count=1), showscale=False,
                            showlegend=False,
                            # colorbar={'orientation': 'h', 'fraction': 0.046, 'pad': 0.1, 'shrink': 0.7})
                            colorbar={'orientation': 'h'}, autocolorscale=False)
    trace_4 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=v_star.flatten(),
                            opacity=0.5, colorscale='rdylbu', surface=dict(count=1), showscale=False,
                            showlegend=False,
                            # colorbar={'orientation': 'h', 'fraction': 0.046, 'pad': 0.1, 'shrink': 0.7})
                            colorbar={'orientation': 'h'}, autocolorscale=False)
    trace_5 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=v_pred.flatten(),
                            opacity=0.5, colorscale='rdylbu', surface=dict(count=1), showscale=False,
                            showlegend=False,  # viridis
                            # colorbar={'orientation': 'h', 'fraction': 0.046, 'pad': 0.1, 'shrink': 0.7})
                            colorbar={'orientation': 'h'}, autocolorscale=False)
    trace_6 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=v_error.flatten(),
                            opacity=0.5, colorscale='rdylbu', surface=dict(count=1), showscale=False,
                            showlegend=False,
                            # colorbar={'orientation': 'h', 'fraction': 0.046, 'pad': 0.1, 'shrink': 0.7})
                            colorbar={}, autocolorscale=False)

    fig.add_trace(trace_1, row=1, col=1)
    fig.add_trace(trace_2, row=1, col=2)
    fig.add_trace(trace_3, row=1, col=3)

    fig.add_trace(trace_4, row=2, col=1)
    fig.add_trace(trace_5, row=2, col=2)
    fig.add_trace(trace_6, row=2, col=3)

    center = 6
    # scene_camera_eye = dict(x=0.1 * center, y=0.5 * center, z=0.2 * center)
    scene_camera_eye = dict(x=0.3 * center, y=0.5 * center, z=0.2 * center)
    sc_dic = dict(xaxis=dict(backgroundcolor='white', gridcolor="silver",
                             # tickvals=[0, 20],tickfont=dict(size=17, family='Times New Roman'),
                             # title=dict(font=dict(size=38, family='Times New Roman')),
                             ),
                  yaxis=dict(backgroundcolor='white', gridcolor="silver",
                             # tickvals=[0, 20], tickfont=dict(size=17, family='Times New Roman'),
                             # title=dict(font=dict(size=38, family='Times New Roman')),
                             ),
                  zaxis=dict(backgroundcolor='white', gridcolor="silver",
                             # tickvals=[0, 20], tickfont=dict(size=17, family='Times New Roman'),
                             # title=dict(font=dict(size=38, family='Times New Roman')),
                             ), camera=dict(eye=scene_camera_eye))

    fig.update_layout(scene=sc_dic, scene1=sc_dic, scene2=sc_dic, scene3=sc_dic, scene4=sc_dic, scene5=sc_dic,
                      scene6=sc_dic)

    # coloraxis_colorbar_dic = dict(yanchor="top", y=1, x=0)
    # coloraxis_colorbar_dic = dict(orientation='h')
    # cb_dic = dict(colorbar=coloraxis_colorbar_dic)
    #
    # fig.update_layout(coloraxis=cb_dic, coloraxis1=cb_dic, coloraxis2=cb_dic, coloraxis3=cb_dic, coloraxis4=cb_dic,
    #                   coloraxis5=cb_dic, coloraxis6=cb_dic)

    # tight layout
    # margin=dict(t=0, l=0, b=0),

    # fig.update_layout(scene=dict(xaxis_title='<i>x</i>', yaxis_title='<i>y</i>', zaxis_title='<i>z</i>'))
    # fig.update_xaxes(title_standoff=100)

    # fig.update_layout(
    #     coloraxis_colorbar=dict(len=0.65, tickfont=dict(size=40, color='black', family='Times New Roman'),
    #                             orientation='h', title="density", tickvals=[0.3, 0.5], ticktext=["red", "blue"]))

    fig.write_image(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item))


def plotR2C3_GO_GS(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    # uv: [t,c,h,w], [1000,2,10,10,10]
    output = output.permute(0, 1, 3, 2)  # [1000,1,2,1000]
    output = output.reshape(output.shape[0], output.shape[1], output.shape[2], args.length, args.width,
                            args.height).cpu().numpy()

    truth_data = truth_data.permute(0, 1, 3, 2)  # [1000,1,2,1000]
    truth_data = truth_data.reshape(truth_data.shape[0], truth_data.shape[1], truth_data.shape[2], args.length,
                                    args.width, args.height).cpu().numpy()

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0, 0:1, ...]
    u_pred = output[num, 0, 0:1, ...]
    v_star = truth_data[num, 0, 1:2, ...]
    v_pred = output[num, 0, 1:2, ...]
    u_error = np.abs(u_star - u_pred)
    v_error = np.abs(v_star - v_pred)

    x_star, y_star, z_star = position[num, :, 0], position[num, :, 1], position[num, :, 2]

    type = 'Isosurface'  # 'volume'
    fig = make_subplots(rows=2, cols=3, subplot_titles=['u_truth', 'u_pred', 'u_error', 'v_truth', 'v_pred', 'v_error'],
                        horizontal_spacing=0.0, vertical_spacing=0.0,  # 2行 3列
                        specs=[[{'type': type}, {'type': type}, {'type': type}],
                               [{'type': type}, {'type': type}, {'type': type}]])
    fig.update_layout(autosize=True, width=1200, height=1200, title_font_family="Times New Roman",
                      title_text='{}d_{}_{}'.format(args.dimension, args.dataset_name, num))

    trace_1 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=u_star.flatten(),
                            isomin=0.3, isomax=0.5, opacity=0.2, colorscale='RdBu',  # 'BlueRed', RdBu
                            surface_count=2,  # number of isosurfaces, 2 by default: only min and max
                            # colorbar_nticks=3,  # colorbar ticks correspond to isosurface values
                            # colorbar=dict(len=0.65, tickfont=dict(size=40, color='black', family='Times New Roman')),
                            showscale=False, showlegend=False, autocolorscale=False)
    trace_2 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=u_pred.flatten(),
                            isomin=0.3, isomax=0.5, opacity=0.2, colorscale='RdBu',  # 'BlueRed', RdBu
                            surface_count=2,  # number of isosurfaces, 2 by default: only min and max
                            # colorbar_nticks=3,  # colorbar ticks correspond to isosurface values
                            # colorbar=dict(len=0.65, tickfont=dict(size=40, color='black', family='Times New Roman')),
                            showscale=False, showlegend=False, autocolorscale=False)
    trace_3 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=u_error.flatten(),
                            isomin=0.3, isomax=0.5, opacity=0.2, colorscale='RdBu',  # 'BlueRed', RdBu
                            surface_count=2,  # number of isosurfaces, 2 by default: only min and max
                            # colorbar_nticks=3,  # colorbar ticks correspond to isosurface values
                            # colorbar=dict(len=0.65, tickfont=dict(size=40, color='black', family='Times New Roman')),
                            showscale=False, showlegend=False, autocolorscale=False)
    trace_4 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=v_star.flatten(),
                            isomin=0.1, isomax=0.3, opacity=0.2, colorscale='RdBu',  # 'BlueRed', RdBu
                            surface_count=2,  # number of isosurfaces, 2 by default: only min and max
                            # colorbar_nticks=3,  # colorbar ticks correspond to isosurface values
                            # colorbar=dict(len=0.65, tickfont=dict(size=40, color='black', family='Times New Roman')),
                            showscale=False, showlegend=False, autocolorscale=False)
    trace_5 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=v_pred.flatten(),
                            isomin=0.1, isomax=0.3, opacity=0.2, colorscale='RdBu',  # 'BlueRed', RdBu
                            surface_count=2,  # number of isosurfaces, 2 by default: only min and max
                            # colorbar_nticks=3,  # colorbar ticks correspond to isosurface values
                            # colorbar=dict(len=0.65, tickfont=dict(size=40, color='black', family='Times New Roman')),
                            showscale=False, showlegend=False, autocolorscale=False)
    trace_6 = go.Isosurface(x=x_star.flatten(), y=y_star.flatten(), z=z_star.flatten(), value=v_error.flatten(),
                            isomin=0.1, isomax=0.3, opacity=0.2, colorscale='RdBu',  # 'BlueRed', RdBu
                            surface_count=2,  # number of isosurfaces, 2 by default: only min and max
                            # colorbar_nticks=3,  # colorbar ticks correspond to isosurface values
                            # colorbar=dict(len=0.65, tickfont=dict(size=40, color='black', family='Times New Roman')),
                            showscale=False, showlegend=False, autocolorscale=False)

    fig.add_trace(trace_1, row=1, col=1)
    fig.add_trace(trace_2, row=1, col=2)
    fig.add_trace(trace_3, row=1, col=3)
    fig.add_trace(trace_4, row=2, col=1)
    fig.add_trace(trace_5, row=2, col=2)
    fig.add_trace(trace_6, row=2, col=3)
    # fig.add_isosurface(trace_1, row=1, col=1)
    # fig.add_isosurface(trace_2, row=1, col=2)
    # fig.add_isosurface(trace_3, row=1, col=3)
    # fig.add_isosurface(trace_4, row=2, col=1)
    # fig.add_isosurface(trace_5, row=2, col=2)
    # fig.add_isosurface(trace_6, row=2, col=3)

    # fig.update_xaxes(title_standoff=100)
    # sc_dic = dict(
    #     xaxis=dict(tickvals=[-50, -25, 0, 25, 50], tickfont=dict(size=17, family='Times New Roman'),
    #                title=dict(font=dict(size=38, family='Times New Roman'))),
    #     # backgroundcolor = 'white', gridcolor="silver",
    #     yaxis=dict(tickvals=[-25, 0, 25, 50], tickfont=dict(size=17, family='Times New Roman'),
    #                title=dict(font=dict(size=38, family='Times New Roman'))),
    #     # backgroundcolor = 'white', gridcolor="silver",
    #     zaxis=dict(tickvals=[-50, -25, 0, 25, 50], tickfont=dict(size=17, family='Times New Roman'),
    #                title=dict(font=dict(size=38, family='Times New Roman'))),
    #     # backgroundcolor = 'white', gridcolor="silver",
    #     xaxis_title='<i>x</i>', yaxis_title='<i>y</i>', zaxis_title='<i>z</i>'
    # )
    c = 1.4
    scene_camera_eye = dict(x=1.5 * c, y=1.3 * c, z=1.2 * c)
    # scene_camera_eye = dict(x=0.3 * c, y=0.5 * c, z=0.2 * c)

    x_dict = y_dict = z_dict = dict(backgroundcolor='white', gridcolor="silver",
                                    # tickvals=[0, 20],
                                    # tickfont=dict(size=17, family='Times New Roman'),
                                    # title=dict(font=dict(size=38, family='Times New Roman'))
                                    )
    sc_dic = dict(xaxis=x_dict, yaxis=y_dict, zaxis=z_dict, camera=dict(eye=scene_camera_eye),
        xaxis_showticklabels = False, yaxis_showticklabels = False, zaxis_showticklabels = False,
                  # xaxis_title='<i>x</i>', yaxis_title='<i>y</i>', zaxis_title='<i>z</i>',
                  )

    fig.update_layout(scene=sc_dic, scene1=sc_dic, scene2=sc_dic, scene3=sc_dic, scene4=sc_dic, scene5=sc_dic,
                      scene6=sc_dic)
    # fig.update_layout(margin=dict(t=0, l=0, b=0),  # tight layout
    #                   scene_camera_eye=dict(x=1.3 * c, y=1.1 * c, z=0.9 * c))

    # coloraxis_colorbar_dic = dict(yanchor="top", y=1, x=0)
    # coloraxis_colorbar_dic = dict(orientation='h')
    # cb_dic = dict(colorbar=coloraxis_colorbar_dic)
    #
    # fig.update_layout(coloraxis=cb_dic, coloraxis1=cb_dic, coloraxis2=cb_dic, coloraxis3=cb_dic, coloraxis4=cb_dic,
    #                   coloraxis5=cb_dic, coloraxis6=cb_dic)

    # tight layout
    # margin=dict(t=0, l=0, b=0),

    # fig.update_layout(scene=dict(xaxis_title='<i>x</i>', yaxis_title='<i>y</i>', zaxis_title='<i>z</i>'))
    # fig.update_xaxes(title_standoff=100)

    # fig.update_layout(
    #     coloraxis_colorbar=dict(len=0.65, tickfont=dict(size=40, color='black', family='Times New Roman'),
    #                             orientation='h', title="density", tickvals=[0.3, 0.5], ticktext=["red", "blue"]))

    # fig.update_layout(coloraxis_colorbar=dict(
    #   title="Species",
    #   tickvals=[0.3, 0.5],
    #   ticktext=["red", "blue"],
    #   len=0.6,
    # ))
    # fig.show()
    # fig.write_image(fig_save_path + '3DGS{}_{}_{}.png'.format('u' if ch == 0 else 'v', method, num))

    fig.write_image(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item))


def plotR2C3_bs(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # (3001, 100* 100, 2)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # if args.dataset_name == 'cf':
    #     face_path = os.path.join('{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
    #                                                                     args.dataset_name, args.all_time_steps,
    #                                                                     args.mesh_node, item))
    #     face = np.load(face_path)
    #     face = torch.from_numpy(face).float()

    # 不指定三角形，根据坐标按照Delaunay 三角剖分确定三角形
    # triangulation = mtri.Triangulation(x_star, y_star)

    # 指定三角形，comsol生成的mesh elements
    mesh_path = os.path.join(
        '{}/{}d_{}_mesh_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                               args.all_time_steps, args.mesh_node, item))
    mesh = np.load(mesh_path, allow_pickle=True)
    triangulation = mtri.Triangulation(x_star, y_star, mesh.T)

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0:1, ...]
    u_pred = output[num, 0:1, ...]
    v_star = truth_data[num, 1:2, ...]
    v_pred = output[num, 1:2, ...]
    p_star = truth_data[num, 2:3, ...]
    p_pred = output[num, 2:3, ...]

    # streamplot

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    list = [u_star, v_star]
    u_min, u_max = get_lim(torch.cat(list, dim=0))

    # 未加噪音数据前的真实模拟数据 U
    # subplot_ax_quiver
    _ = subplot_ax_quiver(ax[0, 0], triangulation, x_star, y_star, u_star, v_star, 'uv(Groud Truth.)' + str(num), fig,
                          args,
                          u_min, u_max, label_title='flow velocity')

    list = [u_pred, v_pred]
    u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 U
    _ = subplot_ax_quiver(ax[0, 1], triangulation, x_star, y_star, u_pred, v_pred, 'uv(Pred.)' + str(num), fig, args,
                          u_min,
                          u_max, label_title='flow velocity')

    # list = [torch.abs(u_pred - u_star), torch.abs(v_pred - v_star)]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))

    # 误差数据 U
    im = subplot_ax_quiver(ax[0, 2], triangulation, x_star, y_star, torch.abs(u_pred - u_star),
                           torch.abs(v_pred - v_star),
                           'uv(Abs Err.)' + str(num), fig, args, u_min, u_max, label_title='flow velocity')

    # list = [u_star, v_star, u_pred, v_pred, torch.abs(u_pred - u_star), torch.abs(v_pred - v_star)]
    # c_min, c_max = get_lim(torch.cat(list, dim=0))
    # rainbow
    _createColorBarVertical(fig, im, ax[0, 2], [ax[0, 0], ax[0, 1], ax[0, 2]], u_min, u_max, label_format="{:02.2f}",
                            cmap='viridis', label_title='velocity')

    p_list = [p_star]
    p_min, p_max = get_lim(torch.cat(p_list, dim=0))
    # 未加噪音数据前的真实模拟数据 U
    _ = subplot_ax_tri(ax[1, 0], triangulation, x_star, y_star, p_star, 'tem(Groud Truth.)' + str(num), fig, args,
                       p_min,
                       p_max, label_title='temperature')

    # p_list = [p_pred]
    # p_min, p_max = get_lim(torch.cat(p_list, dim=0))
    # 模型预测数据 U
    _ = subplot_ax_tri(ax[1, 1], triangulation, x_star, y_star, p_pred, 'tem(Pred.)' + str(num), fig, args, p_min,
                       p_max,
                       label_title='temperature')

    # p_list = [torch.abs(p_pred - p_star)]
    # p_min, p_max = get_lim(torch.cat(p_list, dim=0))

    # 误差数据 U
    im = subplot_ax_tri(ax[1, 2], triangulation, x_star, y_star, torch.abs(p_pred - p_star), 'tem(Abs Err.)' + str(num),
                        fig,
                        args, p_min, p_max, label_title='temperature')

    # list = [p_pred, p_star, torch.abs(p_pred - p_star)]
    # c_min, c_max = get_lim(torch.cat(list, dim=0))
    # rainbow viridis
    _createColorBarVertical(fig, im, ax[1, 2], [ax[1, 0], ax[1, 1], ax[1, 2]], p_min, p_max, label_format="{:02.2f}",
                            cmap='rainbow', label_title='temperature')

    # 存储图片
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    plt.close('all')


def plotR1C3_bs_D2(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # (3001, 100* 100, 2)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # if args.dataset_name == 'cf':
    #     face_path = os.path.join('{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
    #                                                                     args.dataset_name, args.all_time_steps,
    #                                                                     args.mesh_node, item))
    #     face = np.load(face_path)
    #     face = torch.from_numpy(face).float()

    # 不指定三角形，根据坐标按照Delaunay 三角剖分确定三角形
    # triangulation = mtri.Triangulation(x_star, y_star)

    # 指定三角形，comsol生成的mesh elements
    mesh_path = os.path.join(
        '{}/{}d_{}_mesh_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                               args.all_time_steps, args.mesh_node, item))
    mesh = np.load(mesh_path, allow_pickle=True)
    triangulation = mtri.Triangulation(x_star, y_star, mesh.T)

    # Padding x,y axis due to periodic boundary condition
    u_star = truth_data[num, 0:1, ...]
    u_pred = output[num, 0:1, ...]
    v_star = truth_data[num, 1:2, ...]
    v_pred = output[num, 1:2, ...]

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    list = [u_star, v_star]
    u_min, u_max = get_lim(torch.cat(list, dim=0))

    # 未加噪音数据前的真实模拟数据 U
    subplot_ax_quiver(ax[0], triangulation, x_star, y_star, u_star, v_star, 'uv(Groud Truth.)' + str(num), fig, args,
                      u_min, u_max, label_title='flow velocity')

    # list = [u_pred, v_pred]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))
    # 模型预测数据 U
    subplot_ax_quiver(ax[1], triangulation, x_star, y_star, u_pred, v_pred, 'uv(Pred.)' + str(num), fig, args, u_min,
                      u_max, label_title='flow velocity')

    # list = [torch.abs(u_pred - u_star), torch.abs(v_pred - v_star)]
    # u_min, u_max = get_lim(torch.cat(list, dim=0))

    # 误差数据 U
    subplot_ax_quiver(ax[2], triangulation, x_star, y_star, torch.abs(u_pred - u_star), torch.abs(v_pred - v_star),
                      'uv(Abs Err.)' + str(num), fig, args, u_min, u_max, label_title='flow velocity')

    # 存储图片
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    plt.close('all')


def plotR1C3_bs_D1(feature_dim, position, output, truth_data, num, fig_save_path, item, args):
    x_star, y_star = position[num, :, 0], position[num, :, 1]

    # (3001, 100* 100, 2)
    output = output[..., 0:feature_dim]
    truth_data = truth_data[..., 0:feature_dim]
    # [1000,1,1000,2]
    output = output.permute(0, 3, 1, 2)  # [1000,2,1,1000]
    truth_data = truth_data.permute(0, 3, 1, 2)

    # if args.dataset_name == 'cf':
    #     face_path = os.path.join('{}/{}d_{}_face_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension,
    #                                                                     args.dataset_name, args.all_time_steps,
    #                                                                     args.mesh_node, item))
    #     face = np.load(face_path)
    #     face = torch.from_numpy(face).float()

    # 不指定三角形，根据坐标按照Delaunay 三角剖分确定三角形
    # triangulation = mtri.Triangulation(x_star, y_star)

    # 指定三角形，comsol生成的mesh elements
    mesh_path = os.path.join(
        '{}/{}d_{}_mesh_t{}_n{}_{}.npy'.format(args.data_save_path, args.dimension, args.dataset_name,
                                               args.all_time_steps, args.mesh_node, item))
    mesh = np.load(mesh_path, allow_pickle=True)
    triangulation = mtri.Triangulation(x_star, y_star, mesh.T)

    # Padding x,y axis due to periodic boundary condition
    p_star = truth_data[num, 0:1, ...]
    p_pred = output[num, 0:1, ...]

    # if is_3D:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 7), subplot_kw=dict(projection="3d"))
    # else:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    p_list = [p_star]
    p_min, p_max = get_lim(torch.cat(p_list, dim=0))
    # 未加噪音数据前的真实模拟数据 U
    subplot_ax_tri(ax[0], triangulation, x_star, y_star, p_star, 'tem(Groud Truth.)' + str(num), fig, args, p_min,
                   p_max, label_title='temperature')

    # p_list = [p_pred]
    # p_min, p_max = get_lim(torch.cat(p_list, dim=0))
    # 模型预测数据 U
    subplot_ax_tri(ax[1], triangulation, x_star, y_star, p_pred, 'tem(Pred.)' + str(num), fig, args, p_min, p_max,
                   label_title='temperature')

    # p_list = [torch.abs(p_pred - p_star)]
    # p_min, p_max = get_lim(torch.cat(p_list, dim=0))

    # 误差数据 U
    subplot_ax_tri(ax[2], triangulation, x_star, y_star, torch.abs(p_pred - p_star), 'tem(Abs Err.)' + str(num), fig,
                   args, p_min, p_max, label_title='temperature')

    # 存储图片
    plt.savefig(fig_save_path + '{}d_{}_uv_comparison_'.format(args.dimension, args.dataset_name) + str(num).zfill(
        5) + '_{}.png'.format(item), dpi=300)
    plt.close('all')
