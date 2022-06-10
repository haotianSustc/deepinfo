# the main experiments
import os
import sys
import scipy.io as scio

import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from dataset import *
from tools.lib import *
from tools.get_input import *
from model.noise_layer.get_model_noise_layer import get_model_noise_layer
from model.resnet_decoder.get_model_resnet_decoder import get_model_resnet_decoder


def step_capability_jac(root_path, args):
    # hyperparameters
    experiment_path = os.path.join(root_path, 'experiment')
    data_path = os.path.join(root_path, 'data')
    plot_path = os.path.join(root_path, 'plot')
    make_dir(plot_path)

    experiment_dataset_path = os.path.join(experiment_path, args.datasets_name)
    data_datasets_path = os.path.join(data_path, args.datasets_name)
    plot_dataset_path = os.path.join(plot_path, args.datasets_name)
    make_dir(plot_dataset_path)
    mean, std = get_mean_std(args.datasets_name)

    # Get data
    train_set = get_datasets(args.datasets_name, data_path=data_datasets_path, train=False, dataset_mode=1)
    # set dataset_mode to 1, and we get postive samples.
    if args.capability_if_shuffle == 1:
        shuffle = True
    else:
        shuffle = False
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=shuffle)
    # train one image and get one sigma.
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    relu_method = nn.ReLU()

    experiment_dataset_net_path = os.path.join(experiment_dataset_path, args.net_name)
    plot_dataset_net_path = os.path.join(plot_dataset_path, args.net_name)
    make_dir(plot_dataset_net_path)

    if args.capability_if_multiepoch == 1:
        capability_path = os.path.join(experiment_dataset_net_path, 'net_capability_' + args.task_name + '_jac')
        plot_capability_path = os.path.join(plot_dataset_net_path, 'net_capability_' + args.task_name + '_jac')
    else:
        capability_path = os.path.join(experiment_dataset_net_path, 'net_capability_' + args.task_name + '_jac')
        plot_capability_path = os.path.join(plot_dataset_net_path, 'net_capability_' + args.task_name + '_jac')
    make_dir(capability_path)
    make_dir(plot_capability_path)
    net = get_net(net_name=args.net_name, gpu_id=args.gpu_id, model_mode=args.net_mode, out_planes=args.out_planes,
                  if_pretrained=args.net_if_pretrained)
    if args.task_name.startswith('compressed'):
        net_path = os.path.join(experiment_dataset_net_path, 'net_compressed')
        load_net_path = os.path.join(net_path, 'model_' + str(args.net_compressed_id) + '.bin')
    else:
        if args.capability_if_multiepoch == 1:
            net_path = os.path.join(experiment_dataset_net_path, 'net_multiepoch')
            net_sub_path = os.path.join(net_path, str(args.net_id))
            load_net_path = os.path.join(net_sub_path, "net_" + str(args.capability_multiepoch_epoch_num - 1) + ".bin")
        else:
            net_path = os.path.join(experiment_dataset_net_path, 'net_' + str(args.task_name))
            net_sub_path = os.path.join(net_path, str(args.net_id))
            load_net_path = os.path.join(net_sub_path, "net_" + str(args.net_epoch - 1) + ".bin")
    net.load_state_dict(torch.load(load_net_path))
    net.eval()
    if args.net_name.endswith('_autoencoder'):
        net = net.encoder

    get_all_layer = False
    if args.capability_get_all_layer == 1:
        get_all_layer = True
    upsample_method = nn.Upsample(size=args.image_size,mode='nearest')
    downsample_method = nn.AdaptiveAvgPool2d((args.capability_sigma_size, args.capability_sigma_size))
    if args.capability_num_layer == 0:
        num_layer = get_num_layer(args.net_name, get_all_layer)
    else:
        num_layer = args.capability_num_layer

    for batch_id, (image, label) in enumerate(train_loader):
        if batch_id % 50 != 0:
            continue
        # if batch_id < args.capability_num_batch*args.batch_interval_num and batch_id > args.capability_num_batch and batch_id % args.batch_interval_num == 0:
        # if batch_id < args.capability_num_batch*args.batch_interval_num and batch_id % args.batch_interval_num == 0:
        if batch_id < args.capability_num_batch:
            image = image.cuda()
            image = image - mean.expand(image.size()).cuda()
            image = image / std.cuda()
            print('------' + str(batch_id) + '------')
            for layer_id in range(args.capability_start_layer, args.capability_start_layer + num_layer):
                layer_name = get_layer_name(args.net_name, layer_id, get_all_layer)
                capability_data_path = os.path.join(capability_path, 'batch_' + str(batch_id))
                make_dir(capability_data_path)
                data_result_path = os.path.join(capability_data_path, layer_name)
                make_dir(data_result_path)
                plot_capability_data_path = os.path.join(plot_capability_path, 'batch_' + str(batch_id))
                make_dir(plot_capability_data_path)
                plot_result_path = os.path.join(plot_capability_data_path, layer_name)
                make_dir(plot_result_path)

                image.requires_grad_()
                origin_feature = get_feature(image, net, args.net_name, model_mode=args.net_mode,
                                             layer_id=layer_id, get_all_layer=get_all_layer)

                origin_feature = relu_method(origin_feature)

                feature_sum = origin_feature.sum()
                feature_sum.backward()

                jac_grad = image.grad

                jac_grad = upsample_method(downsample_method(jac_grad.mean(1, keepdim=True))).detach()

                visual_data = jac_grad[0][0].data.abs().cpu()
                visual_path = os.path.join(plot_result_path, 'jacobian.png')
                plot_feature(visual_data, visual_path)
                torch.save(jac_grad, os.path.join(data_result_path, "jacobian.bin"))
