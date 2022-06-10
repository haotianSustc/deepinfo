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


def step_unet(root_path, args):
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
    train_set = get_datasets(args, args.datasets_name, data_path=data_datasets_path, train=True, dataset_mode=1)
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
        capability_path = os.path.join(experiment_dataset_net_path, 'net_capability_' + args.task_name + '_x')
        plot_capability_path = os.path.join(plot_dataset_net_path, 'net_capability_' + args.task_name + '_x')
    else:
        capability_path = os.path.join(experiment_dataset_net_path, 'net_capability_' + args.task_name + '_x')
        plot_capability_path = os.path.join(plot_dataset_net_path, 'net_capability_' + args.task_name + '_x')
    make_dir(capability_path)
    make_dir(plot_capability_path)
    net = get_net(net_name=args.net_name, gpu_id=args.gpu_id, model_mode=args.net_mode, out_planes=args.out_planes,
                  if_pretrained=args.net_if_pretrained)
    if args.task_name.startswith('compressed'):
        net_path = os.path.join(experiment_dataset_net_path, 'net_compressed')
        load_net_path = os.path.join(net_path, 'model_' + str(args.net_compressed_id) + '.pth')
    elif args.net_name == "unet":
        load_net_path = os.path.join(experiment_dataset_net_path, 'net_' + str(args.task_name), str(args.net_id), "unet_medical.pth")
    else:
        if args.capability_if_multiepoch == 1:
            net_path = os.path.join(experiment_dataset_net_path, 'net_multiepoch')
            net_sub_path = os.path.join(net_path, str(args.net_id))
            load_net_path = os.path.join(net_sub_path, "net_" + str(args.capability_multiepoch_epoch_num - 1) + ".pth")
        else:
            net_path = os.path.join(experiment_dataset_net_path, 'net_' + str(args.task_name))
            net_sub_path = os.path.join(net_path, str(args.net_id))
            load_net_path = os.path.join(net_sub_path, "net_" + str(args.net_epoch - 1) + ".pth")
    net.load_state_dict(torch.load(load_net_path))
    net.eval()
    if args.net_name.endswith('_autoencoder'):
        net = net.encoder
    # all parameters of net is fixed and dropout will lose efficacy
    # upsample_method = nn.Upsample(size=args.image_size, mode='nearest')
    upsample_method = nn.Upsample(size=args.image_size, mode='bilinear')
    get_all_layer = False
    if args.capability_get_all_layer == 1:
        get_all_layer = True
    if args.capability_num_layer == 0:
        num_layer = get_num_layer(args.net_name, get_all_layer)
    else:
        num_layer = args.capability_num_layer

    for batch_id, (image, label) in enumerate(train_loader):
        if batch_id % 5 != 0:
            continue
        # if batch_id < args.capability_num_batch*args.batch_interval_num and batch_id > args.capability_num_batch and batch_id % args.batch_interval_num == 0:
        # if batch_id < args.capability_num_batch*args.batch_interval_num and batch_id % args.batch_interval_num == 0:
        if batch_id < args.capability_num_batch:
            image = image.cuda()
            # image = image - mean.expand(image.size()).cuda()
            # image = image / std.cuda()
            if args.net_name == "unet":
                y = 1 - net(image).argmax(1).float()
                torchvision.utils.save_image(y, os.path.join(plot_capability_path, f"output_{batch_id}.png"))
            print('------' + str(batch_id) + '------')
            for layer_id in range(args.capability_start_layer, args.capability_start_layer + num_layer):
                labmda_init = args.capability_lambda_init_x
                layer_name = get_layer_name(args.net_name, layer_id, get_all_layer)
                capability_data_path = os.path.join(capability_path, 'batch_' + str(batch_id))
                make_dir(capability_data_path)
                data_result_path = os.path.join(capability_data_path, layer_name)
                make_dir(data_result_path)
                plot_capability_data_path = os.path.join(plot_capability_path, 'batch_' + str(batch_id))
                make_dir(plot_capability_data_path)
                plot_result_path = os.path.join(plot_capability_data_path, layer_name)
                make_dir(plot_result_path)
                noise_layer = get_model_noise_layer(args.gpu_id, torch.zeros(1, 1, args.capability_sigma_size,
                                                                             args.capability_sigma_size).size(),
                                                    args.capability_sigma_init_decay, args.image_size)
                optimizer = torch.optim.SGD([{"params": noise_layer.parameters(), 'lr': args.capability_lr_x,
                                              'initial_lr': args.capability_lr_x}])

                unit_vector = torch.ones(args.capability_batch_size, 1, args.capability_sigma_size,
                                         args.capability_sigma_size).cuda()
                unit_noise = torch.randn(args.capability_batch_size, 3, args.image_size, args.image_size).cuda()
                noise_image, penalty = noise_layer(image, unit_vector, unit_noise)
                noise_feature = get_feature(noise_image, net, args.net_name, model_mode=args.net_mode,
                                            layer_id=layer_id, get_all_layer=get_all_layer)
                origin_feature = get_feature(image, net, args.net_name, model_mode=args.net_mode,
                                             layer_id=layer_id, get_all_layer=get_all_layer)
                origin_feature = origin_feature.detach()
                if args.datasets_name.endswith('_ratio') and args.task_name != 'visual' and layer_id <= 5:
                    noise_channel_vector = get_channel_vector(noise_feature).data.cpu()
                    origin_channel_vector = get_channel_vector(origin_feature).data.cpu()
                    origin_channel_vector = origin_channel_vector.view(1, -1, 1, 1).cuda()
                    noise_channel_vector = noise_channel_vector.view(1, -1, 1, 1).cuda()
                    noise_feature = relu_method(noise_feature * noise_channel_vector * origin_channel_vector)
                    origin_feature = relu_method(origin_feature * noise_channel_vector * origin_channel_vector)
                noise_feature = relu_method(noise_feature)
                origin_feature = relu_method(origin_feature)
                sigma_f = criterion(noise_feature, origin_feature).detach()

                train_feature_loss_list = []
                train_penalty_loss_list = []
                train_lambda_list = []
                train_feature_loss = AverageMeter()
                train_penalty_loss = AverageMeter()
                noise_layer.train()

                # if batch_id < args.capability_visual_sample_num*args.batch_interval_num and batch_id % args.batch_interval_num == 0:
                # if batch_id < args.capability_num_batch*args.batch_interval_num and batch_id % args.batch_interval_num == 0:
                if batch_id < args.capability_num_batch:
                    # visual_image = image * std.cuda()
                    # visual_image = visual_image + mean.expand(image.size()).cuda()
                    # visual_image = visual_image * 255
                    visual_image = image * 255
                    origin_image_visual = visual_image.data.cpu()
                    origin_image_visual = np.array(origin_image_visual[0])
                    origin_image_visual = origin_image_visual
                    origin_image_visual = origin_image_visual.transpose((1, 2, 0))
                    origin_image_visual = np.uint8(origin_image_visual)
                    origin_image_visual = Image.fromarray(origin_image_visual, 'RGB')
                    origin_image_visual.save(
                        os.path.join(plot_capability_path, 'origin_image_' + str(batch_id) + "_.png"))

                for epoch in range(0, args.capability_epoch):
                    # lambda_param = labmda_init * math.e ** (args.capability_lambda_change_ratio * epoch / args.capability_epoch)
                    lambda_param = labmda_init
                    # the parameter lambda is rising over epochs

                    train_feature_loss.reset()
                    train_penalty_loss.reset()

                    # train
                    params_data = optimizer.param_groups[0]['params'][0].data
                    sigma_data = params_data
                    sigma_data = sigma_data.data.cpu()
                    sigma_data = np.array(sigma_data)
                    sigma_path = os.path.join(data_result_path, 'sigma_' + str(epoch) + '.npy')
                    np.save(sigma_path, sigma_data)

                    # if batch_id < args.capability_visual_sample_num*args.batch_interval_num and batch_id % args.batch_interval_num == 0:
                    if batch_id < args.capability_num_batch and epoch % 10 == 9:
                        visual_data = params_data
                        visual_data = 2 * math.pi * math.e * visual_data * visual_data
                        visual_data = torch.log(visual_data)
                        visual_data = upsample_method(visual_data)
                        if layer_id == args.capability_start_layer + num_layer - 1 and epoch == args.capability_epoch - 1:
                            print(visual_data.min(), visual_data.max())
                        visual_data = visual_data[0][0].data.cpu()
                        visual_path = os.path.join(plot_result_path, 'norm_sigma_' + str(epoch) + '.png')
                        plot_feature(visual_data, visual_path)
                        visual_path = os.path.join(plot_result_path, 'sigma_' + str(epoch) + '.png')
                        plot_feature(visual_data, visual_path, norm=False)

                    unit_vector = torch.ones(args.capability_batch_size, 1, args.capability_sigma_size,
                                             args.capability_sigma_size).cuda()
                    unit_noise = torch.randn(args.capability_batch_size, 3, args.image_size, args.image_size).cuda()
                    noise_image, penalty = noise_layer(image, unit_vector, unit_noise)

                    noise_feature = get_feature(noise_image, net, args.net_name, model_mode=args.net_mode,
                                                layer_id=layer_id, get_all_layer=get_all_layer)
                    origin_feature = get_feature(image, net, args.net_name, model_mode=args.net_mode,
                                                 layer_id=layer_id, get_all_layer=get_all_layer)
                    origin_feature = origin_feature.detach()
                    if args.datasets_name.endswith('_ratio') and args.task_name != 'visual' and layer_id <= 5:
                        noise_channel_vector = get_channel_vector(noise_feature).data.cpu()
                        origin_channel_vector = get_channel_vector(origin_feature).data.cpu()
                        origin_channel_vector = origin_channel_vector.view(1, -1, 1, 1).cuda()
                        noise_channel_vector = noise_channel_vector.view(1, -1, 1, 1).cuda()
                        noise_feature = relu_method(noise_feature * noise_channel_vector * origin_channel_vector)
                        origin_feature = relu_method(origin_feature * noise_channel_vector * origin_channel_vector)
                    noise_feature = relu_method(noise_feature)
                    origin_feature = relu_method(origin_feature)
                    feature_loss = criterion(noise_feature, origin_feature)

                    penalty_loss = -penalty * lambda_param
                    feature_loss = feature_loss / sigma_f

                    loss = feature_loss + penalty_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch % 10 == 0:
                        print("Train: [" + str(epoch) + "/" + str(args.capability_epoch) + "]" + "\n"
                              + 'feature_loss: ' + str(float(feature_loss)) + "\n"
                              + 'penalty_loss: ' + str(
                            float(-penalty_loss / lambda_param - math.log(2 * (math.pi) * (math.e)))) + "\n"
                              )

                    train_feature_loss.update(feature_loss.data.cpu())
                    train_penalty_loss.update(penalty_loss.data.cpu())
                    train_feature_loss_list.append(train_feature_loss.avg)
                    train_penalty_loss_list.append(train_penalty_loss.avg)
                    train_lambda_list.append(lambda_param)

                    if epoch == args.capability_epoch - 1:
                        # np.save(sigma_path, sigma_data)
                        save_capability_path = os.path.join(data_result_path, "capability_" + str(epoch) + ".bin")
                        save_list_path = os.path.join(data_result_path, "capability_list_" + str(epoch) + ".bin")
                        if batch_id < args.capability_visual_sample_num:
                            plot_result(train_feature_loss_list, os.path.join(data_result_path, "capability_list.png"))
                        torch.save(noise_layer.state_dict(), save_capability_path)
                        torch.save(
                            [train_feature_loss_list, train_penalty_loss_list, train_lambda_list, float(sigma_f)],
                            save_list_path)
