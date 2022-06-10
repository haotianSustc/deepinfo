# Train the origin net.
import os
import sys
import scipy.io as scio

import torch
import torchvision
from torch import nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from dataset import *
from tools.lib import *
from tools.get_input import *

def step_net(root_path,args):

    experiment_path = os.path.join(root_path, 'experiment')
    make_dir(experiment_path)
    data_path = os.path.join(root_path, 'data')
    make_dir(data_path)
    experiment_dataset_path = os.path.join(experiment_path,args.datasets_name)
    make_dir(experiment_dataset_path)
    data_dataset_path = os.path.join(data_path, args.datasets_name)
    make_dir(data_dataset_path)
    mean,std = get_mean_std(args.datasets_name)

    # Get data
    training_set = get_datasets(args,args.datasets_name,data_path=data_dataset_path,train=True,dataset_mode=0)
    val_set = get_datasets(args,args.datasets_name,data_path=data_dataset_path,train=False,dataset_mode=0)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size = args.net_batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = args.net_batch_size, shuffle = False)

    experiment_dataset_net_path = os.path.join(experiment_dataset_path,args.net_name)
    make_dir(experiment_dataset_net_path)
    net_path = os.path.join(experiment_dataset_net_path,'net_'+args.task_name)
    make_dir(net_path)
    net_sub_path = os.path.join(net_path, str(args.net_id))
    make_dir(net_sub_path)

    # Get net.
    net = get_net(net_name=args.net_name,gpu_id=args.gpu_id,model_mode=args.net_mode,out_planes=args.out_planes,if_pretrained=args.net_if_pretrained)
    print(net)
    if args.datasets_name == 'CUB' and args.net_name == 'alexnet':
        seed_net_path = os.path.join(experiment_dataset_net_path,'seed_net.bin')
        if os.path.exists(seed_net_path) and args.net_if_pretrained==0:
            net.load_state_dict(torch.load(seed_net_path))
        else:
            torch.save(net.state_dict(),seed_net_path)
    optimizer = torch.optim.SGD([{"params":net.parameters(),'lr':args.net_lr,'initial_lr':args.net_lr}],weight_decay=args.net_weight_decay,momentum=args.net_momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=args.net_milestones,gamma=args.net_gama,last_epoch=-1)
    if args.datasets_name == 'cifar10' or args.datasets_name.startswith('VOC') or args.datasets_name == 'CUB':
        if not args.net_name.endswith('_autoencoder'):
            criterion = torch.nn.CrossEntropyLoss(size_average=True).cuda()
        else:
            criterion = torch.nn.BCELoss(size_average=True).cuda()
            sigmoid_method = torch.nn.Sigmoid()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(size_average=True).cuda()

    val_loss_list = []
    train_loss_list = []
    val_error_list = []
    train_error_list = []
    val_loss = AverageMeter()
    train_loss = AverageMeter()
    val_error = AverageMeter()
    train_error = AverageMeter()

    for epoch in range(0,args.net_epoch):

        net.train()
        val_loss.reset()
        train_loss.reset()
        val_error.reset()
        train_error.reset()

        for i,(image,label) in enumerate(train_loader):
            image = image - mean.expand(image.size()).float()
            image = image/std.float()
            image_size = image.size()
            batch_use = image_size[0]
            image = image.cuda()
            if args.datasets_name == 'cifar10' or args.datasets_name.startswith('VOC') or args.datasets_name == 'CUB':
                if not args.net_name.endswith('_autoencoder'):
                    label = label[0:int(batch_use)]
                    label = label.cuda()
                    outputs = net(image)
                    error = get_multilabel_error(outputs, label.long())
                    loss = criterion(outputs, label.long())
                else:
                    outputs = net(image)
                    image = image.detach()
                    image = image*std.cuda()
                    image = image + mean.expand(image.size()).cuda()
                    outputs = sigmoid_method(outputs)
                    reconstruction_loss = criterion(outputs,image)
                    loss = reconstruction_loss
                    error = 1
            else:
                label = label.view(-1, 1).float().cuda()
                outputs = net(image)
                error = get_binary_error(outputs,label.long(),binary_threshold=0.5)
                loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.data.cpu(),n=batch_use)
            train_error.update(error,n=batch_use)
            if i % 10 == 0:
                print("Train: [" + str(i) + "/" + str(len(train_loader)) + "] of epoch " + str(epoch) + "\n"
                      + "loss: " + str(float(loss)) +"\n"
                      + "error: " + str(float(error)) +"\n"
                      )
        train_loss_list.append(train_loss.avg)
        train_error_list.append(train_error.avg)
        scheduler.step()

        net.eval()

        with torch.no_grad():

            for i,(image,label) in enumerate(val_loader):
                image = image - mean.expand(image.size()).float()
                image = image/std.float()
                image_size = image.size()
                batch_use = image_size[0]
                image = image.cuda()
                if args.datasets_name == 'cifar10' or args.datasets_name.startswith('VOC') or args.datasets_name == 'CUB':
                    if not args.net_name.endswith('_autoencoder'):
                        label = label[0:int(batch_use)]
                        label = label.cuda()
                        outputs = net(image)
                        error = get_multilabel_error(outputs, label.long())
                        loss = criterion(outputs, label.long())
                    else:
                        outputs = net(image)
                        image = image.detach()
                        image = image*std.cuda()
                        image = image + mean.expand(image.size()).cuda()
                        outputs = sigmoid_method(outputs)
                        reconstruction_loss = criterion(outputs,image)
                        loss = reconstruction_loss
                        error = 1
                else:
                    label = label.view(-1,1).float().cuda()
                    outputs = net(image)
                    error = get_binary_error(outputs,label.long(),binary_threshold=0.5)
                    loss = criterion(outputs,label)
                val_loss.update(loss.data.cpu(),n=batch_use)
                val_error.update(error,n=batch_use)
                if i % 10 == 0:
                    print("Val: [" + str(i) + "/" + str(len(val_loader)) + "] of epoch " + str(epoch) + "\n"
                              + "loss: " + str(float(loss)) +"\n"
                              + "error: " + str(float(error)) +"\n"
                              )
            val_loss_list.append(val_loss.avg)
            val_error_list.append(val_error.avg)

        plot_result_path = os.path.join(net_sub_path,args.task_name+'_'+str(args.net_id)+".png")
        plot_loss_error(train_loss_list, val_loss_list, train_error_list, val_error_list, plot_result_path)
        save_net_path = os.path.join(net_sub_path, "net_" + str(epoch) + ".bin")
        save_list_path = os.path.join(net_sub_path, "net_list_" + str(epoch) + ".bin")
        torch.save(net.state_dict(), save_net_path)
        torch.save([train_loss_list, train_error_list, val_loss_list, val_error_list], save_list_path)

    final_error = val_error_list[len(val_error_list) - 1]
    result_path = os.path.join(net_path, 'result.csv')
    result_item = [str(args.net_id), str(final_error)]
    save_csv_result(result_path,result_item)
