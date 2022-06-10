# get results from file saved in experiments.
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
import matplotlib.mlab as mlab

from tools.lib import *
from tools.get_input import *

def step_result(root_path,args,x_or_y):

    get_all_layer = False
    if args.capability_get_all_layer == 1:
        get_all_layer = True
    num_layer = get_num_layer(args.net_name,args.capability_get_all_layer)
    plot_path = os.path.join(root_path, 'plot')
    plot_dataset_path = os.path.join(plot_path,args.datasets_name)
    plot_dataset_net_path = os.path.join(plot_dataset_path,args.net_name)

    experiment_path = os.path.join(root_path, 'experiment')
    result_path = os.path.join(experiment_path,'result.csv')
    experiment_dataset_path = os.path.join(experiment_path, args.datasets_name)
    experiment_dataset_net_path = os.path.join(experiment_dataset_path, args.net_name)
    capability_path = os.path.join(experiment_dataset_net_path, 'net_capability_' + args.task_name + '_' + str(x_or_y))
    output_mean = np.zeros((args.capability_num_batch,num_layer))
    output_mean = output_mean
    if args.datasets_name.endswith('_ratio'):
        output_background_foreground = np.zeros((args.capability_num_batch,num_layer))

    for batch_id in range(0,args.capability_num_batch):
        if batch_id % 5 != 0:
            continue
        print('-------')
        for layer_id in range(args.capability_start_layer, args.capability_start_layer+num_layer):
            layer_name = get_layer_name(args.net_name,layer_id,get_all_layer)
            print(layer_name)
            capability_data_path = os.path.join(capability_path,'batch_'+str(batch_id))
            data_result_path = os.path.join(capability_data_path,layer_name)
            save_list_path = os.path.join(data_result_path,"capability_list_" + str(args.capability_epoch-1) + ".bin")
            capability_list = torch.load(save_list_path)
            feature_loss_list = capability_list[0]
            feature_capability = np.zeros((args.capability_epoch))
            for epoch_id in range(0,args.capability_epoch):
                feature_capability[epoch_id] = abs(float(feature_loss_list[epoch_id]) - args.result_distance_clip)
            # we choose the epoch where the least distance between feature_loss and DISTANCE_CLIP.
            best_epoch_id = np.argmin(feature_capability[:])

            for epoch_id in range(0,args.capability_epoch):
                if epoch_id == best_epoch_id:
                    print(epoch_id)
                    sigma_path = os.path.join(data_result_path,'sigma_'+str(epoch_id)+'.npy')
                    image_path = os.path.join(data_result_path,'image_'+str(epoch_id)+'.npy')
                    if x_or_y == 'x':
                        entropy_capability = np.load(sigma_path)[0][0]
                    elif x_or_y == 'y':
                        entropy_capability = np.load(image_path)[0][0]
                    else:
                        entropy_capability = np.zeros((1,1))
                        print('the suffix of task name must be "_x" or "_y" Error')
                    best_capability = np.log(entropy_capability*entropy_capability+1e-6)
                    # plot log(sigma*2) or log(image*2)
                    output_mean[batch_id][layer_id-args.capability_start_layer] = np.mean(best_capability)
                    # if args.datasets_name.endswith('_ratio') and args.task_name != 'visual':
                    #     best_capability = torch.from_numpy(best_capability).view(1,1,args.capability_sigma_size,args.capability_sigma_size)
                    #     small_bounding_capability = best_capability[:,:,int(args.capability_sigma_size/6):int(args.capability_sigma_size*5/6),
                    #                           int(args.capability_sigma_size/6):int(args.capability_sigma_size*5/6)]
                    #     background_size = args.capability_sigma_size*args.capability_sigma_size-small_bounding_capability.size(2)*small_bounding_capability.size(3)
                    #     background_entropy = torch.sum(best_capability)/background_size - torch.sum(small_bounding_capability)/background_size
                    #     foreground_entropy = torch.mean(small_bounding_capability)
                    #     output_background_foreground[batch_id][layer_id-1] = background_entropy - foreground_entropy



    for item in output_mean:
        print(item)
    output_mean = np.mean(output_mean, axis=0)
    print('----show results-----')
    print('datasets_name: '+str(args.datasets_name))
    print('net_name: '+str(args.net_name))
    print('task_name: '+str(args.task_name))
    print('x_or_y: '+str(x_or_y))
    print('result:')
    print(output_mean)

    output_new_mean = np.zeros(num_layer)
    output_new_mean[0] = output_mean[0]
    for i in range(1,num_layer):
        if output_mean[i] < output_new_mean[i-1]:
            output_new_mean[i] = output_new_mean[i-1]
        else:
            output_new_mean[i] = output_mean[i]
    output_mean = output_new_mean

    np.save(os.path.join(experiment_dataset_net_path,args.task_name+'_'+str(x_or_y)+'_entropy'),output_mean)
    plot_mean_path = os.path.join(experiment_dataset_net_path,args.task_name+'_'+str(x_or_y)+'_entropy.png')
    plot_result(output_mean,plot_mean_path)
    output_mean_result = [str(args.datasets_name),str(args.net_name),str(args.task_name),str(x_or_y),'mean']
    for item in output_mean:
        output_mean_result.append(float(item))
    print(output_mean_result)
    save_csv_result(result_path,output_mean_result)
    if args.datasets_name.endswith('_ratio') and args.task_name != 'visual':
        output_background_foreground = np.mean(output_background_foreground,axis=0)
        print(output_background_foreground)
        np.save(os.path.join(experiment_dataset_net_path,args.task_name+'_'+str(x_or_y) + '_background_foreground'),output_background_foreground)
        plot_background_foreground_path = os.path.join(experiment_dataset_net_path,args.task_name+'_'+str(x_or_y)+'_background_foreground.png')
        plot_result(output_background_foreground,plot_background_foreground_path)
        output_background_foreground_result = [str(args.datasets_name),str(args.net_name),str(args.task_name),str(x_or_y),'background_foreground']
        for item in output_background_foreground:
            output_background_foreground_result.append(float(item))
        print(output_background_foreground_result)
        save_csv_result(result_path,output_background_foreground_result)






