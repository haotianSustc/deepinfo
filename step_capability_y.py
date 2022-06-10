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

def step_capability_y(root_path,args):

    experiment_path = os.path.join(root_path,'experiment')
    data_path = os.path.join(root_path,'data')
    plot_path = os.path.join(root_path,'plot')
    make_dir(plot_path)

    experiment_dataset_path = os.path.join(experiment_path,args.datasets_name)
    data_datasets_path = os.path.join(data_path,args.datasets_name)
    plot_dataset_path = os.path.join(plot_path,args.datasets_name)
    make_dir(plot_dataset_path)
    mean,std = get_mean_std(args.datasets_name)

    # Get data
    train_set = get_datasets(args, args.datasets_name,data_path=data_datasets_path,train=True,dataset_mode=1)
    # Just for CUB_ratio dataset. set dataset_mode to 1, and we get postive samples.
    if args.capability_if_shuffle == 1:
        shuffle=True
    else:
        shuffle=False
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=shuffle)
    # train one image and get one sigma.
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    relu_method = nn.ReLU()

    experiment_dataset_net_path = os.path.join(experiment_dataset_path,args.net_name)
    plot_dataset_net_path = os.path.join(plot_dataset_path,args.net_name)
    make_dir(plot_dataset_net_path)

    net_path = os.path.join(experiment_dataset_net_path,'net_'+args.task_name)
    net_sub_path = os.path.join(net_path,str(args.net_id))
    if args.capability_if_multiepoch == 1:
        capability_path = os.path.join(experiment_dataset_net_path,'net_capability_'+args.task_name+'_y_multiepoch_'+str(args.capability_multiepoch_epoch_num-1))
        plot_capability_path = os.path.join(plot_dataset_net_path,'net_capability_'+args.task_name+'_y_multiepoch_'+str(args.capability_multiepoch_epoch_num-1))
    else:
        capability_path = os.path.join(experiment_dataset_net_path,'net_capability_'+args.task_name+'_y'+f"_{args.decoder_num_block}")
        plot_capability_path = os.path.join(plot_dataset_net_path,'net_capability_'+args.task_name+'_y'+f"_{args.decoder_num_block}")
    make_dir(capability_path)
    make_dir(plot_capability_path)
    net = get_net(net_name=args.net_name, gpu_id=args.gpu_id, model_mode=args.net_mode, out_planes=args.out_planes,
                  if_pretrained=args.net_if_pretrained)
    if args.task_name.startswith('compressed'):
        net_path = os.path.join(experiment_dataset_net_path,'net_compressed')
        load_net_path = os.path.join(net_path,'model_'+str(args.net_compressed_id)+'.bin')
    else:
        net_path = os.path.join(experiment_dataset_net_path,'net_'+args.task_name)
        net_sub_path = os.path.join(net_path,str(args.net_id))
        if args.capability_if_multiepoch == 1:
            load_net_path = os.path.join(net_sub_path,"net_" + str(args.capability_multiepoch_epoch_num-1) + ".bin")
        else:
            load_net_path = os.path.join(net_sub_path,"net_" + str(args.net_epoch-1) + ".bin")
    net.load_state_dict(torch.load(load_net_path))
    net.eval()
    # all parameters of net is fixed and dropout will lose efficacy
    upsample_method = nn.Upsample(size=args.image_size,mode='nearest')
    get_all_layer = False
    if args.capability_get_all_layer == 1:
        get_all_layer = True
    num_layer = get_num_layer(args.net_name,get_all_layer)
    all_decoder = []

    for layer_id in range(1,num_layer+1):

        image_sample = torch.ones(1, 3, args.image_size, args.image_size).cuda()
        decoder_feature_sample = get_feature(image_sample, net, args.net_name,
                                             model_mode=args.net_mode, layer_id=layer_id)
        decoder_feature_sample_channel = decoder_feature_sample.size(1)
        decoder_feature_sample_size = decoder_feature_sample.size(2)
        if args.if_alexnet == 1:
            num_block_upsample = [2,3,4,4,4][layer_id-1]
        else:
            num_block_upsample = math.log((args.image_size/decoder_feature_sample_size),2)
        # upsample block can make the size of feature double.
        decoder = get_model_resnet_decoder(gpu_id=args.gpu_id,
                                           num_input_channels=decoder_feature_sample_channel,
                                           num_block=args.decoder_num_block, num_block_upsample=num_block_upsample,if_alexnet=args.if_alexnet)
        decoder_path = os.path.join(experiment_dataset_net_path,'decoder_'+args.task_name+f"_{args.decoder_num_block}")
        decoder_sub_path = os.path.join(decoder_path, str(layer_id))
        load_decoder_path = os.path.join(decoder_sub_path, "decoder_" + str(args.decoder_epoch - 1) + '.bin')
        decoder.load_state_dict(torch.load(load_decoder_path))
        decoder.eval()
        all_decoder.append(decoder)

    for batch_id,(image,label) in enumerate(train_loader):
        if batch_id % 50 != 0:
            continue
        # if batch_id < args.capability_num_batch*args.batch_interval_num and batch_id > args.capability_num_batch and batch_id % args.batch_interval_num == 0:
        # if batch_id < args.capability_num_batch*args.batch_interval_num and batch_id % args.batch_interval_num == 0:
        if batch_id < args.capability_num_batch:
        # if batch_id < args.capability_num_batch*args.batch_interval_num and batch_id > args.capability_num_batch and batch_id % args.batch_interval_num == 0:
            image = image.cuda()
            decoder_image = image
            image = image - mean.expand(image.size()).cuda()
            image = image / std.cuda()
            print('------'+str(batch_id)+'------')
            for layer_id in range(1,num_layer+1):
                decoder = all_decoder[int(layer_id)-1]
                labmda_init = args.capability_lambda_init_y
                layer_name = get_layer_name(args.net_name,layer_id,get_all_layer)
                capability_data_path = os.path.join(capability_path,'batch_'+str(batch_id))
                make_dir(capability_data_path)
                data_result_path = os.path.join(capability_data_path,layer_name)
                make_dir(data_result_path)
                plot_capability_data_path = os.path.join(plot_capability_path,'batch_'+str(batch_id))
                make_dir(plot_capability_data_path)
                plot_result_path = os.path.join(plot_capability_data_path,layer_name)
                make_dir(plot_result_path)
                noise_layer = get_model_noise_layer(args.gpu_id,torch.zeros(1,1,args.capability_sigma_size,args.capability_sigma_size).size(),
                                                    args.capability_sigma_init_decay,args.image_size)
                optimizer = torch.optim.SGD([{"params":noise_layer.parameters(),'lr':args.capability_lr_y,'initial_lr':args.capability_lr_y}])

                train_feature_loss_list = []
                train_feature_loss = AverageMeter()
                noise_layer.train()

                if batch_id < args.capability_visual_sample_num*args.batch_interval_num and batch_id % args.batch_interval_num == 0:
                    visual_image = image*std.cuda()
                    visual_image = visual_image + mean.expand(image.size()).cuda()
                    visual_image = visual_image*255
                    origin_image_visual = visual_image.data.cpu()
                    origin_image_visual = np.array(origin_image_visual[0])
                    origin_image_visual = origin_image_visual
                    origin_image_visual = origin_image_visual.transpose((1,2,0))
                    origin_image_visual = np.uint8(origin_image_visual)
                    origin_image_img = Image.fromarray(origin_image_visual, 'RGB')
                    origin_image_img.save(os.path.join(plot_capability_path,'origin_image_'+str(batch_id)+"_.png"))

                unit_vector = torch.ones(args.capability_batch_size,1,args.capability_sigma_size,args.capability_sigma_size).cuda()
                unit_noise = torch.randn(args.capability_batch_size,3,args.image_size,args.image_size).cuda()
                noise_image, penalty = noise_layer(image,unit_vector,unit_noise)
                noise_feature = get_feature(noise_image,net,args.net_name,model_mode=args.net_mode,
                                            layer_id=layer_id)
                origin_feature = get_feature(image,net,args.net_name,model_mode=args.net_mode,
                                             layer_id=layer_id)
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
                sigma_f = criterion(noise_feature,origin_feature).detach()

                for epoch in range(0,args.capability_epoch):
                    lambda_param = labmda_init * math.e ** (args.capability_lambda_change_ratio * epoch / args.capability_epoch)
                    # the parameter lambda is rising over epochs

                    train_feature_loss.reset()

                    # train
                    params_data = optimizer.param_groups[0]['params'][0].data
                    sigma_data = params_data
                    sigma_data = sigma_data.data.cpu()
                    sigma_data = np.array(sigma_data)
                    sigma_path = os.path.join(data_result_path,'sigma_'+str(epoch)+'.npy')
                    np.save(sigma_path,sigma_data)

                    unit_vector = torch.ones(args.capability_batch_size,1,args.capability_sigma_size,args.capability_sigma_size).cuda()
                    unit_noise = torch.randn(args.capability_batch_size,3,args.image_size,args.image_size).cuda()
                    noise_image, penalty = noise_layer(image,unit_vector,unit_noise)
                    noise_feature = get_feature(noise_image,net,args.net_name,model_mode=args.net_mode,
                                                layer_id=layer_id)
                    origin_feature = get_feature(image,net,args.net_name,model_mode=args.net_mode,
                                                 layer_id=layer_id)
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
                    feature_loss = criterion(noise_feature,origin_feature)

                    noise_decoder_feature = decoder(noise_feature)

                    image_var = noise_decoder_feature - decoder_image.detach()
                    image_data = np.array(image_var.data.cpu())
                    if batch_id < args.capability_visual_sample_num*args.batch_interval_num and batch_id % args.batch_interval_num == 0:
                        visual_data = image_var.data.cpu()
                        visual_data = 2*math.pi*math.e*visual_data*visual_data
                        visual_data = torch.log(visual_data)
                        visual_data = upsample_method(visual_data)
                        visual_data = visual_data[0][0].data.cpu()
                        visual_path = os.path.join(plot_result_path,'image_'+str(epoch)+'.png')
                        plot_feature(visual_data,visual_path)
                    image_path = os.path.join(data_result_path, 'image_' + str(epoch) + '.npy')
                    np.save(image_path, image_data)
                    image_var = torch.mean(torch.log(2*math.pi*math.e*image_var*image_var + 1e-6))

                    feature_loss = feature_loss/sigma_f
                    penalty_loss = -lambda_param*image_var
                    loss = feature_loss+penalty_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch % 2 == 0:
                        print("Train: [" + str(epoch) + "/" + str(args.capability_epoch) + "]" + "\n"
                              +'feature_loss: '+str(float(feature_loss))+"\n"
                              +'penalty_loss: '+str(float(-penalty_loss/lambda_param-math.log(2*(math.pi)*(math.e))))+"\n"
                              )

                    train_feature_loss.update(feature_loss.data.cpu())
                    train_feature_loss_list.append(train_feature_loss.avg)

                    if epoch == args.capability_epoch-1:
                        save_capability_path = os.path.join(data_result_path,"capability_" + str(epoch) + ".bin")
                        save_list_path = os.path.join(data_result_path,"capability_list_" + str(epoch) + ".bin")
                        torch.save(noise_layer.state_dict(),save_capability_path)
                        torch.save([train_feature_loss_list,float(sigma_f)],save_list_path)
