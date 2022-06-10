# from origin net get decoder.
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
from model.resnet_decoder.get_model_resnet_decoder import get_model_resnet_decoder
from tools.lib import *
from tools.get_input import *

def step_decoder(root_path,args):

    experiment_path = os.path.join(root_path, 'experiment')
    data_path = os.path.join(root_path,'data')
    experiment_dataset_path = os.path.join(experiment_path,args.datasets_name)
    data_dataset_path = os.path.join(data_path,args.datasets_name)
    mean,std = get_mean_std(args.datasets_name)

    experiment_dataset_net_path = os.path.join(experiment_dataset_path,args.net_name)
    net_path = os.path.join(experiment_dataset_net_path,'net_'+args.task_name)
    net_sub_path = os.path.join(net_path,str(args.net_id))
    decoder_path = os.path.join(experiment_dataset_net_path,'decoder_'+args.task_name+f"_{args.decoder_num_block}")
    make_dir(decoder_path)

    # Get net.
    net = get_net(net_name=args.net_name, gpu_id=args.gpu_id, model_mode=args.net_mode, out_planes=args.out_planes,
                  if_pretrained=args.net_if_pretrained)
    criterion= torch.nn.MSELoss(size_average=True)
    relu_method = nn.ReLU()
    if args.task_name.startswith('compressed'):
        net_path = os.path.join(experiment_dataset_net_path,'net_compressed')
        load_net_path = os.path.join(net_path,'model_'+str(args.net_compressed_id)+'.bin')
    else:
        net_path = os.path.join(experiment_dataset_net_path,'net_'+args.task_name)
        net_sub_path = os.path.join(net_path,str(args.net_id))
        load_net_path = os.path.join(net_sub_path,"net_" + str(args.net_epoch-1) + ".bin")
    net.load_state_dict(torch.load(load_net_path))
    net.eval()

    training_set = get_datasets(args.datasets_name,data_path=data_dataset_path,train=True,dataset_mode=0)
    val_set = get_datasets(args.datasets_name,data_path=data_dataset_path,train=False,dataset_mode=0)
    train_loader = torch.utils.data.DataLoader(training_set,batch_size=args.decoder_batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=args.decoder_batch_size,shuffle=False)
    # print(net)


    get_all_layer = False
    if args.capability_get_all_layer == 1:
        get_all_layer = True
    num_layer = get_num_layer(args.net_name,get_all_layer)

    image_sample = torch.ones(1, 3, args.image_size, args.image_size).cuda()
    out_sample = net(image_sample)

    for layer_id in range(1,num_layer+1):

        layer_name = get_layer_name(args.net_name,layer_id,get_all_layer)
        image_sample = torch.ones(1,3,args.image_size,args.image_size).cuda()
        decoder_feature_sample = get_feature(image_sample,net,args.net_name,model_mode=args.net_mode,layer_id=layer_id)
        decoder_feature_sample_channel = decoder_feature_sample.size(1)
        decoder_feature_sample_size = decoder_feature_sample.size(2)
        if args.if_alexnet == 1:
            num_block_upsample = [2,3,4,4,4][layer_id-1]
        else:
            num_block_upsample = math.log((args.image_size/decoder_feature_sample_size),2)
        # upsample block can make the size of feature double.

        decoder = get_model_resnet_decoder(gpu_id=args.gpu_id,num_input_channels=decoder_feature_sample_channel,num_block=args.decoder_num_block,num_block_upsample=num_block_upsample,if_alexnet=args.if_alexnet)
        optimizer = torch.optim.Adam(decoder.parameters(),lr=args.decoder_lr)
        decoder_sub_path = os.path.join(decoder_path,str(layer_id))
        make_dir(decoder_sub_path)

        val_loss_list = []
        train_loss_list = []
        val_loss = AverageMeter()
        train_loss = AverageMeter()

        for epoch in range(0,args.decoder_epoch):

            visual_path = os.path.join(decoder_sub_path,str(epoch))
            make_dir(visual_path)

            for i,(image,label) in enumerate(train_loader):
                image = image-mean.expand(image.size())
                image = image/std
                image = image.cuda()
                batch_use = image.size(0)
                decoder_feature = get_feature(image,net,args.net_name,model_mode=args.net_mode,layer_id=layer_id)
                image = image * std.cuda()
                image = image + mean.expand(image.size()).cuda()
                decoder_image = image
                decoder_feature = decoder_feature.detach()
                decoder_out = decoder(decoder_feature)
                loss = criterion(decoder_out,decoder_image)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.update(loss.data.cpu(),n=batch_use)
                if i % 10 == 0:
                    print("Train: [" + str(i) + "/" + str(len(train_loader)) + "] of epoch " + str(epoch) + "\n"
                          + "loss: " + str(float(loss)) +"\n"
                          )
            train_loss_list.append(train_loss.avg)

            # validate
            net.eval()
            with torch.no_grad():
                for i,(image,label) in enumerate(val_loader):
                    if i < args.capability_visual_sample_num:
                        image = image-mean.expand(image.size())
                        image = image/std
                        image = image.cuda()
                        batch_use = image.size(0)
                        decoder_feature = get_feature(image,net,args.net_name,model_mode=args.net_mode,layer_id=layer_id)
                        decoder_feature = relu_method(decoder_feature)
                        image = image * std.cuda()
                        image = image + mean.expand(image.size()).cuda()
                        decoder_image = image
                        decoder_feature = decoder_feature.detach()
                        decoder_out = decoder(decoder_feature)
                        loss = criterion(decoder_out,decoder_image)
                        val_loss.update(loss.data.cpu(),n=batch_use)
                        if i % 10 == 0:
                            print("Val: [" + str(i) + "/" + str(len(val_loader)) + "] of epoch " + str(epoch) + "\n"
                                      + "loss: " + str(float(loss)) +"\n"
                                      )
                        image = image*255
                        origin_image_visual = image.data.cpu()
                        origin_image_visual = np.array(origin_image_visual[0])
                        origin_image_visual = origin_image_visual
                        origin_image_visual = origin_image_visual.transpose((1,2,0))
                        origin_image_visual = np.uint8(origin_image_visual)
                        origin_image_img = Image.fromarray(origin_image_visual, 'RGB')
                        origin_image_img.save(os.path.join(visual_path,'origin_image_'+str(i)+"_.png"))
                        decoder_out = decoder(decoder_feature)
                        decoder_out = decoder_out*255
                        decoder_out_visual = decoder_out.data.cpu()
                        decoder_out_visual = np.array(decoder_out_visual[0])
                        decoder_out_visual = decoder_out_visual.transpose((1,2,0))
                        decoder_out_visual = np.uint8(decoder_out_visual)
                        decoder_out_img = Image.fromarray(decoder_out_visual, 'RGB')
                        decoder_out_img.save(os.path.join(visual_path,'decoder_image_'+str(i)+"_.png"))
            val_loss_list.append(val_loss.avg)

            save_decoder_path = os.path.join(decoder_sub_path,"decoder_"+str(epoch)+".bin")
            torch.save(decoder.state_dict(), save_decoder_path)
            save_list_path = os.path.join(decoder_sub_path,"decoder_list"+str(epoch)+".bin")
            torch.save([train_loss_list,val_loss_list], save_list_path)
            plot_result_path = os.path.join(decoder_sub_path,args.task_name+'_'+str(layer_id)+".png")
            plot_loss(train_loss_list, val_loss_list, plot_result_path)

