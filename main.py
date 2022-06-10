# main
import os
import argparse
from step_net import step_net
from step_adv_net import step_adv_net
from step_decoder import step_decoder
from step_capability_x import step_capability_x
from step_capacility_x_imagenet import step_capability_x_imagenet
from step_capability_jacobian import step_capability_jac
from step_capability_y import step_capability_y
from step_result import step_result
from step_capability_unet import step_unet

root_path = os.getcwd()

parser = argparse.ArgumentParser('parameters')

parser.add_argument('--datasets_name',type=str,default='ISBI',help='cifar10, ImageNet')
parser.add_argument('--net_name',type=str,default='unet',help='resnet20,resnet32,resnet44')
parser.add_argument('--task_name',type=str,default='origin',help='')
parser.add_argument('--step_name',type=str,default='unet',help='net,x,y,jac,adv,unet')

parser.add_argument('--gpu_id',type=int,default=0,help='')

parser.add_argument('--out_planes',type=int,default=200,help='10 for cifar10')
parser.add_argument('--net_id',type=int,default=1,help='sub_id of the net')
parser.add_argument('--net_epoch',type=int,default=10,help='')
parser.add_argument('--net_batch_size',type=int,default=128,help='')
parser.add_argument('--net_lr',type=float,default=1e-2,help='')
parser.add_argument('--net_weight_decay',type=float,default=1e-4,help='')
parser.add_argument('--net_momentum',type=float,default=0.9,help='')
parser.add_argument('--net_milestones',type=list,default=[50,100],help='')
parser.add_argument('--net_gama',type=float,default=0.1,help='')
parser.add_argument('--net_if_pretrained',type=int,default=1,help='0 for false,1 for true.')
parser.add_argument('--image_size',type=int,default=224,help='the size of image in datasets.')
parser.add_argument('--if_alexnet',type=int,default=0,help='parameter for Decoder structure. 0 for false and 1 for true.')

parser.add_argument('--decoder_batch_size',type=int,default=16,help='')
parser.add_argument('--decoder_num_block',type=int,default=6,help='all experiments use 6.')
parser.add_argument('--decoder_epoch',type=int,default=5,help='')
parser.add_argument('--decoder_lr',type=float,default=0.01,help='')

parser.add_argument('--num_layer',type=int,default=5,help='5 layers of resnet20. 8 layers of resnet32.11 layers of resnet44')
parser.add_argument('--capability_epoch',type=int,default=100,help='')
parser.add_argument('--capability_batch_size',type=int,default=8,help='if the batchsize is so small, the result may be wrong.')
parser.add_argument('--capability_lr_x',type=float,default=1e-4,help='')
parser.add_argument('--capability_lr_y',type=float,default=1e-4,help='')
parser.add_argument('--capability_lambda_init_x',type=float,default=5,help='lambda can be changed.')
parser.add_argument('--capability_lambda_init_y',type=float,default=1e+4,help='lambda can be changed.')
# lambda in nets,tasks and layers can be different.
parser.add_argument('--capability_lambda_change_ratio',type=float,default=1,help='')
parser.add_argument('--capability_sigma_init_decay',type=float,default=0.01,help='the initialization of sigma is setting same value of all sigma')
parser.add_argument('--capability_sigma_size',type=int,default=16,help='if the size of image is 224 and the size of sigma is 16, then 1 sigma have 14*14 pixels.')
parser.add_argument('--capability_first_epoch_train_steps',type=int,default=1,help='training steps of first epoch to compute sigma_f')
parser.add_argument('--capability_num_batch',type=int,default=50,help='')
parser.add_argument('--capability_start_layer',type=int,default=3,help='the start layer id')
parser.add_argument('--capability_num_layer',type=int,default=3,help='num of layers')
parser.add_argument('--result_distance_clip',type=float,default=1.5,help='all experiments use the same distance clip')

parser.add_argument('--capability_if_shuffle',type=int,default=0,help='1 for true, 0 for false')
parser.add_argument('--capability_get_all_layer',type=int,default=0,help='1 for true, 0 for false')
parser.add_argument('--batch_interval_num',type=int,default=25,help='num of intervals of batch')


# we will show other parameters soon

# distillation
parser.add_argument('--teacher_net_name',type=str,default='resnet18',help='resnet18,resnet34,vgg16')
parser.add_argument('--teacher_net_epoch',type=int,default=100,help='')

# destroyed
parser.add_argument('--net_mode',type=int,default=0,help=' model mode is used to describe how we destroy the net. '
                                                         'when model mode is 1, the task mane should be "destroyed1"'
                                                         '0,1,2,3')
parser.add_argument('--distillation_lambda',type=float,default=1,help='parameter for DeepDistillation')

# compression
parser.add_argument('--net_compressed_id',type=int,default=1,help='parameter for DeepCompression')
parser.add_argument('--s_fc',type=float,default=1,help='parameter for DeepCompression')

# multiepoch
parser.add_argument('--capability_if_multiepoch',type=int,default=0,help='1 for true, 0 for false')

# adv
parser.add_argument('--adv_type', type=str, default="PGD")
parser.add_argument('--adv_lr', type=float, default=1e-3)
parser.add_argument('--adv_epoch', type=int, default=10)

# visual
parser.add_argument('--capability_visual_sample_num',type=int,default=25,help='num of visualization samples')

args = parser.parse_args()

if 'net' == args.step_name:
    step_net(root_path, args)
if 'x' == args.step_name:
    step_capability_x(root_path,args)
    step_result(root_path, args, x_or_y='x')
if 'y' == args.step_name:
    step_decoder(root_path, args)
    step_capability_y(root_path,args)
    step_result(root_path, args, x_or_y='y')
if 'xy' == args.step_name:
    step_capability_x(root_path,args)
    step_result(root_path, args, x_or_y='x')
    step_decoder(root_path, args)
    step_capability_y(root_path,args)
    step_result(root_path, args, x_or_y='y')
if "adv" == args.step_name:
    step_adv_net(root_path, args)
if "unet" == args.step_name:
    step_unet(root_path, args)
if "jac" == args.step_name:
    step_capability_jac(root_path, args)



