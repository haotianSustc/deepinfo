# datasets
import os
import sys
import h5py
import scipy.io as io

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
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import re
import os
import sys
import torch
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


root = os.getcwd()
src_path = root + '/data/CelebA'
dst_path = root + '/data/cropped_CelebA'
train_num = 162770


class CelebADataset(Dataset):
    def __init__(self, category, dst_path='/data/cropped_CelebA', training=True, number=1000000):
        fn = open(src_path + '/Anno/' + category, 'r')
        fh2 = open(src_path + '/Eval/list_eval_partition.txt', 'r')
        imgs = []
        lbls = []
        ln = 0
        regex = re.compile('\s+')
        num = 0
        for line in fn:
            ln += 1
            if ln <= 2:
                continue
            if (ln - 2 <= train_num and training) or\
                (ln - 2 > train_num and not training):
                line = line.rstrip('\n')
                line_value = regex.split(line)
                imgs.append(line_value[0])
                lbls.append(list(int(i) if int(i) > 0 else 0 for i in line_value[1:]))
                num += 1
                if num >= number:
                    break
        self.imgs = imgs
        self.lbls = lbls
        self.is_train = training
        self.dst_path = root + dst_path
        if "best" in self.dst_path:
            num = len(self.imgs) // 3 * 3
            self.imgs = self.imgs[:num]
            self.lbls = self.lbls[:num]
        if training:
            self.transform = transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __getitem__(self, idx):
        fn = self.imgs[idx]
        lbls = self.lbls[idx]
        if self.is_train:
            imgs = Image.open(self.dst_path + '/train/' + fn)
        else:
            imgs = Image.open(self.dst_path + '/test/' + fn)
        imgs = self.transform(imgs)
        lbls = torch.Tensor(lbls)
        return [imgs, lbls]

    def __len__(self):
        return len(self.imgs)


class CUB_200_2011(Dataset):

    def __init__(self,data_path,train=True):
        super(CUB_200_2011, self).__init__()

        self.train = train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.RandomCrop((224, 224), 32),
        #     transforms.RandomHorizontalFlip(0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=self.mean, std=self.std)
        # ])
        # deepinfo目前不能评测数据增强的task
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.CUB_root = data_path
        self.classes_file = os.path.join(self.CUB_root, 'classes.txt') # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(self.CUB_root, 'image_class_labels.txt') # <image_id> <class_id>
        self.images_file = os.path.join(self.CUB_root, 'images.txt') # <image_id> <image_name>
        self.train_test_split_file = os.path.join(self.CUB_root, 'train_test_split.txt') # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(self.CUB_root, 'bounding_boxes.txt') # <image_id> <x> <y> <width> <height>

        self._train_ids = []
        self._test_ids = []
        self._image_class_labels = {}
        self._image_bounding_boxes = {}
        self._train_path_label = []
        self._test_path_label = []

        self._train_test_split()
        self._get_image_class_labels()
        self._get_image_path()
        self._get_image_bounding_box()

    def _train_test_split(self):
        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                    self._train_ids.append(image_id)
            elif label == '0':
                    self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_image_class_labels(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_class_labels[image_id] = class_id

    def _get_image_bounding_box(self):
        for line in open(self.bounding_boxes_file):
            image_id,x,y,width,height = line.strip('\n').split()
            self._image_bounding_boxes[image_id] = [x,y,width,height]

    def _get_image_path(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            label = self._image_class_labels[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label))
            else:
                self._test_path_label.append((image_name, label))

    def __getitem__(self, index):

        if self.train:
            image_name, label = self._train_path_label[index]
            image_path = os.path.join(self.CUB_root, 'images', image_name)
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')
            label = int(label)-1
            # ------------------------- Bounding box -----------------------------------
            bbox_x, bbox_y, bbox_w, bbox_h = self._image_bounding_boxes[self._train_ids[index]]
            ori_height, ori_width = img.height, img.width
            img = self.transform(img)
            new_height, new_width = img.size(-2), img.size(-1)
            bbox_y = int(float(bbox_y) / ori_height * new_height)
            bbox_x = int(float(bbox_x) / ori_width * new_width)
            bbox_h = int(float(bbox_h) / ori_height * new_height)
            bbox_w = int(float(bbox_w) / ori_width * new_width)
            # --------------------------------------------------------------------------
        else:
            image_name, label = self._test_path_label[index]
            image_path = os.path.join(self.CUB_root, 'images', image_name)
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')
            label = int(label)-1
            # ------------------------- Bounding box -----------------------------------
            bbox_x, bbox_y, bbox_w, bbox_h = self._image_bounding_boxes[self._test_ids[index]]
            ori_height, ori_width = img.height, img.width
            img = self.transform(img)
            new_height, new_width = img.size(-2), img.size(-1)
            bbox_y = int(float(bbox_y) / ori_height * new_height)
            bbox_x = int(float(bbox_x) / ori_width * new_width)
            bbox_h = int(float(bbox_h) / ori_height * new_height)
            bbox_w = int(float(bbox_w) / ori_width * new_width)
            # --------------------------------------------------------------------------
        img = img.float()

        return img,label, bbox_x, bbox_y, bbox_w, bbox_h

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)
