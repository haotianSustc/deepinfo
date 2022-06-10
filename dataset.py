# dataset
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


class ISBI_dataset(Dataset):
    def __init__(self, data_path, num_classes, image_size):
        super(ISBI_dataset).__init__()
        self.data_path = data_path
        self.len = len(os.listdir(os.path.join(data_path, "Images")))
        self.transformations = transforms.Compose([transforms.ToTensor()])
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.len

    def _letterbox_image(self, image, label, size):
        label = Image.fromarray(np.array(label))
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        label = label.resize((nw, nh), Image.NEAREST)
        new_label = Image.new('L', size, (0))
        new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

        return new_image, new_label

    def __getitem__(self, item):
        img_pth = os.path.join(self.data_path, "Images", f"{item}.png")
        lbl_pth = os.path.join(self.data_path, "Labels", f"{item}.png")
        img = Image.open(img_pth)
        lbl = Image.open(lbl_pth)

        img, lbl = self._letterbox_image(img, lbl, (int(self.image_size[1]), int(self.image_size[0])))

        # img = np.transpose(np.array(img), [2, 0, 1]) / 255
        lbl = np.array(lbl)
        modify_png = np.zeros_like(lbl)
        modify_png[lbl <= 127.5] = 1

        seg_labels = modify_png
        seg_labels = np.eye(self.num_classes + 1)[seg_labels.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.image_size[1]), int(self.image_size[0]), self.num_classes + 1))

        img = self.transformations(img)
        lbl = torch.from_numpy(lbl) / 255.
        lbl = lbl.unsqueeze_(2).expand(self.image_size[1], self.image_size[0], self.num_classes + 1)

        return img, lbl


class CUBBounding(Dataset):

    def __init__(self,data_path,train=True,dataset_mode=0):
        super(CUBBounding, self).__init__()

        self.train = train
        self.dataset_mode = dataset_mode
        self.upsample_method = nn.Upsample(size=224,mode='bilinear')
        self.transformations = transforms.Compose([transforms.ToTensor()])

        self.CUB_root = data_path
        self.CUB_200_211_root = os.path.join(self.CUB_root, 'CUB_200_2011')
        self.CUB_200_2011_bounding_root = os.path.join(self.CUB_root, 'CUB_200_2011_bounding')
        self.CUB_200_2011_negetive_root = os.path.join(self.CUB_root, 'CUB_200_2011_negetive')

        self.classes_file = os.path.join(self.CUB_200_2011_bounding_root, 'classes.txt') # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(self.CUB_200_2011_bounding_root, 'image_class_labels.txt') # <image_id> <class_id>
        self.images_file = os.path.join(self.CUB_200_2011_bounding_root, 'images.txt') # <image_id> <image_name>
        self.train_test_split_file = os.path.join(self.CUB_200_2011_bounding_root, 'train_test_split.txt') # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(self.CUB_200_2011_bounding_root, 'bounding_boxes.txt') # <image_id> <x> <y> <width> <height>
        self.if_use_file = os.path.join(self.CUB_200_2011_bounding_root,'if_use.txt') # <used_image_id>

        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []
        self._if_use = []

        self._get_if_use()
        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _get_if_use(self):
        for line in open(self.if_use_file):
            used_id, = line.strip('\n').split()
            self._if_use.append(used_id)

    def _train_test_split(self):
        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if image_id in self._if_use:
                if label == '1':
                        self._train_ids.append(image_id)
                elif label == '0':
                        self._test_ids.append(image_id)
                else:
                    raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            if image_id in self._if_use:
                self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            if image_id in self._if_use:
                label = self._image_id_label[image_id]
                if image_id in self._train_ids:
                    self._train_path_label.append((image_name, label))
                else:
                    self._test_path_label.append((image_name, label))

    def __getitem__(self, index):

        if self.train:

            image_name, label = self._train_path_label[index]
            image_path = os.path.join(self.CUB_200_2011_bounding_root, 'images', image_name)
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')
            label = int(label)-1
            img = np.array(img)
            img = self.transformations(img)
            img = self.upsample_method(img.view(1,img.size()[0],img.size()[1],img.size()[2]))[0]

        else:

            image_name, label = self._test_path_label[index]
            image_path = os.path.join(self.CUB_200_2011_bounding_root, 'images', image_name)
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')
            label = int(label)-1
            img = np.array(img)
            img = self.transformations(img)
            img = self.upsample_method(img.view(1,img.size()[0],img.size()[1],img.size()[2]))[0]

        img = img.float()

        return img,label

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)

class CUBBinaryBounding(Dataset):

    def __init__(self, data_path,train=True,dataset_mode=0):
        super(CUBBinaryBounding, self).__init__()

        self.train = train
        self.dataset_mode = dataset_mode
        self.upsample_method = nn.Upsample(size=224,mode='bilinear')
        self.transformations = transforms.Compose([transforms.ToTensor()])

        self.CUB_root = os.path.join(data_path, "..", "CUB")
        self.CUB_200_211_root = os.path.join(self.CUB_root,'CUB_200_2011')
        self.CUB_binary_bounding_root = os.path.join(self.CUB_root,'CUB_binary_bounding')
        self.CUB_200_2011_negetive_root = os.path.join(self.CUB_root,'CUB_200_2011_negetive')
        
        self.classes_file = os.path.join(self.CUB_binary_bounding_root, 'classes.txt') # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(self.CUB_binary_bounding_root, 'image_class_labels.txt') # <image_id> <class_id>
        self.images_file = os.path.join(self.CUB_binary_bounding_root, 'images.txt') # <image_id> <image_name>
        self.train_test_split_file = os.path.join(self.CUB_binary_bounding_root, 'train_test_split.txt') # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(self.CUB_binary_bounding_root, 'bounding_boxes.txt') # <image_id> <x> <y> <width> <height>
        self.if_use_file = os.path.join(self.CUB_binary_bounding_root,'if_use.txt') # <used_image_id>
        
        self.classes_file_negetive = os.path.join(self.CUB_200_2011_negetive_root, 'classes.txt') # <class_id> <class_name>
        self.image_class_labels_file_negetive = os.path.join(self.CUB_200_2011_negetive_root, 'image_class_labels.txt') # <image_id> <class_id>
        self.images_file_negetive = os.path.join(self.CUB_200_2011_negetive_root, 'images.txt') # <image_id> <image_name>
        self.train_test_split_file_negetive = os.path.join(self.CUB_200_2011_negetive_root, 'train_test_split.txt') # <image_id> <is_training_image>
        self.bounding_boxes_file_negetive = os.path.join(self.CUB_200_2011_negetive_root, 'bounding_boxes.txt') # <image_id> <x> <y> <width> <height>
        self.if_use_file_negetive = os.path.join(self.CUB_200_2011_negetive_root,'if_use.txt') # <used_image_id>
        
        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []
        self._if_use = []
        
        self._train_ids_negetive = []
        self._test_ids_negetive = []
        self._image_id_label_negetive = {}
        self._train_path_label_negetive = []
        self._test_path_label_negetive = []
        self._if_use_negetive = []
        
        self._get_if_use()
        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _get_if_use(self):
        for line in open(self.if_use_file):
            used_id, = line.strip('\n').split()
            self._if_use.append(used_id)
        for line in open(self.if_use_file_negetive):
            used_id, = line.strip('\n').split()
            self._if_use_negetive.append(used_id)

    def _train_test_split(self):
        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if image_id in self._if_use:
                if label == '1':
                        self._train_ids.append(image_id)
                elif label == '0':
                        self._test_ids.append(image_id)
                else:
                    raise Exception('label error')
        for line in open(self.train_test_split_file_negetive):
            image_id, label = line.strip('\n').split()
            if image_id in self._if_use_negetive:
                if label == '1':
                        self._train_ids_negetive.append(image_id)
                elif label == '0':
                        self._test_ids_negetive.append(image_id)
                else:
                    raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            if image_id in self._if_use:
                self._image_id_label[image_id] = class_id
        for line in open(self.image_class_labels_file_negetive):
            image_id, class_id = line.strip('\n').split()
            if image_id in self._if_use_negetive:
                self._image_id_label_negetive[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            if image_id in self._if_use:
                label = self._image_id_label[image_id]
                if image_id in self._train_ids:
                    self._train_path_label.append((image_name, label))
                else:
                    self._test_path_label.append((image_name, label))
        for line in open(self.images_file_negetive):
            image_id, image_name = line.strip('\n').split()
            if image_id in self._if_use_negetive:
                label = self._image_id_label_negetive[image_id]
                if image_id in self._train_ids:
                    self._train_path_label_negetive.append((image_name, label))
                else:
                    self._test_path_label_negetive.append((image_name, label))

    def __getitem__(self, index):

        if self.dataset_mode == 0:
            mode_num = random.randint(0,1)
        else:
            mode_num = 1

        if self.train:

            if mode_num == 1:
                image_name, label = self._train_path_label[index]
                label = 1
                image_path = os.path.join(self.CUB_binary_bounding_root, 'images', image_name)
                img = Image.open(image_path)
                if img.mode == 'L':
                    img = img.convert('RGB')
                img = np.array(img)
                img = self.transformations(img)
                img = self.upsample_method(img.view(1,img.size()[0],img.size()[1],img.size()[2]))[0]
            else:
                index = index % len(self._train_path_label_negetive)
                image_name, label = self._train_path_label_negetive[index]
                label = 0
                image_path = os.path.join(self.CUB_200_2011_negetive_root, 'images', image_name)
                img = Image.open(image_path)
                if img.mode == 'L':
                    img = img.convert('RGB')
                img = np.array(img)
                img = self.transformations(img)
                img = self.upsample_method(img.view(1,img.size()[0],img.size()[1],img.size()[2]))[0]

        else:

            if mode_num == 1:
                image_name, label = self._test_path_label[index]
                label = 1
                image_path = os.path.join(self.CUB_binary_bounding_root, 'images', image_name)
                img = Image.open(image_path)
                if img.mode == 'L':
                    img = img.convert('RGB')
                img = np.array(img)
                img = self.transformations(img)
                img = self.upsample_method(img.view(1,img.size()[0],img.size()[1],img.size()[2]))[0]
            else:
                index = index % len(self._test_path_label_negetive)
                image_name, label = self._test_path_label_negetive[index]
                label = 0
                image_path = os.path.join(self.CUB_200_2011_negetive_root, 'images', image_name)
                img = Image.open(image_path)
                if img.mode == 'L':
                    img = img.convert('RGB')
                img = np.array(img)
                img = self.transformations(img)
                img = self.upsample_method(img.view(1,img.size()[0],img.size()[1],img.size()[2]))[0]

        img = img.float()

        return img,label

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)

