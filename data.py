import torch
import os
from torch.utils.data import Dataset
import random
import data_utils
import transforms
from transforms import train_t, test_t
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.Loader)

data_config = config['data_config']


class ImageData(Dataset):
    def __init__(self, random_state=34, train=True, tratio=.8):
        self.random_state = random_state

        self.seed()

        try:
            labels = os.listdir(data_config['data_loc'])
        except FileNotFoundError:
            print('`data directory` does not exist... Creating')
            os.makedirs(data_config['data_loc'])

        files = []


        for label in labels:
            if label != 'unsplash': continue
            for f in os.listdir(os.path.join(data_config['data_loc'], label)):
                
                path = os.path.join(data_config['data_loc'], label, f)

                files.append(path)

        
        random.shuffle(files)


        if train:
            self.files = files[:int(tratio*len(files))]
            self.transforms = train_t
        
        else:
            self.files = files[int(tratio*len(files)):]
            self.transforms = test_t

        

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        x = self.files[idx]

        x = data_utils.read_img(x)

        x, y = data_utils.random_manipulation(x)
        
        x = self.transforms(x)

        y = torch.Tensor(y)

        return x, y


    def seed(self):
        random.seed(self.random_state)


class DeterDataset(Dataset):
    def __init__(self,train=True, subset=-1):
        self.train = train

        if train:
            self.df = pd.read_csv('data/generated/train.csv').iloc[:subset]
            self.transforms = transforms.vgg_train_transform
        else:
            self.df = pd.read_csv('data/generated/test.csv').iloc[:subset]
            self.transforms = transforms.vgg_transform
            
        
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        point = self.df.iloc[idx]

        x = point['img_id']

        x = data_utils.read_img(f'data/generated/images/{x}.jpg')
        x = self.transforms(x)
        
        brightness = point['brightness'].item()
        saturation = point['saturation'].item()
        contrast = point['contrast'].item()
        sharpness = point['sharpness'].item()

        y = torch.Tensor([brightness, saturation, contrast, sharpness])

        return x, y


class MaskData(Dataset):
    def __init__(self, train=True, subset=-1):
        super().__init__()

        self.train = train

        if train:
            self.df = pd.read_csv('data/mask_data/train.csv').iloc[:subset]
            self.transforms = transforms.vgg_train_transform
        else:
            self.df = pd.read_csv('data/generated/test.csv').iloc[:subset]
            self.transforms = transforms.vgg_transform
            

    
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        point = self.df.iloc[idx]

        aug_img_id = point['img_id']
        original_img_path = point['original_img_path']


        aug_img = data_utils.read_img(f'data/mask_data/images/{aug_img_id}.jpg')
        original_img = data_utils.read_img(original_img_path)

        x = self.transforms(aug_img)

        # the label is the difference between channels in the orignal and augumented images for different colorspaces

        aug_img_hsv = cv2.cvtColor(np.asarray(aug_img), cv2.COLOR_RGB2HSV)
        original_img_hsv = cv2.cvtColor(np.asarray(original_img), cv2.COLOR_RGB2HSV)

        saturation_difference_channel = (original_img_hsv[:, :, 1] - aug_img_hsv[:, :, 1])/100
        brightness_difference_channel = (original_img_hsv[:, :, 2] - aug_img_hsv[:, :, 2])/255

        y = torch.from_numpy(np.stack([brightness_difference_channel, saturation_difference_channel, ], axis=0))


        return x, y
