import torch
import os
from torch.utils.data import Dataset
import random

import data_utils

from transforms import train_t, test_t

import yaml

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