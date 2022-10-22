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


        if labels == []:
            raise ValueError('No data in `data` directory')

        self.label_to_idx = {label:idx for idx in range(0, len(labels))}

        files = []


        for label in labels:
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

        label_enc = self.label_to_idx[x.split('/')[1]]

        x = data_utils.read_img(x)
        
        x = self.transforms(x)

        labels = torch.Tensor(labels)/2 #normalize


        return x, label_enc, labels


    def seed(self):
        random.seed(self.random_state)