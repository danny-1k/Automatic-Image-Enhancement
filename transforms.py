import torch
import torchvision.transforms as transforms
import yaml

config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.Loader)

data_config = config['data_config']

mean = data_config['mean']
std = data_config['std']


train_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomHorizontalFlip(.4),
    transforms.RandomVerticalFlip(.4),


])

test_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])