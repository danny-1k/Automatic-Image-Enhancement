import torch
import torchvision.transforms as transforms
import yaml

config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.Loader)

data_config = config['data_config']

mean = data_config['mean']
std = data_config['std']
img_size = data_config['img_size']


train_t = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(.4),
    transforms.RandomVerticalFlip(.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_t = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])