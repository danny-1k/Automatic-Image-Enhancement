import torch
import torchvision.transforms as transforms
import yaml

config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.Loader)

data_config = config['data_config']

mean = data_config['mean']
std = data_config['std']
img_size = data_config['img_size']


vgg_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

vgg_train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(.4),
    transforms.RandomVerticalFlip(.4),
    transforms.RandomAffine(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

])


train_t = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(.4),
    transforms.RandomVerticalFlip(.4),
    transforms.RandomAffine(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_t = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])