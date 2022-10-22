import torch
import torchvision.transforms as transforms

train_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(), std=()),
    transforms.RandomHorizontalFlip(.4),

])

test_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(), std=()),
])