import torch
import unittest
from models import AutoCorrectorBaseLine
from models import fetch_model

x = torch.zeros((1, 3, 256, 256))
n = AutoCorrectorBaseLine()

print(n(x, torch.Tensor([0]).long()))

print(fetch_model('autosomthing baseline'))