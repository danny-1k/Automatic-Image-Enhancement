import torch
import unittest
from models import AutoCorrectorBaseLine, SmallNet,Model1, Model2, Model3
from models import fetch_model

x = torch.zeros((1, 3, 256, 256))
smallnet = SmallNet()

# m1 = Model1()
# m2 = Model2()
# m3 = Model3()

# print(m1(x))
# print(m2(x))
# print(m3(x))

print(smallnet(x))
