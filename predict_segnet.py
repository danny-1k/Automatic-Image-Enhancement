import torch
from data_utils import read_img
from transforms import vgg_transform
from models import MapModel, MapModelPretrainedVGG
import cv2
import numpy as np

import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2


transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
    ToTensorV2(p=1)
])

f =  'doggo.jpg'#'data/unsplash/2b444765.jpg/'
img = read_img(f, size=(224, 224))

net = MapModelPretrainedVGG(use_batch_norm=False)#MapModel()
# net.load_state_dict(torch.load('models/MapModel/segnet_run_MSELoss_001/best.pt'))
net.load_state_dict(torch.load('models/MapModelPretrainedVGG/segnet_pretrained_sigmoid_nobn_run_MSELoss_004/best.pt'))

net.eval()
net.requires_grad_(False)

x = transforms(image=np.asarray(img))['image'].unsqueeze(0)

pred_map = net(x).squeeze(0).numpy() # (2, h, w) -> (Brightness, Saturation)
pred_map[1, :, :] *= 100
pred_map[1, :, :] += .5 #helps in rounding to uint8

pred_map[0, :, :] *= 255
pred_map[0, :, :] += .5 #helps in rounding to uint8

pred_map = pred_map.astype('uint8')

plt.imshow(pred_map[0, :])
plt.show()

plt.imshow(pred_map[1, :])
plt.show()

print(pred_map.shape)

img_hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV)

img_hsv[:, :, 1] += pred_map[1, :, :] # correct saturation

img_hsv[:, :, 2] += pred_map[0, :, :] # correct saturation

#img_hsv = img_hsv.astype('uint8') 

img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

cv2.imwrite('corrected.png', img_rgb)
#img_rgb.