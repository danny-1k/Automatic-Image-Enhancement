from models import MapModelPretrainedVGG
from trainer import Trainer
import torch
from torch.nn import MSELoss, L1Loss
from data import MaskData
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import yaml

from tqdm import tqdm

import os

import sys


train_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['train_parameters']['models']['segnet_pretrained']
test_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['test_parameters']['models']['segnet_pretrained']


is_l1 = False


if is_l1:
    DEVICE = 'cpu'
    RUN_NAME = 'segnet_pretrained_sigmoid_nobn_run_l1Loss_004'
    lossfn = L1Loss()

else:
    DEVICE = 'cpu'
    RUN_NAME = 'segnet_pretrained_sigmoid_nobn_run_MSELoss_004'
    lossfn = MSELoss()


net = MapModelPretrainedVGG(use_batch_norm=False)


if os.path.exists(f'models/{net.__class__.__name__}/{RUN_NAME}'):
    print('This has already been run with the current run name')
    sys.exit(0)

else:
    os.makedirs(f'models/{net.__class__.__name__}/{RUN_NAME}')


EPOCHS = train_config['epochs']
LR = 1e-8
TRAIN_BATCH_SIZE = train_config['batch_size']
TEST_BATCH_SIZE = test_config['batch_size']

optimizer = Adam(net.parameters(), lr=3e-5)
scheduler = ExponentialLR(optimizer, gamma=.7)

train = DataLoader(MaskData(train=True), batch_size=16, shuffle=True)
test = DataLoader(MaskData(train=False), batch_size=16, shuffle=True)


writer = SummaryWriter(f'runs/{RUN_NAME}')

best_loss = float('inf')


for epoch in tqdm(range(EPOCHS)):

    train_loss_epoch = 0
    test_loss_epoch = 0

    net.train()

    for x,y in train:
        optimizer.zero_grad()

        p = net(x)
        loss = lossfn(p, y)
        loss.backward()

        optimizer.step()

        train_loss_epoch = (train_loss_epoch*.4) + (.6*loss.item())


    with torch.no_grad():
        net.eval()

        for x,y in train:

            p = net(x)
            loss = lossfn(p, y)

            test_loss_epoch = (test_loss_epoch*.4) + (.6*loss.item())

    
    writer.add_scalars(
        main_tag='Loss',
        tag_scalar_dict={
            'train': train_loss_epoch,
            'test': test_loss_epoch,
        },
        global_step=epoch+1
    )

    scheduler.step()


    if test_loss_epoch < best_loss:
        best_loss = test_loss_epoch

        torch.save(net.state_dict(), f'models/{net.__class__.__name__}/{RUN_NAME}/best.pt')

