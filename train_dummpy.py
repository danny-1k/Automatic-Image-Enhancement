from models import MobileNet
from trainer import Trainer
from torch.nn import MSELoss
from data import DeterDataset
import torch.nn as nn
import torch

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader
import yaml
from torch.utils.tensorboard import SummaryWriter


class dData(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.x = torch.eye(10)
        self.y = torch.ones((10, 1))

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# train = DataLoader(dData(), batch_size=1)
# test = DataLoader(dData(), batch_size=1)


# net = nn.Linear(10, 1)
# lossfn = MSELoss()
# optimizer = Adam(net.parameters(), 1e-2)
    
# writer = SummaryWriter(f"runs/dummy_run_002")

# for epoch in range(100):

#     train_loss_epoch = 0
#     test_loss_epoch = 0

#     net.train()
#     for x,y in train:
#         optimizer.zero_grad()
#         p = net(x)

#         loss = lossfn(p, y)
#         loss.backward()

#         optimizer.step()

#         train_loss_epoch = train_loss_epoch=(1-.6)*train_loss_epoch + .6*loss.item()

#     net.eval()
#     for x,y in test:

#         p = net(x)

#         loss = lossfn(p, y)

#         test_loss_epoch = test_loss_epoch=(1-.6)*test_loss_epoch + .6*loss.item()


#     writer.add_scalar('Loss/train', train_loss_epoch, epoch+1)
#     writer.add_scalar('Loss/test', test_loss_epoch, epoch+1)





# train_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['train_parameters']['models']['mobilenet_pretrained']
# test_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['test_parameters']['models']['mobilenet_pretrained']

train_config = {
    'lr': 1e-2,
    'batch_size': 1,
    'epochs': 100
}

test_config = {
    'batch_size': 1,
}


trainer_params = {
    'device': 'cpu',
    'run_name': 'dummy_run_001',
    'model': nn.Linear(10, 1),
    'lossfn': MSELoss()
}


t = Trainer(train_config=train_config, test_config=test_config, trainer_params=trainer_params)
t.run(
    traindata=dData(),
    testdata=dData()
)