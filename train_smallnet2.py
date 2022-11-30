from models import SmallNet2
from trainer import Trainer
from torch.nn import MSELoss
from data import DeterDataset

import yaml


train_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['train_parameters']['models']['smallnet2b']
test_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['test_parameters']['models']['smallnet2b']

trainer_params = {
    'device': 'cpu',
    'run_name': 'smallnet2b_run_001',
    'model': SmallNet2(),
    'lossfn': MSELoss()
}


t = Trainer(train_config=train_config, test_config=test_config, trainer_params=trainer_params)
t.run(
    traindata=DeterDataset(train=True),
    testdata=DeterDataset(train=False)
)