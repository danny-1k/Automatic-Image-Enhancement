from models import MobileNet
from trainer import Trainer
from torch.nn import MSELoss
from data import DeterDataset

import yaml


train_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['train_parameters']['models']['mobilenet_pretrained']
test_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['test_parameters']['models']['mobilenet_pretrained']

trainer_params = {
    'device': 'cpu',
    'run_name': 'mobilenet_run_001',
    'model': MobileNet(),
    'lossfn': MSELoss()
}


t = Trainer(train_config=train_config, test_config=test_config, trainer_params=trainer_params)
t.run(
    traindata=DeterDataset(train=True),
    testdata=DeterDataset(train=False)
)