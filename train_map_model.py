from models import MapModel
from trainer import Trainer
from torch.nn import MSELoss, L1Loss
from data import MaskData

import yaml


train_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['train_parameters']['models']['segnet']
test_config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)['test_parameters']['models']['segnet']


is_l1 = False


if is_l1:

    trainer_params = {
        'device': 'cpu',
        'run_name': 'segnet_run_L1Loss_001',
        'model': MapModel(),
        'lossfn': L1Loss()
    }


    t = Trainer(train_config=train_config, test_config=test_config, trainer_params=trainer_params)
    t.run(
        traindata=MaskData(train=True),
        testdata=MaskData(train=False)
    )

else:
    trainer_params = {
        'device': 'cpu',
        'run_name': 'segnet_run_MSELoss_001',
        'model': MapModel(),
        'lossfn': MSELoss()
    }


    t = Trainer(train_config=train_config, test_config=test_config, trainer_params=trainer_params)
    t.run(
        traindata=MaskData(train=True),
        testdata=MaskData(train=False)
    )