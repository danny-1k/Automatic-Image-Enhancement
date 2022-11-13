import yaml

import os

from models import Model1, Model2, Model3, SmallNet
from trainer import Trainer

from torch.nn import MSELoss

config = yaml.load(open('config.yml','r').read(), Loader=yaml.Loader)

train_config = config['train_parameters']['models']['smallnet']
test_config = config['test_parameters']['models']['smallnet']

if not os.path.exists('runs/smallnet001'):
    trainer_params = {
        'device':'cpu',
        'run_name':'smallnet001',
        'model': SmallNet(),
	'lossfn': MSELoss(),
    }

    t = Trainer(train_config, test_config, trainer_params)
    t.run()

else:
    print('SmallNet has already been run on this run_name')

