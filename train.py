import os
import torch
import torch.nn as nn
import yaml
import argparse
from models import fetch_model
from trainer import Trainer


if __name__ == '__main__':
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.Loader)

    trainconfig = config['train_parameters']
    testconfig = config['test_parameters']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--model_name', required=True, help='Model name. Should be in the config.yml')
    parser.add_argument('--run_name', required=True)

    args = parser.parse_args()

    model_name = args.model_name
    run_name = args.run_name

    trainconfig = trainconfig[model_name]
    testconfig = testconfig[model_name]

    trainer_params = {
        'device':device,
        'lossfn': nn.MSELoss(),
        'run_name': run_name,
        'model': fetch_model(model_name)()
    }


    t = Trainer(trainconfig, testconfig, trainer_params)

    t.run()