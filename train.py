import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import yaml

from tqdm import tqdm

from data import ImageData

import argparse

from models import fetch_model


def train(model_name, device, train_config, test_config, writer):


    traindata = ImageData(train=True)
    testdata = ImageData(train=False)

    train = DataLoader(traindata, batch_size=train_config['batch_size'], shuffle=True)
    test = DataLoader(testdata, batch_size=test_config['batch_size'], shuffle=True)

    net = fetch_model(model_name)(num_labels=len(testdata.labels))

    net.to(device)

    try:
        train_config = train_config['models'][model_name]
    except:
        raise ValueError(f'Configuration for model `{model_name}` does not exist')

    optimizer = Adam(net.parameters(), lr=train_config['lr'])
    scheduler = ReduceLROnPlateau(optimizer)

    lossfn = nn.MSELoss()


    for epoch in tqdm(range(train_config['epochs'])):

        net.train()

        for x, label, y in train:
            
            x = x.to(device)
            label = label.to(device).long()
            y = y.to(device)

            p = net(x, label)
            loss = lossfn(p, y)

            writer.add_scalar('Loss/train', loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        net.eval()
        with torch.no_grad():
            for x, label, y in test:

                x = x.to(device)
                label = label.to(device).long()
                y = y.to(device)

                p = net(x, label)
                loss = lossfn(p, y)

                writer.add_scalar('Loss/test', loss.item())



        print(f'Epoch : {epoch + 1} test_loss : {loss.item()}')




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

    writer = SummaryWriter("runs/{run_name}")


    train(model_name=model_name, train_config=trainconfig, test_config=testconfig, device=device)