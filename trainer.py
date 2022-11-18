import os
import shutil
import pickle
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data import ImageData


class Trainer:
    def __init__(self, train_config, test_config, trainer_params):
        """
        train_config: train config
        test_config: test config
        trainer_params: dict containing run_name, model, lossfn, optimizer and device

        """

        self.train_config = train_config
        self.test_config = test_config
        self.trainer_params = trainer_params
        self.device = trainer_params['device']
        self.lossfn = trainer_params['lossfn']
        self.run_name = trainer_params['run_name']

        self.model = self.trainer_params['model'].to(self.device) #eventually won't use that much
        self.model_name = self.model.__class__.__name__ #eventually won't use that much

        self.writer = SummaryWriter(f"runs/{self.run_name}")

        self.metrics = {
            'best_loss': float('inf'),
            'train_loss': 0,
            'test_loss': 0,
        }


    def train_epoch(self, trainloader, model, optimizer):
        model.train()

        for x,y in trainloader:     
            optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)

            p = model(x)
            loss = self.lossfn(p, y)

            self.update_metrics(train=(1-.6)*self.metrics['train_loss'] + .6*loss.item())
            loss.backward()
            optimizer.step()
            

    def test_epoch(self, testloader, model, scheduler):
        model.eval()
        with torch.no_grad():
            for x,y in testloader:     

                x = x.to(self.device)
                y = y.to(self.device)

                p = model(x)
                loss = self.lossfn(p, y)

                self.update_metrics(test=(1-.6)*self.metrics['test_loss'] + .6*loss.item())

            scheduler.step(loss.item())

    def update_metrics(self, train=None, test=None):
        if train:
            self.metrics['train_loss'] = train
        if test:
            self.metrics['test_loss'] = test


    def tb_write_metrics(self, epoch):
        self.writer.add_scalars(
            main_tag='Loss',
            tag_scalar_dict={
                'train': self.metrics['train_loss'],
                'test': self.metrics['test_loss']
            },
            global_step=epoch
        )

    
    def tb_close_writer(self):
        self.writer.close()

    def save_best(self, model):
        if self.metrics['test_loss'] < self.metrics['best_loss']:
            torch.save(model.state_dict(), f'models/{self.model_name}/{self.run_name}/best.pt')
            self.metrics['best_loss'] = self.metrics['test_loss']
        

    def run(self, traindata=None, testdata=None):
        if not os.path.exists(f'models/{self.model_name}/{self.run_name}'):
            os.makedirs(f'models/{self.model_name}/{self.run_name}')

        if not os.path.exists(f'run_tracker/{self.model_name}/{self.run_name}'):
            os.makedirs(f'run_tracker/{self.model_name}/{self.run_name}')

        pickle.dump(self.train_config, open(f'run_tracker/{self.model_name}/{self.run_name}/train_config.pkl', 'wb'))
        pickle.dump(self.test_config, open(f'run_tracker/{self.model_name}/{self.run_name}/test_config.pkl', 'wb'))
        pickle.dump(self.trainer_params, open(f'run_tracker/{self.model_name}/{self.run_name}/trainer_params.pkl', 'wb'))
        shutil.copy('config.yml', f'run_tracker/{self.model_name}/{self.run_name}/config.yml')



        traindata = traindata or ImageData(train=True)
        testdata = testdata or ImageData(train=False)

        train = DataLoader(traindata, batch_size=self.train_config['batch_size'], shuffle=True)
        test = DataLoader(testdata, batch_size=self.test_config['batch_size'], shuffle=True)
        

        model = self.trainer_params['model'].to(self.device)
        self.model_name = model.__class__.__name__
        optimizer = Adam(model.parameters(), lr=self.train_config['lr'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)


        for epoch in tqdm(range(self.train_config['epochs'])):

            self.train_epoch(train, model, optimizer)
            self.test_epoch(test, model, scheduler)

            self.tb_write_metrics(epoch+1)

            self.save_best(model)
