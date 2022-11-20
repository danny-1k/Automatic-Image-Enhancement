if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_group', required=True)
    parser.add_argument('--device', default='cpu')


    args = parser.parse_args()

    experiment_group = args.experiment_group
    device = args.device


    import os
    import yaml

    from models import SmallNet, SmallNet2, LessSmallNet, MobileNet, fetch_model
    from trainer import Trainer

    from torch.nn import MSELoss

    from data import DeterDataset, MaskData


    experiments_config = yaml.load(open('experiments.yml','r').read(), Loader=yaml.Loader)['experiments']

    global_params = experiments_config['global_params']

    assert experiment_group in experiments_config

    global_params = experiments_config['global_params']

    run_config = experiments_config[experiment_group]

    for run_name in list(run_config.keys()):

        if not run_name in os.listdir('runs') or os.listdir(f'runs/{run_name}') == 0:

            train_config = run_config[run_name]
            test_config = {'batch_size': train_config['batch_size']}

            for param in list(global_params.keys()):
                train_config[param] = global_params[param]


            trainer_params = {
                'device': device,
                'run_name': run_name,
                'model': fetch_model(experiment_group.split('_')[0])(),
                'lossfn': MSELoss()
            }


            t = Trainer(train_config=train_config, test_config=test_config, trainer_params=trainer_params)
            t.run(
                traindata=DeterDataset(train=True, subset=100) if 'map' not in experiment_group else MaskData(train=True, subset=100),
                testdata=DeterDataset(train=False, subset=100) if 'map' not in experiment_group else MaskData(train=False, subset=100)
            )