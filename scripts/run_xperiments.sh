#!/usr/bin/env bash

#script to run all the experiments, like wtf doesn't love automation

cd ..

python run_experiments.py --experiment_group smallnet_lr_0.1
python run_experiments.py --experiment_group smallnet_lr_0.01
python run_experiments.py --experiment_group smallnet_lr_0.001
python run_experiments.py --experiment_group smallnet_lr_0.001
python run_experiments.py --experiment_group smallnet_lr_0.0001

python run_experiments.py --experiment_group smallnet2_lr_0.1
python run_experiments.py --experiment_group smallnet2_lr_0.01
python run_experiments.py --experiment_group smallnet2_lr_0.001
python run_experiments.py --experiment_group smallnet2_lr_0.001
python run_experiments.py --experiment_group smallnet2_lr_0.0001

