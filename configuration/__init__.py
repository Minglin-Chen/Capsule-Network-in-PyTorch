"""
    This file used to configurate the train procedure
"""

import torch
import torch.optim as optim

def get_config_from_dataset(dataset_name):

    _config = None

    if dataset_name == 'MNIST':
        _config = {
            'dataset_name': dataset_name,
            'dataset_root': './data/',
            'num_class': 10,
            'batch_size': 8,
            'num_workers': 8,
            'model': 'CapsNet_with_Decoder',
            'model_param': {'image_channels':1, 'num_class':10},
            'criterion': 'capsnetloss',
            'criterion_param': {},
            'optimizer': optim.Adam,
            'optimizer_param': {'lr':1e-4},
            'scheduler': optim.lr_scheduler.StepLR,
            'scheduler_param': {'step_size': 2000, 'gamma': 0.96},
            'eval_per_epoch': 1,
            'num_epoch': 300000,
            'device_ids': [2],
            'num_folds': 5 }
        
    return _config

config = get_config_from_dataset(dataset_name='MNIST')