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
            'batch_size': 64,
            'num_workers': 8,
            'model': 'CapsNet',
            'model_param': {'image_channels':1, 'num_class':10},
            'criterion': 'nhybrid',
            'criterion_param': {'mapping':[[1,2,3],[1,3],[3]]},
            'optimizer': optim.Adam,
            'optimizer_param': {'lr':1e-4},
            'scheduler': None,
            'scheduler_param': {},
            'eval_per_epoch': 1,
            'num_epoch': 10,
            'device_ids': [0],
            'num_folds': 5 }
        
    return _config

config = get_config_from_dataset(dataset_name='MNIST')