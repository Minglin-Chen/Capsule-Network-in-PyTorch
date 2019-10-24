import torch
import torch.nn as nn
import torch.nn.functional as F

from .CapsNetLoss import CapsNetLoss

crtierion = {
    'ce': nn.CrossEntropyLoss,
    'capsnetloss': CapsNetLoss
}

def criterion_provider(name, **kwargs):

    loss = crtierion[name](**kwargs)
    
    return loss