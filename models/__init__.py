from .BaselineNet import BaselineNet
from .CapsNet import CapsNet

model_zoo = {
    'BaselineNet': BaselineNet,
    'CapsNet': CapsNet
}

def model_provider(name, **kwargs):

    model_ret = model_zoo[name](**kwargs)
    
    return model_ret
