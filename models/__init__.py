from .BaselineNet import BaselineNet
from .CapsNet import CapsNet, CapsNet_with_Decoder

model_zoo = {
    'BaselineNet': BaselineNet,
    'CapsNet': CapsNet,
    'CapsNet_with_Decoder': CapsNet_with_Decoder
}

def model_provider(name, **kwargs):

    model_ret = model_zoo[name](**kwargs)
    
    return model_ret
