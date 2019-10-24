import torch
import torch.nn as nn
import torch.nn.functional as F


def _squash(in_caps, dim):
    norm = in_caps.norm(dim=dim, keepdim=True)
    norm_squared = norm * norm
    return (in_caps / (norm + 1e-8)) * (norm_squared / (1 + norm_squared))

def _routing(pred_caps, bias, num_iter):
    """
        pred_caps: (batch_size, in_caps_num, out_caps_num, out_caps_dim, height, width)
        bias: (1, in_caps_num, out_caps_num, 1, 1, 1)
        out_caps: (batch_size, out_caps_num, out_caps_dim, height, width)
    """
    # initial logits
    logits = bias.expand_as(pred_caps)

    # loop to update logits
    for _ in range(num_iter):

        coupling_coef = F.softmax(logits, dim=2)

        out_caps = coupling_coef * pred_caps
        out_caps = out_caps.sum(dim=1, keepdim=True)
        out_caps = _squash(out_caps, dim=3)
        
        agreement = torch.sum(out_caps*pred_caps, dim=3, keepdim=True)
        logits = logits + agreement

    # forward
    coupling_coef = F.softmax(logits, dim=2)

    out_caps = coupling_coef * pred_caps
    out_caps = out_caps.sum(dim=1, keepdim=False)
    out_caps = _squash(out_caps, dim=2)

    return out_caps


class CapsConv2D(nn.Module):

    def __init__(self, in_caps_num, in_caps_dim, out_caps_num, out_caps_dim, 
                    kernel_size, stride=1, padding=0, num_iter=3, bias_train=True):
        super(CapsConv2D, self).__init__()
        
        self.in_caps_num = in_caps_num
        self.in_caps_dim = in_caps_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.num_iter = num_iter

        self.op = nn.Conv2d(
            self.in_caps_num*self.in_caps_dim, 
            self.in_caps_num*self.out_caps_num*self.out_caps_dim, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=self.in_caps_num, 
            bias=False)

        # coupling coefficient logits
        if num_iter > 0:
            self.bias = nn.Parameter( 
                torch.zeros(1, self.in_caps_num, self.out_caps_num, 1, 1, 1),
                requires_grad=bias_train)
    
    def forward(self, in_caps):
        """
            in_caps: (batch_size, in_caps_num, in_caps_dim, height, width)
            out_caps: (batch_size, out_caps_num, out_caps_dim, height, width)
        """
        # input capsules
        batch_size, _, _, h, w = in_caps.shape
        in_caps = in_caps.reshape(batch_size, self.in_caps_num*self.in_caps_dim, h, w)

        # prediction capsules
        pred_caps = self.op(in_caps)
        batch_size, _, h, w = pred_caps.shape
        pred_caps = pred_caps.reshape(batch_size, self.in_caps_num, self.out_caps_num, self.out_caps_dim, h, w)

        # dynamic routing
        if self.num_iter > 0:
            out_caps = _routing(pred_caps, self.bias, self.num_iter)
        else:
            out_caps = pred_caps.sum(dim=1)
            out_caps = _squash(out_caps, dim=2)
            
        return out_caps