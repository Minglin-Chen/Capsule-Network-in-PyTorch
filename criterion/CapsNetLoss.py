import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsNetLoss(nn.Module):

    def __init__(self, w_recon=0.0005):
        super(CapsNetLoss, self).__init__()

        self.cls_loss_func = MarginLoss()
        self.reconstruction_loss_func = nn.MSELoss(reduction='sum')
        self.w_recon = w_recon

    def forward(self, probability, target, reconstruction=None, image=None):
        
        total_loss = self.cls_loss_func(probability, target)

        if (reconstruction is not None) and (image is not None):
            batch_size = image.shape[0]
            reconstruction_loss = self.reconstruction_loss_func(reconstruction, image.reshape(batch_size, -1))
            reconstruction_loss = reconstruction_loss / float(batch_size)
            total_loss = total_loss + self.w_recon * reconstruction_loss

        return total_loss


class MarginLoss(nn.Module):

    def __init__(self, m_pos=0.9, m_neg=0.1, w_neg=0.5, onehot=False):
        super(MarginLoss, self).__init__()

        self.m_pos = m_pos
        self.m_neg = m_neg
        self.w_neg = w_neg
        self.onehot = onehot

    def forward(self, input, target):

        target_onehot = target.float() if self.onehot else torch.zeros(input.shape, device=target.device).scatter_(1, target.unsqueeze(dim=1), 1)

        pos_term = (self.m_pos - input).relu_() ** 2
        neg_term = (input - self.m_neg).relu_() ** 2

        loss = target_onehot * pos_term + self.w_neg * (1.0 - target_onehot) * neg_term
        loss = loss.mean()

        return loss