import torch
import torch.nn as nn
import torch.nn.functional as F

from .CapsLayer import CapsConv2D

class CapsNet(nn.Module):

    def __init__(self, image_channels=1, num_class=10):
        super(CapsNet, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(image_channels, 256, 9, stride=1, padding=0, bias=False), 
            nn.ReLU(inplace=True))

        self.PrimaryCaps = CapsConv2D(in_caps_num=1, in_caps_dim=256, out_caps_num=32, out_caps_dim=8, 
                    kernel_size=9, stride=2, padding=0, num_iter=0, bias_train=True)

        self.DigitCaps = CapsConv2D(in_caps_num=32, in_caps_dim=8, out_caps_num=num_class, out_caps_dim=16, 
                    kernel_size=6, stride=1, padding=0, num_iter=3, bias_train=True)

        # Init weights
        for m in self.Conv1.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0.0, std=5e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
        
        for m in self.PrimaryCaps.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
        
        for m in self.DigitCaps.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
            x: (batch_size, image_channels, 28, 28)
            probability: (batch_size, num_class)
            capsule_embedding: (batch_size, num_class, 16)
        """

        x = self.Conv1(x)
        x = x.unsqueeze(dim=1)
        x = self.PrimaryCaps(x)
        capsule_embedding = self.DigitCaps(x)

        probability = capsule_embedding.norm(dim=2, keepdim=False)

        return probability[:,:,0,0], capsule_embedding[:,:,:,0,0]

    def params_count(self):
        total_num = sum([param.nelement() for param in self.parameters()])
        return total_num


class CapsNet_with_Decoder(nn.Module):

    def __init__(self, image_channels=1, num_class=10):
        super(CapsNet_with_Decoder, self).__init__()

        self.capsnet = CapsNet(image_channels, num_class)

        self.decoder = nn.Sequential(
            nn.Linear(160, 512, bias=False), nn.ReLU(inplace=True),
            nn.Linear(512, 1024, bias=False), nn.ReLU(inplace=True),
            nn.Linear(1024, 784, bias=False), nn.Sigmoid() )

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0.0, std=5e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x, gt=None):
        """
            x: (batch_size, image_channels, 28, 28)
            gt: (batch_size, )
            probability: (batch_size, num_class)
            capsule_embedding: (batch_size, num_class, 16)
            x_reconstruction: (batch_size, 28*28)
        """

        probability, capsule_embedding = self.capsnet(x)

        if gt is not None:
            gt_onehot = torch.zeros(capsule_embedding.shape[:2], device=gt.device)
            gt_onehot.scatter_(1, gt.unsqueeze(dim=1), 1.0)
            capsule_embedding_mask = capsule_embedding * gt_onehot.unsqueeze(dim=2)
            batch_size, num_class, caps_dim = capsule_embedding_mask.shape
            capsule_embedding_flat = capsule_embedding_mask.reshape(batch_size, num_class*caps_dim)
            reconstruction = self.decoder(capsule_embedding_flat)
            return probability, capsule_embedding, reconstruction

        return probability, capsule_embedding
    
    def params_count(self):
        total_num = sum([param.nelement() for param in self.parameters()])
        return total_num


if __name__=='__main__':

    # 1. fake data
    x = torch.rand([8, 1, 28, 28]).cuda()
    gt = torch.tensor([0,1,2,3,4,5,6,7], dtype=torch.long).cuda()

    # 2. model
    net1 = CapsNet().cuda()
    net2 = CapsNet_with_Decoder().cuda()
    print('CapsNet Number of parameters: {:.2f}M'.format(net1.params_count() / 1e6))
    print('CapsNet_with_Decoder Number of parameters: {:.2f}M'.format(net2.params_count() / 1e6))

    # 3. forward
    probability, capsule_embedding = net1(x)
    print('CapsNet probability {}, capsule_embedding {}'.format(probability.shape, capsule_embedding.shape))
    probability, capsule_embedding = net2(x)
    print('CapsNet_with_Decoder probability {}, capsule_embedding {}'.format(
        probability.shape, capsule_embedding.shape))
    probability, capsule_embedding, reconstruction = net2(x, gt)
    print('CapsNet_with_Decoder probability {}, capsule_embedding {}, reconstruction {}'.format(
        probability.shape, capsule_embedding.shape, reconstruction.shape))


