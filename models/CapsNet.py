import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsLayer2D(nn.Module):

    def __init__(self, in_caps_num, in_caps_dim, out_caps_num, out_caps_dim,
                    kernel_size, stride=1, padding=0, routing=True):
        super(CapsLayer2D, self).__init__()
        
        self.in_caps_num = in_caps_num
        self.in_caps_dim = in_caps_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.routing = routing

        # W_ij in Eq.2
        for i in range(in_caps_num):
            for j in range(out_caps_num):
                setattr(self, 'W_{}{}'.format(i+1, j+1), 
                    nn.Conv2d(in_caps_dim, out_caps_dim, kernel_size, stride, padding, bias=False))
    
    def forward(self, in_caps):
        """
        in_caps: (batch_size, in_caps_num, in_caps_dim, height, width)
        output: (batch_size, out_caps_num, out_caps_dim, height, width)
        """

        u_hat = []
        for i in range(self.in_caps_num):
            u_i = in_caps[:, i]
            u_hat_list = []
            for j in range(self.out_caps_num):
                u_hat_ji = getattr(self, 'W_{}{}'.format(i+1, j+1))(u_i)
                u_hat_list.append(u_hat_ji)
            u_hat.append(torch.stack(u_hat_list, dim=1))
        u_hat = torch.stack(u_hat,dim=1)    # (batch_size, in_caps_num, out_caps_num, out_caps_dim, height, width)

        if self.routing:
            out_caps = self.routing_fn(u_hat, r=3)
        else:
            u_hat = torch.squeeze(u_hat, dim=1)
            out_caps = self.squash_fn(u_hat)

        return out_caps

    def squash_fn(self, capsule):

        """
        capsule: (batch_size, caps_num, caps_dim, ...)
        output: (batch_size, caps_num, caps_dim, ...)
        """

        s = capsule
        s_sqr = s ** 2
        s_sqr_sum = torch.sum(s_sqr, 2, keepdim=True)
        v = s * s_sqr_sum / (torch.sqrt(s_sqr_sum) * (1+s_sqr_sum))

        return v

    def routing_fn(self, u_hat, r):

        """
        input: u_hat (batch_size, in_caps_num, out_caps_num, out_caps_dim, height, width)
        """
        batch_size, in_caps_num, out_caps_num, _, _, _ = u_hat.shape

        b = torch.zeros(batch_size, in_caps_num, out_caps_num).cuda()

        for _ in range(r):
            c = F.softmax(b, dim=1)         # Eq.3
            
            s = []
            for j in range(out_caps_num):
                s_j = c[:, 0, j].reshape([batch_size, 1, 1, 1]) * u_hat[:, 0, j]
                for i in range(1, in_caps_num):
                    s_j += c[:, i, j].reshape([batch_size, 1, 1, 1]) * u_hat[:, i, j]
                s.append(s_j)
            s = torch.stack(s, dim=1)
            
            print('v {}'.format(s.shape))

            v = self.squash_fn(s)

            v_ = v.unsqueeze(dim=1)
            agreement = torch.sum(u_hat*v_, dim=(3, 4, 5))
            b += agreement
        return v

class CapsNet(nn.Module):

    def __init__(self, image_channels=1, num_class=10):
        super(CapsNet, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(image_channels, 256, 9, stride=1, padding=0, bias=False), 
            nn.ReLU(inplace=True))

        self.PrimaryCaps = CapsLayer2D(in_caps_num=1, in_caps_dim=256, out_caps_num=32, out_caps_dim=8, 
                    kernel_size=9, stride=2, padding=0, routing=False)

        self.DigitCaps = CapsLayer2D(in_caps_num=32, in_caps_dim=8, out_caps_num=num_class, out_caps_dim=16, 
                    kernel_size=6, stride=1, padding=0, routing=True)

    def forward(self, x):

        x = self.Conv1(x)
        x = x.unsqueeze(dim=1)
        x = self.PrimaryCaps(x)
        x = self.DigitCaps(x)
        y = x.squeeze()

        return y

if __name__=='__main__':

    x = torch.rand([8, 1, 28, 28]).cuda(0)

    net = CapsNet().cuda(0)

    y = net(x)

    print(y.shape)

