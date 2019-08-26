import torch
import torch.nn as nn
import torch.nn.functional as F 

class BaselineNet(nn.Module):

    def __init__(self, image_channels=1, num_class=10):
        super(BaselineNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(image_channels, 256, 5, padding=2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, 5, padding=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 128, 5, padding=2), nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(nn.Linear(128*28*28, 328), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(328, 192), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(192, num_class)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128*28*28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        y = self.fc3(x)

        return y

    def params_count(self):

        total_num = sum([param.nelement() for param in self.parameters()])

        return total_num

if __name__=='__main__':

    # 1. fake data
    images = torch.rand(8, 1, 28, 28).cuda(0)

    # 2. build the model
    net = BaselineNet().cuda(0)
    print('Number of parameters: {:.2f}M'.format(net.params_count() / 1e6))

    # 3. forward
    prob = net(images)

    # 4. info
    print(prob.shape)