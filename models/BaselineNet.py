import torch
import torch.nn as nn
import torch.nn.functional as F 

class BaselineNet(nn.Module):

    def __init__(self, image_channels=1, num_class=10):
        super(BaselineNet, self).__init__()

        # Choice 0
        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(image_channels, 512, 5, padding=2), 
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(512, 256, 5, padding=2), 
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2))
        
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(256*7*7, 1024), 
        #     nn.ReLU(inplace=True))
        
        # self.head_layer = nn.Linear(1024, num_class)

        # Choice 1
        self.conv_layer = nn.Sequential(
            nn.Conv2d(image_channels, 256, 5, padding=2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 5, padding=2),
            nn.ReLU(inplace=True) )

        self.fc_layer = nn.Sequential(
            nn.Linear(128*28*28, 328), 
            nn.ReLU(inplace=True),
            nn.Linear(328, 192), 
            nn.ReLU(inplace=True) )

        self.head_layer = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(192, num_class) )

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
    
    def forward(self, x):

        batch_size = x.shape[0]

        x = self.conv_layer(x)
        x = x.reshape(batch_size, -1)
        x = self.fc_layer(x)
        y = self.head_layer(x)

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