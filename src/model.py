import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class InceptionModule(nn.Module):
    def __init__(self, in_nc, out_nc, bn='BatchNorm'):
        super(InceptionModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1x1 = nn.BatchNorm2d(out_nc//4)

        self.conv1x1_2 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(out_nc//4, out_nc//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3x3 = nn.BatchNorm2d(out_nc//4)  # nn.Sequential()  #

        self.conv1x1_3 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv5x5 = nn.Conv2d(out_nc//4, out_nc//4, kernel_size=5, stride=1, padding=2, bias=False)
        self.norm5x5 = nn.BatchNorm2d(out_nc//4)

        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_4 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.normpooling = nn.BatchNorm2d(out_nc//4)

        self.conv1x1_5 = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU()  # nn.ReLU(True)

    def forward(self, x):
        out1x1 = self.relu(self.norm1x1(self.conv1x1(x)))
        out3x3 = self.relu(self.conv1x1_2(x))
        out3x3 = self.relu(self.norm3x3(self.conv3x3(out3x3)))
        out5x5 = self.relu(self.conv1x1_3(x))
        out5x5 = self.relu(self.norm5x5(self.conv5x5(out5x5)))
        outmaxpooling = self.maxpooling(x)
        outmaxpooling = self.relu(self.norm5x5(self.conv1x1_4(outmaxpooling)))

        out = torch.cat([out1x1, out3x3, out5x5, outmaxpooling], dim=1)
        residual = self.conv1x1_5(x)
        out = out + residual
        return out

class HidingNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=1):
        super(HidingNet, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(16)
        self.block1 = InceptionModule(in_nc=16, out_nc=32)
        self.block2 = InceptionModule(in_nc=32, out_nc=64)
        self.block3 = InceptionModule(in_nc=64, out_nc=128)
        self.block7 = InceptionModule(in_nc=128, out_nc=256)
        self.block8 = InceptionModule(in_nc=256, out_nc=128)
        self.block4 = InceptionModule(in_nc=128, out_nc=64)
        self.block5 = InceptionModule(in_nc=64, out_nc=32)
        self.block6 = InceptionModule(in_nc=32, out_nc=16)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(3, out_nc, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.LeakyReLU()  # nn.ReLU(True)
        self.tanh = nn.Tanh()
        # self.init_weight()

    def forward(self, secret, cover):
        # convert cover from rgb to yuv
        Y = 0 + 0.299 * cover[:, 0, :, :] + 0.587 * cover[:, 1, :, :] + 0.114 * cover[:, 2, :, :]
        CB = 128.0/255 - 0.168736 * cover[:, 0, :, :] - 0.331264 * cover[:, 1, :, :] + 0.5 * cover[:, 2, :, :]
        CR = 128.0/255 + 0.5 * cover[:, 0, :, :] - 0.418688 * cover[:, 1, :, :] - 0.081312 * cover[:, 2, :, :]
        Y = Y.unsqueeze(1)
        x = torch.cat([secret, Y], dim=1)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.tanh(self.conv3(out))
        cover[:, 0, :, :] = out[:, 0, :, :] + 1.402 * CR - 1.402 * 128.0/255
        cover[:, 1, :, :] = out[:, 0, :, :] - 0.344136 * CB + 0.344136 * 128.0/255 - 0.714136 * CR + 0.714136 * 128.0/255
        cover[:, 2, :, :] = out[:, 0, :, :] + 1.772 * CB - 1.772 * 128.0/255
        return cover

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform(m.weight)



class RevealNet(nn.Module):
    def __init__(self, nc=1, nhf=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf * 4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function())

    def forward(self, stego):
        Y = 0 + 0.299 * stego[:, 0, :, :] + 0.587 * stego[:, 1, :, :] + 0.114 * stego[:, 2, :, :]
        Y = Y.unsqueeze(1)
        output = self.main(Y)
        return output

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform(m.weight)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: {}'.format(num_params))

def test():
    model = UNet()
    print_network(model)

    x = torch.rand(1, 2, 256, 256)
    from torch.autograd import Variable
    x = Variable(x)
    y = model(x)
    print y.size()



class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DisBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()  # nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return self.relu(out)

class Discriminator(nn.Module):
    def __init__(self, in_nc=1):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(in_nc, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.group1 = DisBlock(3, 16)       # 1/2
        self.group2 = DisBlock(16, 32)      # 1/4
        self.group3 = DisBlock(32, 64)      # 1/8
        self.group4 = DisBlock(64, 128)     # 1/16
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.linear = nn.Linear(128*4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.conv(img)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return self.sigmoid(out)

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, mean, std)
