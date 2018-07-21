import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSampleBlock(nn.Module):
    def __init__(self, in_nc=2, nc=32):
        super(DownSampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, nc, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(nc)
        self.conv2 = nn.Conv2d(nc, nc*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(nc*2)
        self.conv3 = nn.Conv2d(nc*2, nc*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(nc*4)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.relu(self.norm3(self.conv3(out)))
        return out

class ICNetModule(nn.Module):
    def __init__(self, in_nc=128, out_nc=1):
        super(ICNetModule, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_nc, in_nc*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_nc*2)
        self.conv2 = nn.Conv2d(in_nc*2, in_nc*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(in_nc*2)
        self.conv3 = nn.Conv2d(in_nc, in_nc*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = nn.BatchNorm2d(in_nc*2)
        self.conv4 = nn.Conv2d(in_nc*2, in_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(in_nc)
        self.conv5 = nn.Conv2d(in_nc, in_nc*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm5 = nn.BatchNorm2d(in_nc*2)
        self.conv6 = nn.Conv2d(in_nc*2, in_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm6 = nn.BatchNorm2d(in_nc)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.predict1 = nn.Conv2d(in_nc*2, in_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.predict1_norm = nn.BatchNorm2d(in_nc)
        self.predict2 = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, f1, f2):
        f1 = self.norm1(self.relu(self.upsample(f1)))
        predict = self.relu(self.predict1_norm(self.predict1(f1)))
        predict = self.tanh(self.predict2(predict))
        f1 = self.norm2(self.relu(self.conv2(f1)))
        f2 = self.norm3(self.relu(self.conv3(f2)))
        fusion = f1 + f2
        fusion = self.relu(self.norm4(self.conv4(fusion)))
        fusion = self.relu(self.norm5(self.conv5(fusion)))
        fusion = self.relu(self.norm6(self.conv6(fusion)))
        return predict, fusion

class ICNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=128):
        super(ICNet, self).__init__()
        self.block1 = DownSampleBlock()
        self.dilatedConv1 = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.dilatednorm = nn.BatchNorm2d(nc)
        self.block2 = DownSampleBlock()
        self.block3 = DownSampleBlock()
        self.fusion1 = ICNetModule()
        self.fusion2 = ICNetModule()
        self.upsample1 = nn.ConvTranspose2d(nc, nc//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(nc//2)
        self.upsample2 = nn.ConvTranspose2d(nc//2, nc//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(nc//2)
        # self.pixelShuffle = nn.PixelShuffle(2)
        self.upsample3 = nn.ConvTranspose2d(nc//2, nc//4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(nc//4)
        self.conv1 = nn.Conv2d(nc//4, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, out_nc, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv3 = nn.Conv2d(nc//2, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm5 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.init_weight()

    def forward(self, x1, x2, x3):
        x1 = self.block1(x1)
        x1 = self.relu(self.dilatednorm(self.dilatedConv1(x1)))
        x2 = self.block2(x2)
        x3 = self.block3(x3)
        # print(x1.size(), x2.size())
        stego1, x2 = self.fusion1(x1, x2)
        stego2, x3 = self.fusion2(x2, x3)
        stego = self.relu(self.norm1(self.upsample1(x3)))
        stego3 = self.relu(self.norm5(self.conv3(stego)))
        stego3 = self.tanh(self.conv4(stego3))

        stego = self.relu(self.norm2(self.upsample2(stego)))
        # stego = self.pixelShuffle(stego)
        stego = self.relu(self.norm3(self.upsample3(stego)))
        stego = self.relu(self.norm4(self.conv1(stego)))
        stego4 = self.tanh(self.conv2(stego))

        return stego1, stego2, stego3, stego4


    # def forward(self, secret, cover1, cover2):
    #     secret = self.block1(secret)
    #     secret = self.relu(self.dilatednorm(self.dilatedConv1(secret)))
    #     cover1 = self.block2(cover1)
    #     cover2 = self.block3(cover2)
    #     stego = self.fusion1(secret, cover1)
    #     stego = self.fusion2(stego, cover2)
    #     stego = self.relu(self.norm1(self.upsample1(stego)))
    #     stego = self.relu(self.norm2(self.upsample2(stego)))
    #     # stego = self.pixelShuffle(stego)
    #     stego = self.relu(self.norm3(self.upsample3(stego)))
    #     stego = self.relu(self.norm4(self.conv1(stego)))
    #     stego = self.tanh(self.conv2(stego))
    #     return stego

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.xavier_uniform(m.weight)

class RevealNet(nn.Module):
    def __init__(self, nc=1, nhf=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256
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
            output_function()
        )

    def forward(self, input):
        output = self.main(input)
        return output

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.xavier_uniform(m.weight)




# class RevealNet(nn.Module):
#     def __init__(self, in_nc=1, out_nc=1):
#         super(RevealNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_nc, 16, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm3 = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.norm4 = nn.BatchNorm2d(64)
#         self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         self.upnorm1 = nn.BatchNorm2d(32)
#         self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         self.upnorm2 = nn.BatchNorm2d(16)
#         self.upsample3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         self.upnorm3 = nn.BatchNorm2d(8)
#         self.conv5 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False)
#         self.norm5 = nn.BatchNorm2d(3)
#         self.conv6 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU(True)
#         self.tanh = nn.Tanh()
#         self.init_weight()
#
#     def forward(self, stego):
#         secret = self.relu(self.norm1(self.conv1(stego)))
#         secret = self.relu(self.norm2(self.conv2(secret)))
#         secret = self.relu(self.norm3(self.conv3(secret)))
#         secret = self.relu(self.norm4(self.conv4(secret)))
#         secret = self.relu(self.upnorm1(self.upsample1(secret)))
#         secret = self.relu(self.upnorm2(self.upsample2(secret)))
#         secret = self.relu(self.upnorm3(self.upsample3(secret)))
#         secret = self.relu(self.norm5(self.conv5(secret)))
#         secret = self.relu(self.conv6(secret))
#         return secret
#
#     def init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform(m.weight)
#             elif isinstance(m, nn.ConvTranspose2d):
#                 nn.init.xavier_uniform(m.weight)
#             # elif isinstance(m, nn.BatchNorm2d):
#             #     nn.init.xavier_uniform(m.weight)
#
