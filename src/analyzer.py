import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
import utils
import argparse
import os, time, copy
import numpy as np
from XuNet import XuNet
import matplotlib.pyplot as plt
from ssim import SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='train batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=256, help='resize scale (0 is false')
parser.add_argument('--crop_size', type=int, default=0, help='crop size (0 if false)')
parser.add_argument('--fliplr', type=bool, default=False, help='random fliplr True or False')
parser.add_argument('--num_epochs', type=int, default=50, help='num of epoch to train')
parser.add_argument('--decay_epoch', type=int, default=15, help='start decaying learning rate after this number')
parser.add_argument('--log_schedule', type=int, default=20, help='number of batch size to save snapshot after')
parser.add_argument('--lambdaA', type=float, default=1, help='lambadA for encoder')
parser.add_argument('--lambdaB', type=float, default=0.75, help='lambdaB for decoder')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda or not')
params = parser.parse_args()
print(params)

model_dir = '/home/marcovaldo/Lab/YUV/Inception/analyzer/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolder1='cover', subfolder2='stego', transform=None):
        super(DatasetFromFolder, self).__init__()
        self.cover_paths = os.path.join(image_dir, subfolder1)
        self.stego_paths = os.path.join(image_dir, subfolder2)
        self.stego_fns = os.listdir(self.stego_paths)
        if 'lfw' in subfolder1:
            self.cover_fns = [fn.replace('.bmp', '.jpg') for fn in self.stego_fns]
        elif 'VOC' in image_dir:
            self.cover_fns = [fn.replace('.bmp', '.jpg') for fn in self.stego_fns]
        else:
            self.cover_fns = [fn.replace('.bmp', '.JPEG') for fn in self.stego_fns]
        self.cover_fns = [[fn, 1] for fn in self.cover_fns]
        self.stego_fns = [[fn, 2] for fn in self.stego_fns]
        # cover_labels = [1] * len(self.cover_fns)
        # stego_labels = [2] * len(self.stego_fns)
        # np.random.shuffle((self.stego_fns, cover_labels))
        # np.random.shuffle((self.cover_fns, stego_labels))
        self.fns = self.cover_fns + self.stego_fns
        np.random.shuffle(self.fns)
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.fns[index]
        if label == 1:
            fn = os.path.join(self.cover_paths, fn)
            img = Image.open(fn).convert('RGB')
        else:
            fn = os.path.join(self.stego_paths, fn)
            img = Image.open(fn)
        if label == 1:
            img = img.resize((params.input_size, params.input_size), Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        # print(fn, img.shape)
        return img, label-1

    def __len__(self):
        return len(self.fns)



class DatasetFromFolder2(data.Dataset):
    def __init__(self, image_dir='/home/marcovaldo/Data', cover_dir='ILSVRC2012_img_val',
                 stego_dir='UNet_ImageNet_stego', transform=None):
        super(DatasetFromFolder2, self).__init__()
        # image_dir = '/home/marcovaldo/Data'
        # cover_dir = 'ILSVRC2012_img_val'
        # stego_dir = 'Inception_ImageNet_stego'
        self.cover_dir = os.path.join(image_dir, cover_dir)
        self.stego_dir = os.path.join(image_dir, stego_dir)
        self.fns = os.listdir(self.stego_dir)
        self.transform = transform

    def __getitem__(self, index):
        fn = self.fns[index]
        stego = Image.open(os.path.join(self.stego_dir, fn))
        cover = Image.open(os.path.join(self.cover_dir, fn.replace('.png', '.JPEG'))).convert('RGB')
        cover = cover.resize((params.input_size, params.input_size), Image.BILINEAR)

        if self.transform is not None:
            cover = self.transform(cover)
            stego = self.transform(stego)
        return cover, stego

    def __len__(self):
        return len(self.fns)

transform = transforms.Compose([transforms.ToTensor()])
image_dir = '/home/marcovaldo/Data'  # '/home/marcovaldo/Data'
subfolder1 = 'lfw_images/'  # 'ILSVRC2012_img_val'
subfolder2 = 'lfw_stego/'  # 'Inception_ImageNet_stego'
dataloader = DatasetFromFolder(image_dir=image_dir, subfolder1=subfolder1,
                               subfolder2=subfolder2, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset=dataloader, batch_size=params.batch_size, shuffle=True)

valData = DatasetFromFolder(image_dir=image_dir, subfolder1=subfolder1,
                            subfolder2='lfw_stego_val', transform=transform)
valDataloader = torch.utils.data.DataLoader(dataset=valData, batch_size=16, shuffle=True)
testData = DatasetFromFolder(image_dir=image_dir, subfolder1=subfolder1,
                             subfolder2='lfw_stego_test_100', transform=transform)
testDataloader = torch.utils.data.DataLoader(dataset=testData, batch_size=16, shuffle=False)

model = XuNet()
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
fig1, ax1 = plt.subplots()
losses = list()

def train(epoch):
    model.train()
    correct, total = 0, 0
    if epoch > 15 and epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.9
    print('current learning rate: {:.6f}'.format(optimizer.param_groups[0]['lr']))
    for idx, (imgs, labels) in enumerate(dataloader, 1):
        imgs = imgs.cuda()
        labels = labels.cuda()
        imgs = Variable(imgs)
        labels = Variable(labels)

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

        losses.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if idx % 100 == 0:
        #     print('Epoch {}/{} step {}/{}, loss {:.6f}'.format(epoch, params.num_epochs, idx*params.batch_size, len(dataloader.dataset), loss.data[0]))
    print('Epoch %d Accuracy of the network on the 5000 train images: %d %%' % (epoch, 100 * correct / total))
    torch.save(model.state_dict(), model_dir + 'model_{}.pkl'.format(epoch))

def val(epoch):
    model.eval()
    correct, total = 0, 0
    for idx, (imgs, labels) in enumerate(valDataloader, 1):
        imgs = imgs.cuda()
        labels = labels.cuda()
        imgs = Variable(imgs)
        labels = Variable(labels)

        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('Accuracy of the network on the 2000 val images: %d %%' % (100 * correct / total))

def test(epoch):
    model.load_state_dict(torch.load(model_dir + 'analyzermodel_{}.pkl'.format(epoch)))
    model.eval()
    correct, total = 0, 0
    for idx, (imgs, labels) in enumerate(testDataloader, 1):
        imgs = imgs.cuda()
        labels = labels.cuda()
        imgs = Variable(imgs)
        labels = Variable(labels)

        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('Accuracy of the network on the 1000 test images: %f %%' % (100 * correct / float(total)))

def get_ssim():
    image_dir = '/home/marcovaldo/Data'
    cover_dir = 'ILSVRC2012_img_val'
    stego_dir = 'UNet_ImageNet_stego'

    ssim_criterion = SSIM(window_size=11).cuda()
    total_loss, idx = list(), 0
    dataloader2 = DatasetFromFolder2(image_dir=image_dir, cover_dir=cover_dir,
                                     stego_dir=stego_dir, transform=transform)
    dataloader2 = torch.utils.data.DataLoader(dataset=dataloader2, batch_size=params.batch_size, shuffle=False)
    for idx, (covers, stegos) in enumerate(dataloader2, 1):
        covers = Variable(covers.cuda())
        stegos = Variable(stegos.cuda())
        loss = ssim_criterion(covers, stegos)
        total_loss.append(loss.data[0])
        if idx % 1 == 0:
            print('{}/{} loss: {:.6f}'.format(idx*params.batch_size, len(dataloader2.dataset), loss.data[0]))
    print(sum(total_loss)/len(total_loss))

if __name__ == '__main__':
    # for epoch in range(1, params.num_epochs + 1):
    #     train(epoch)
    #     val(epoch)
    for epoch in range(63, 74):
        print(epoch)
        test(epoch)
    # get_ssim()