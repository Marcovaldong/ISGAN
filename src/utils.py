import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ssim import SSIM
import os
import imageio
import random
from copy import deepcopy
from math import log10
import math

# For logger
def to_np(x):
    return x.data.cpu().numpy()

def to_var():
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# Denormalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def RGB2YUV(img):
    img = Image.fromarray(img, 'RGB')
    img = img.convert('YCbCr')
    return np.array(img)

def YUV2RGB(img):
    img = Image.fromarray(img, 'YCbCr')
    img = img.convert('RGB')
    return np.array(img)

def Y_UV2RGB(y, img):
    # print img[:, :, 0] == np.array(y)
    img[:, :, 0] = np.array(y)
    yuv = img
    img = Image.fromarray(yuv, 'YCbCr')
    img = img.convert('RGB')
    return img

transform = transforms.Compose([transforms.ToTensor()])
ssim_loss = SSIM(window_size=11).cuda()
mse_loss = nn.MSELoss().cuda()

def ssim_covers(stegos, imgs):
    imgs = imgs.numpy()
    stegos = to_np(stegos)
    for i in range(len(stegos)):
        img = stegos[i]
        img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
        stegos[i] = img
    origin = deepcopy(imgs)
    stegos = [Y_UV2RGB(stego, img) for stego, img in zip(stegos, origin)]
    covers = [Image.fromarray(YUV2RGB(img), 'RGB') for img in imgs]

    psnr = 0
    diff = 0.0
    for i in range(len(stegos)):
        mse = np.mean((np.array(stegos[i]) - np.array(covers[i])) ** 2)
        diff += np.mean(np.absolute((np.array(stegos[i]) - np.array(covers[i]))))
        if mse == 0:
            psnr += 100
        else:
            psnr += 20 * log10(255.0 / math.sqrt(mse))
    psnr /= len(stegos)
    diff /= len(stegos)

    stegos = [transform(stego) for stego in stegos]
    covers = [transform(cover) for cover in covers]
    for stego in stegos:
        stego.resize_(1, 3, 256, 256)
    for cover in covers:
        cover.resize_(1, 3, 256, 256)
    stegos = torch.cat(stegos, dim=0)
    covers = torch.cat(covers, dim=0)
    stegos = Variable(stegos.cuda())
    covers = Variable(covers.cuda())
    # mse = mse_loss(stegos, covers)
    # psnr = 10 * log10(1/mse.data[0])
    return ssim_loss(stegos, covers).data[0], psnr

def plotCovers(covers, stegos, imgs, epoch, save=True, save_dir='results/', show=False, fig_size=(5, 5), name='cover'):
    covers = to_np(covers)
    for i in range(len(covers)):
        img = covers[i]
        img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
        covers[i] = img
    stegos = to_np(stegos)
    for i in range(len(stegos)):
        img = stegos[i]
        img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
        stegos[i] = img
    covers = [Y_UV2RGB(cover, img) for cover, img in zip(covers, imgs)]
    stegos = [Y_UV2RGB(stego, img) for stego, img in zip(stegos, imgs)]
    # imgs = list(covers) + list(stegos)
    imgs = list()
    for i in range(len(covers)):
        imgs.append(covers[i])
        imgs.append(stegos[i])
    fig, axes = plt.subplots(len(covers), 2, figsize=fig_size)
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, title, ha='center')
    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + '{}_epoch_{:d}'.format(name, epoch) + '.png'
        plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()

# def saveStegos(imgs, stegos, fns, save_dir='./Data/ImageNet_Incpetion/'):
#     imgs = imgs.numpy()
#     stegos = to_np(stegos)
#     for i in range(len(stegos)):
#         img = stegos[i]
#         img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
#         stegos[i] = img
#     stegos = [Y_UV2RGB(stego, img) for stego, img in zip(stegos, imgs)]
#     for i in range(len(fns)):
#         fn = os.path.join(save_dir, fns[i].replace('.JPEG', '.bmp'))
#         fn = os.path.join(save_dir, fns[i].replace('.jpg', '.bmp'))
#         img = stegos[i]  # Image.fromarray(stegos[i])
#         img.save(fn)

def saveStegos(stegos, fns, save_dir='./'):
    stegos = to_np(stegos)
    stego = list()
    for i in range(len(stegos)):
        img = stegos[i]
        img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(np.uint8)
        stego.append(img)
    stegos = stego
    for i in range(len(fns)):
        fn = os.path.join(save_dir, fns[i].replace('.JPEG', '.bmp'))
        fn = os.path.join(save_dir, fns[i].replace('.jpg', '.bmp'))
        fn = os.path.join(save_dir, fns[i].replace('.png', '.bmp'))
        img = Image.fromarray(stegos[i], 'RGB')
        img.save(fn)

def plotSecrets(secrets, secretsRevealed, epoch, save=True, save_dir='results/', show=False,
                fig_size=(5, 5), name='secret'):
    secrets = to_np(secrets)
    secretsRevealed = to_np(secretsRevealed)
    # imgs = list(secrets) + list(secretsRevealed)
    imgs = list()
    for i in range(len(secrets)):
        imgs.append(secrets[i])
        imgs.append(secretsRevealed[i])
    fig, axes = plt.subplots(len(imgs) // 2, 2, figsize=fig_size)
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
        ax.imshow(img[0], cmap='gray', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, title, ha='center')
    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + '{}_epoch_{:d}'.format(name, epoch) + '.png'
        plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()

def plotCoverSecret(covers, stegos, secrets, secretsRevealed, epoch, save=True, save_dir='results/',
                    show=False, fig_size=(5, 5), name='performance'):
    stegos = to_np(stegos)
    stego = list()
    for i in range(len(stegos)):
        img = stegos[i]
        img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(np.uint8)
        stego.append(img)
    stegos = stego
    secrets = to_np(secrets)
    secretsRevealed = to_np(secretsRevealed)

    imgs = list()
    for i in range(len(covers)):
        imgs.append(covers[i])
        imgs.append(secrets[i])
        imgs.append(stegos[i])
        imgs.append(secretsRevealed[i])
    fig, axes = plt.subplots(len(imgs) // 4, 4, figsize=fig_size)
    idx = 0
    for ax, img in zip(axes.flatten(), imgs):
        idx += 1
        if idx % 2 == 1:
            ax.axis('off')
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            ax.axis('off')
            img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
            ax.imshow(img[0], cmap='gray', aspect='equal')
    title = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, title, ha='center')
    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + '{}_epoch_{:d}'.format(name, epoch) + '.png'
        plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()

def plot2imgs(imgs, epoch, save=False, save_dir='results/', show=False, fig_size=(5, 5), name='secret'):
    fig, axes = plt.subplots(len(imgs) // 2, 2, figsize=fig_size)

    # imgs = [to_np(secret), to_np(cover), to_np(stego), to_np(secret2)]
    imgs = [to_np(img) for img in imgs]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        # Scale to 0-255
        img = img.squeeze()
        img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    title = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + '{}_epoch_{:d}'.format(name, epoch) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

def plot_train_result(imgs, epoch, save=False, save_dir='results/', save_name='Result', show=False, fig_size=(5, 5)):
    fig, axes = plt.subplots(len(imgs) // 4, 4, figsize=fig_size)

    # imgs = [to_np(secret), to_np(cover), to_np(stego), to_np(secret2)]
    imgs = [to_np(img) for img in imgs]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        # Scale to 0-255
        img = img.squeeze()
        try:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        except:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
            ax.imshow(img, cmap='gray', aspect='equal')
        # ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    title = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + '{}_epoch_{:d}'.format(save_name, epoch) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

# Make gif
def make_gif(num_epochs, name, savename, save_dir='./results/'):
    gen_image_plots = []
    for epoch in range(1, num_epochs + 1):
        # plot for generating gif
        save_fn = save_dir + '{:s}_epoch_{:d}'.format(name, epoch) + '.png'
        gen_image_plots.append(imageio.imread(save_fn))
    imageio.mimsave(save_dir + '{:s}_epochs_{:d}'.format(savename, num_epochs) + '.gif', gen_image_plots, fps=5)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
            return_images = Variable(torch.cat(return_images, 0))
            return return_images

def plot_result(name, result):
    plt.figure()
    n = len(result[0])
    x = np.linspace(1, n, n)
    y1 = result[0]
    y2 = result[1]
    y3 = result[2]
    y4 = result[3]
    l1, = plt.plot(x, y1, label='line', color='red', linewidth=1.0)
    l3, = plt.plot(x, y2, label='line', color='blue', linewidth=1.0)
    l2, = plt.plot(x, y3, label='parabola', color='red', linewidth=1.0, linestyle='--')
    l4, = plt.plot(x, y4, label='parabola', color='blue', linewidth=1.0, linestyle='--')

    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.title('Google-{:s}'.format(name))
    plt.legend(handles=[l1, l2, l3, l4], labels=['train_encoder', 'train_decoder', 'val_encoder', 'val_decoder'], loc='best')
    plt.savefig('{:s}_result.jpg'.format(name))