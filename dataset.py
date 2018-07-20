from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import random
from copy import deepcopy

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolder='train', transform=None, resize_scale=None, crop_size=None,
                 fliplr=False, name=False):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        image_names = os.listdir(self.input_path)
        random.shuffle(image_names)
        self.secret_images = image_names[:len(image_names)//2]
        self.cover_images = image_names[len(image_names)//2:]
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.name = name

    def __getitem__(self, index):
        # Load Image
        secret_fn = os.path.join(self.input_path, self.secret_images[index])
        cover_fn = os.path.join(self.input_path, self.cover_images[index])
        secret = Image.open(secret_fn).convert('L')
        cover = Image.open(cover_fn).convert('RGB')
        target = Image.open(cover_fn).convert('RGB')

        # preprocessing
        if self.resize_scale:
            secret = secret.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            cover = cover.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            target = target.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        # if self.crop_size:
        #     x = random.randint(0, self.resize_scale - self.crop_size + 1)
        #     y = random.randint(0, self.resize_scale - self.crop_size + 1)
        #     secret = secret.crop(x, y, x + self.crop_size, y + self.crop_size)
        #     cover = cover.crop(x, y, x + self.crop_size, y + self.crop_size)
        #
        #
        #
        # if self.fliplr:
        #     if random.random() < 0.5:
        #         secret = secret.transpose(Image.FLIP_LEFT_RIGHT)
        #         cover = cover.transpose(Image.FLIP_LEFT_RIGHT)

        # target = deepcopy(cover)
        if self.transform is not None:
            secret = self.transform(secret)
            cover = self.transform(cover)
            target = self.transform(target)
        # print(secret.size(), cover.size(), target.size())
        if not self.name:
            return secret, cover, target
        else:
            return secret, cover, target, self.cover_images[index]

    def __len__(self):
        return len(self.secret_images)

class DatasetFromFolder2(data.Dataset):
    def __init__(self, image_dir, subfolder='train.txt', transform=None, resize_scale=None, crop_size=None,
                 fliplr=False, name=False):
        super(DatasetFromFolder2, self).__init__()
        self.input_path = image_dir  # os.path.join(image_dir, subfolder)
        image_names = list()
        with open(os.path.join(image_dir, subfolder), 'r') as f:
            image_names = f.readlines()
        image_names = [image_name.strip() for image_name in image_names]
        random.shuffle(image_names)
        self.secret_images = image_names[:len(image_names)//2]
        self.cover_images = image_names[len(image_names)//2:]
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.name = name

    def __getitem__(self, index):
        # Load Image
        secret_fn = os.path.join(self.input_path, self.secret_images[index])
        cover_fn = os.path.join(self.input_path, self.cover_images[index])
        secret = Image.open(secret_fn).convert('L')
        cover = Image.open(cover_fn).convert('RGB')
        target = Image.open(cover_fn).convert('RGB')

        # preprocessing
        if self.resize_scale:
            secret = secret.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            cover = cover.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            target = target.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        # if self.crop_size:
        #     x = random.randint(0, self.resize_scale - self.crop_size + 1)
        #     y = random.randint(0, self.resize_scale - self.crop_size + 1)
        #     secret = secret.crop(x, y, x + self.crop_size, y + self.crop_size)
        #     cover = cover.crop(x, y, x + self.crop_size, y + self.crop_size)
        #
        #
        # if self.fliplr:
        #     if random.random() < 0.5:
        #         secret = secret.transpose(Image.FLIP_LEFT_RIGHT)
        #         cover = cover.transpose(Image.FLIP_LEFT_RIGHT)
        # # cover = np.array(cover.convert('YCbCr'))
        # # cover, cover_uv = Image.fromarray(cover[:, :, 0], 'L'), cover[:, :, 1:]

        if self.transform is not None:
            secret = self.transform(secret)
            cover = self.transform(cover)
            target = self.transform(target)
        # print(secret.size(), cover.size())

        if not self.name:
            return secret, cover, target
        else:
            return secret, cover, target, self.cover_images[index].split('/')[-1]

    def __len__(self):
        return len(self.secret_images)

def getCoverExample(path, img_size):
    img = Image.open(path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    target = deepcopy(img)
    return img, target

def getSecretExample(path, img_size):
    img = Image.open(path).convert('L')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return img

# class DatasetFromFolder3(data.Dataset):
#     def __init__(self, image_dir, subfolder='train', transform=None, resize_scale=None, crop_size=None, fliplr=False):
#         super(DatasetFromFolder3, self).__init__()
#         self.input_path = os.path.join(image_dir, subfolder)
#         image_names = os.listdir(self.input_path)
#         random.shuffle(image_names)
#         self.secret_images = image_names[:len(image_names)//2]
#         self.cover_images = image_names[len(image_names)//2:]
#         self.transform = transform
#         self.resize_scale = resize_scale
#         self.crop_size = crop_size
#         self.fliplr = fliplr
#
#     def __getitem__(self, index):
#         # Load Image
#         secret_fn = os.path.join(self.input_path, self.secret_images[index])
#         cover_fn = os.path.join(self.input_path, self.cover_images[index])
#         secret = Image.open(secret_fn).convert('L')
#         cover = Image.open(cover_fn).convert('L')
#         rgb = Image.open(cover_fn)
#
#         # preprocessing
#         if self.resize_scale:
#             secret = secret.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
#             cover = cover.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
#             rgb = rgb.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
#
#         if self.crop_size:
#             x = random.randint(0, self.resize_scale - self.crop_size + 1)
#             y = random.randint(0, self.resize_scale - self.crop_size + 1)
#             secret = secret.crop(x, y, x + self.crop_size, y + self.crop_size)
#             cover = cover.crop(x, y, x + self.crop_size, y + self.crop_size)
#
#
#         if self.fliplr:
#             if random.random() < 0.5:
#                 secret = secret.transpose(Image.FLIP_LEFT_RIGHT)
#                 cover = cover.transpose(Image.FLIP_LEFT_RIGHT)
#         # cover = np.array(cover.convert('YCbCr'))
#         # cover, cover_uv = Image.fromarray(cover[:, :, 0], 'L'), cover[:, :, 1:]
#
#         if self.transform is not None:
#             secret = self.transform(secret)
#             cover = self.transform(cover)
#         # print(secret.size(), cover.size())
#         return secret, cover, np.array(rgb.convert('YCbCr')), self.cover_images[index]
#
#     def __len__(self):
#         return len(self.secret_images)

