import random
import numpy as np
import cv2.cv2 as cv2
import torch.utils.data as data
from data import common
from .tools import get_circle_matrix


class Dataset(data.Dataset):
    def __init__(self, args, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.mode = mode

        self.rotational = args.rotational
        self.height = args.rotational_height
        self.width = args.rotational_width
        self.rotational_patch_size = args.patch_size
        self.noise_level = args.noise_level

        self.step_range = args.rotational_test_step

        self.testangle = 3.0

        center = (int(self.height / 2), int(self.width / 2))
        circle_dic, radius, hole_matrix = get_circle_matrix((self.height, self.width), center=center)
        self.circle_dic = circle_dic
        self.radius = radius
        self.hole_matrix = hole_matrix

    def set_teststep(self, testangle):
        self.testangle = testangle
        return

    def __getitem__(self, idx):

        pad_width = 0
        if self.mode == 'train':
            blurred_path = self.datasets['train'][idx]
            sharp_path = blurred_path.replace('/blur/', '/sharp/')

            blurred = cv2.imread(blurred_path, -1)
            sharp = cv2.imread(sharp_path, -1)
            imgs = [blurred, sharp]
            imgs = common.chose_part(*imgs, ps=self.rotational_patch_size)
            if self.args.augment:
                imgs = common.augment(*imgs, hflip=True, rot=True, shuffle=True, change_saturation=False,
                                      rgb_range=self.args.rgb_range)

        else:
            step = self.testangle
            blurred_path = self.datasets[step][idx]
            sharp_path = blurred_path.replace('/blur/', '/sharp/')
            blurred = cv2.imread(blurred_path)
            sharp = cv2.imread(sharp_path)
            imgs = [blurred, sharp]

        if self.args.gaussian_pyramid:
            imgs = common.generate_pyramid(*imgs, n_scales=self.args.n_scales)

        imgs = common.np2tensor(*imgs)
        relpath = blurred_path.split('/')[-1].split('start')[0] + '.bmp'
        blur = imgs[0]
        sharp = imgs[1] if len(imgs) > 1 else False
        blur_field = 1

        return blur, sharp, pad_width, blur_field, idx, relpath

    def __len__(self):
        if self.mode == 'train':
            return len(self.datasets[self.mode])
        else:
            return len(self.datasets[self.step_range[0]])