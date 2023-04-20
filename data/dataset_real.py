import random
import numpy as np
import cv2.cv2 as cv2
import torch.utils.data as data
import copy
from data import common
from .tools import get_circle_matrix
from utils import interact

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
        self.step_range = [70, 85, 100, 110, 125, 140]
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

            [blurred, sharp] = common.augment_for_real(*[blurred, sharp])
            std = random.uniform(0.01, 0.05)
            blurred = common.add_noise(blurred, std=std)
            
            imgs = [blurred, sharp]
            imgs = common.chose_part(*imgs, ps=self.rotational_patch_size)

        else:
            step = self.testangle
            blurred_path = self.datasets[step][idx]
            sharp_path = blurred_path.replace('/blur/', '/sharp/')
            blurred = cv2.imread(blurred_path)
            sharp = cv2.imread(sharp_path)
            imgs = [blurred, sharp]

            # cv2.imwrite('./blurred.bmp', imgs[0].astype(np.uint8))
            # cv2.imwrite('./sharp.bmp', imgs[1].astype(np.uint8))

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