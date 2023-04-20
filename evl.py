import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
torch.cuda.set_device(0)
from data.tools import  get_circle_matrix, get_blur, get_deblur, interpolate, add_noise, get_containedpoints
import cv2, copy, argparse
from option import args, setup, cleanup
from model.DeformResNet import build_model
import numpy as np
from skimage.transform import pyramid_gaussian
from data.common import np2tensor_evl


class evl():
    def __init__(self, path_model='', img_size=320):
        self.img_size = img_size
        self.containedpoints = get_containedpoints((self.img_size, self.img_size))
        self.model = build_model(setup(args))
        self.model.load_state_dict(torch.load(path_model)['G'])
        self.model.eval()
        self.model.cuda()

    def __call__(self, img=None):
        img =  img.astype(np.float)
        img = list(pyramid_gaussian(img, 2, multichannel=True))
        img = np2tensor_evl(*img)

        res = self.model(img)

        res = res[0].cpu().detach().squeeze(0).numpy()
        res = np.transpose(res, (1, 2, 0))
        res = res.clip(0, 255.0).astype(np.uint8)
        return res


def deblur_sny(path_model=''):
    theta = 10.08
    path_sharp = './imgs/3096.jpg'
    sharp = cv2.imread(path_sharp, -1)

    if sharp.shape[0] == 481:
        sharp = sharp[81:401, 1:, :]
    else:
        sharp = sharp[1:, 81:401, :]

    deblurer = evl(path_model=path_model)

    circle_dic, radius, hole_matrix = get_circle_matrix((320, 320), center=(160, 160))
    img_blurred = get_blur(copy.deepcopy(sharp), circle_dic, radius, theta=theta)
    img_blurred = add_noise(img_blurred, std=0.01)
    first_stage = get_deblur(copy.deepcopy(img_blurred), circle_dic, radius, theta=theta)
    first_stage = interpolate(first_stage, hole_matrix, r=1)
    first_stage[deblurer.containedpoints==0] = 128
    first_stage = first_stage.clip(0, 255).astype(np.uint8)

    res = deblurer(img=first_stage)
    cv2.imwrite('./imgs/3096_deblurred.bmp', res)
    return None


def deblur_real(path_model=''):

    path_blurred = './imgs/word4_10.08.bmp'
    img_blurred = cv2.imread(path_blurred, -1)
    deblurer = evl(path_model=path_model)
    circle_dic, radius, hole_matrix = get_circle_matrix((320, 320), center=(160, 160))
    img_blurred = add_noise(img_blurred, std=0.01)
    first_stage = get_deblur(copy.deepcopy(img_blurred), circle_dic, radius, theta=10.08)
    first_stage = interpolate(first_stage, hole_matrix, r=1)
    first_stage[deblurer.containedpoints==0] = 128
    first_stage = first_stage.clip(0, 255).astype(np.uint8)

    res = deblurer(img=first_stage)
    cv2.imwrite('./imgs/word4_10.08_deblurred.bmp', res)
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='syn', help='syn or real')
    parser.add_argument('--path_model', type=str, default='./experiment/baseline/models/model-250.pt')
    opt = parser.parse_args()

    if opt.dataset=='syn':
        deblur_sny(path_model=opt.path_model)
    elif opt.dataset=='real':
        deblur_real(path_model=opt.path_model)
    else:
        print('--dataset should be syn or real!')