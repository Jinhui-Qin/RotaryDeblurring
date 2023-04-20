import numpy as np
import math
from skimage import metrics
import skimage
import pickle
import os
from numpy import fft
import copy


def save_obj(obj, name: str, save_root=None):
    if save_root==None:
        save_root = './'
    with open(save_root + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name: str, load_root=None):
    if load_root==None:
        load_root = './'
    with open(load_root + name + '.pkl', 'rb') as f:
        return pickle.load(f)


para_default = {
    'reg_D12' :{
        'gamma1': 0.05, 'gamma2': 0.03, 'weight_min': 0.1, 'weight_max': 1.5, 'weight': 2.5
    }
}


def weight_r(r, w, be_min=5.04 * 5, be_max=10.08 * 160, para_dic=None):

    weight_min = para_dic['weight_min']
    weight_max = para_dic['weight_max']

    if r <= be_min:
        return 1.0

    weight = (r - be_min) / (be_max - be_min) * (weight_max - weight_min) + weight_min
    weight = weight ** w

    return weight


def reg_D12(g, length_blur_path, para_dic=None, r=1, weight=-1.0):

    gamma1 = para_dic['gamma1']
    gamma2 = para_dic['gamma2']

    if weight >= 0:
        gamma1 = weight_r(r, weight, para_dic=para_dic) * gamma1
        gamma2 = weight_r(r, weight, para_dic=para_dic) * gamma2

    h = [1 / length_blur_path] * length_blur_path + (len(g) - length_blur_path) * [0]
    d1 = [-1] + [0] * (len(g) - 2) + [1]
    d2 = [-2, 1] + [0] * (len(g) - 3) + [1]

    G = fft.fft(g)
    H = fft.fft(h)
    D1 = fft.fft(d1)
    D2 = fft.fft(d2)

    B = np.abs(H) ** 2 + gamma1 * np.abs(D1) ** 2 + gamma2 * np.abs(D2) ** 2
    A = np.conj(H) * G

    F = A / B
    f = np.abs(fft.ifft(F))

    return f


def blur_1d(f, h):

    H = fft.fft(h)
    F = fft.fft(f)
    G = H * F
    g = fft.ifft(G)
    g = np.abs(g)

    return g


def bresenham(R):

    assert R >= 1

    circle = []

    x = 0
    y = R
    delta = 2 * (1 - R)

    while y >= 0:

        circle.append([y, x])

        if delta < 0:
            delta1 = 2 * (delta + y) - 1
            if(delta1 <= 0):
                direction = 'H'
            else:
                direction = 'D'
        elif delta > 0:
            delta2 = 2 * (delta - x) - 1
            if (delta2 <= 0):
                direction = 'D'
            else:
                direction = 'V'
        else:
            direction = 'D'

        if direction == 'H':
            x = x + 1
            delta += 2 * x + 1
        elif direction == 'D':
            x = x + 1
            y = y - 1
            delta += 2 * x -2 * y + 2
        else:
            y = y - 1
            delta += -2 * y + 1

    circle_1 = []
    for point in copy.deepcopy(circle):
        p1 = point[0]
        p2 = -1 * point[1]
        circle_1.append([p1, p2])

    circle_tmp = copy.deepcopy(circle)[::-1]
    del(circle_tmp[-1])
    res = circle_tmp + circle_1

    circle_2 = []
    for point in copy.deepcopy(res):
        p1 = -1 * point[0]
        p2 = point[1]
        circle_2.append([p1, p2])

    res_tmp = copy.deepcopy(res)
    del(res_tmp[-1])
    res = res_tmp + circle_2[::-1]
    del(res[-1])

    return res


def get_bresenham_circle_dic(size, mode='bresenham'):

    if os.path.exists('./CircleDic_height{}_width{}_mode{}.pkl'.format(size[0], size[1], mode)):
        circle_dic = load_obj('CircleDic_height{}_width{}_mode{}'.format(size[0], size[1], mode))
        if circle_dic[-1][0] == size[0] and circle_dic[-1][1] == size[1]:
            print('Using CircleDic_height{}_width{}_mode{}.pkl'.format(size[0], size[1], mode))
            return circle_dic
        else:
            circle_dic = {}
            circle_dic[-1] = [size[0], size[1]]
    else:
        circle_dic = {}
        circle_dic[-1] = [size[0], size[1]]

    print('--------------- Creating circle_dic! ---------------')
    rmax = int(np.sqrt((size[0] / 2)**2 + (size[1] / 2)**2))

    for r in range(1, rmax, 1):
        circle = bresenham(r)

        circle_new = []
        if (r > int(np.min(size) / 2) and r <= int(np.max(size) / 2)):
            for coo in circle:
                if coo[1] < 0:
                    circle_new.append(coo)
            circle_dic[r] = circle_new
        elif r > int(np.max(size) / 2):
            for coo in circle:
                if coo[0] >= 0 and coo[1] >= 0:
                    circle_new.append(coo)
            circle_dic[r] = circle_new
        else:
            circle_dic[r] = circle

    save_obj(circle_dic, 'CircleDic_height{}_width{}_mode{}'.format(size[0], size[1], mode))

    print('--------------- circle_dic Done! ---------------')
    return circle_dic


def get_para_circle_dic(size, mode='para'):

    if os.path.exists('./CircleDic_height{}_width{}_mode{}.pkl'.format(size[0], size[1], mode)):
        circle_dic = load_obj('CircleDic_height{}_width{}_mode{}'.format(size[0], size[1], mode))
        if circle_dic[-1][0] == size[0] and circle_dic[-1][1] == size[1]:
            print('Using CircleDic_height{}_width{}_mode{}.pkl'.format(size[0], size[1], mode))
            return circle_dic
        else:
            circle_dic = {}
            circle_dic[-1] = [size[0], size[1]]
    else:
        circle_dic = {}
        circle_dic[-1] = [size[0], size[1]]

    print('--------------- Creating circle_dic! ---------------')
    rmax = int(np.sqrt((size[0] / 2)**2 + (size[1] / 2)**2))

    for r in range(1, rmax, 1):

        circle = []
        N = int((2 * r + 1) * np.pi / 2 + 1) * 6
        deltatheta = 2 * np.pi / N

        x_ = 0
        y_ = 0

        for i in range(N):
            theta = i * deltatheta
            y = int(r * math.sin(theta))
            x = int(r * math.cos(theta))

            if(r > int(np.min(size) / 2) and r <= int(np.max(size) / 2)):
                if(not(i >= int(N / 4) and (i <= 3 * int(N / 4)))):
                    continue
            if(r > int(np.max(size) / 2)):
                if(i > int(N / 4)):
                    continue

            if (x_ == x) & (y_ == y):
                continue
            else:
                circle.append([y, x])
                x_ = x
                y_ = y

        circle_dic[r] = circle

    save_obj(circle_dic, 'CircleDic_height{}_width{}_mode{}'.format(size[0], size[1], mode))

    print('--------------- circle_dic Done! ---------------')
    return circle_dic


def get_circle_matrix(size=(320, 640), center=(160, 320)):

    circle_dic = get_bresenham_circle_dic(size)
    CenterY = center[0]
    CenterX = center[1]
    new_circle_dic = []
    radius_matrix = []
    hole_matrix = np.ones(size) * -1

    def _get_new_circle(circle, symmetry=None, reverse=False):

        for point in circle:
            if(symmetry == 'y'):
                point[1] = -1 * point[1]
            elif(symmetry == 'x'):
                point[0] = -1 * point[0]
            elif(symmetry == 'yx' or symmetry == 'xy'):
                point[1] = -1 * point[1]
                point[0] = -1 * point[0]

        if(reverse):
            circle.reverse()

        new_circle = []
        for point in circle:
            point[0] = CenterY - point[0]
            point[1] = CenterX + point[1]
            if((point[1] < 0) or (point[1] > size[1] - 1) or (point[0] < 0) or (point[0] > size[0] - 1)):
                continue
            else:
                new_circle.append([point[0], point[1]])

        return new_circle

    for r in circle_dic.keys():

        if r < 0:
            continue

        circle = circle_dic[r]
        if(r <= int(np.min(size) / 2)):
            new_circle = _get_new_circle(copy.deepcopy(circle))
            new_circle_dic.append(new_circle)
            radius_matrix.append(r)
        if(r > int(np.min(size) / 2) and r <= int(np.max(size) / 2)):
            new_circle = _get_new_circle(copy.deepcopy(circle))
            new_circle_dic.append(new_circle)
            radius_matrix.append(r)
            new_circle = _get_new_circle(copy.deepcopy(circle), symmetry='y', reverse=True)
            new_circle_dic.append(new_circle)
            radius_matrix.append(r)
        if(r > int(np.max(size) / 2)):
            new_circle = _get_new_circle(copy.deepcopy(circle))
            new_circle_dic.append(new_circle)
            radius_matrix.append(r)
            new_circle = _get_new_circle(copy.deepcopy(circle), symmetry='y', reverse=True)
            new_circle_dic.append(new_circle)
            radius_matrix.append(r)
            new_circle = _get_new_circle(copy.deepcopy(circle), symmetry='yx')
            new_circle_dic.append(new_circle)
            radius_matrix.append(r)
            new_circle = _get_new_circle(copy.deepcopy(circle), symmetry='x', reverse=True)
            new_circle_dic.append(new_circle)
            radius_matrix.append(r)

    for c in new_circle_dic:
        for coo in c:
            hole_matrix[coo[0], coo[1]] = 1

    return new_circle_dic, radius_matrix, hole_matrix


def get_blur(image, circle_dic, radius, theta=5.0):

    assert image.shape[0] <= image.shape[1]

    new_axis = True if len(image.shape) == 2 else False
    image = np.expand_dims(image, 2) if new_axis else image

    blurred = np.ones(image.shape) * -1

    for r in range(len(circle_dic)):

        padding = True if r > int(min(image.shape[0], image.shape[1]) / 2) else False

        circle = circle_dic[r]
        length_blur_path = int(theta / 180 * np.pi * radius[r])

        for channel in range(image.shape[2]):

            if length_blur_path == 0:
                for i in range(len(circle)):
                    blurred[circle[i][0], circle[i][1], channel] = image[circle[i][0], circle[i][1], channel]
                continue

            f = []
            for c in circle:
                f.append(image[c[0], c[1], channel])

            if padding:
                f = np.pad(f, (length_blur_path, 0), mode='symmetric')

            h = [1 / length_blur_path] * length_blur_path + (len(f) - length_blur_path) * [0]
            blurred_1d = blur_1d(f, h)

            if padding:
                for i in range(len(circle)):
                    blurred[circle[i][0], circle[i][1], channel] = blurred_1d[i + length_blur_path]
            else:
                for i in range(len(circle)):
                    blurred[circle[i][0], circle[i][1], channel] = blurred_1d[i]

    blurred = np.squeeze(blurred, 2) if new_axis else blurred
    return blurred


def interpolate(image, hole_matrix, r=1):
    new_axis = True if len(image.shape) == 2 else False
    image = np.expand_dims(image, 2) if new_axis else image
    for channel in range(image.shape[2]):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if hole_matrix[y, x] < 0:
                    tmp = []
                    for y_kernel in range(-1 * r, r + 1):
                        for x_kernel in range(-1 * r, r + 1):
                            try:
                                if hole_matrix[y_kernel + y, x_kernel + x] > 0:
                                    tmp.append(image[y_kernel + y, x_kernel + x, channel])
                                else:
                                    continue
                            except:
                                continue
                    if tmp != []:
                        image[y, x, channel] = np.mean(tmp)
    image = np.squeeze(image, 2) if new_axis else image

    return image


def add_noise(image, mean=0, std=0.01, rgb_range=255.0):
    # noise level : std -> var ** 0.5

    image = image.astype(np.float) / rgb_range
    noisy = skimage.util.random_noise(image, mode='gaussian', mean=mean, var=std ** 2)
    noisy = (noisy * rgb_range).clip(0, rgb_range).astype(np.uint8)

    return noisy


extension_list = ['bmp', 'png', 'jpg', 'pkl', 'npy']
def scan_file(root=None):
    data_list = []
    for sub, dirs, files in os.walk(root):
        if not dirs:
            file_list = []
            for f in files:
                if f.split('.')[-1] in extension_list:
                    file_list.append(os.path.join(sub, f))
            data_list += file_list

    return data_list


def get_deblur(image, circle_dic, radius, theta=5.0, method=reg_D12):

    assert image.shape[0] <= image.shape[1]
    new_axis = True if len(image.shape) == 2 else False
    image = np.expand_dims(image, 2) if new_axis else image

    deblurred = np.ones(image.shape) * -1

    for r in range(len(circle_dic)):

        if r > int(min(image.shape[0], image.shape[1]) / 2):
            continue

        padding = True if r > int(min(image.shape[0], image.shape[1]) / 2) else False

        circle = circle_dic[r]
        length_blur_path = int(theta / 180 * np.pi * radius[r])

        for channel in range(image.shape[2]):
            if length_blur_path == 0:
                for i in range(len(circle)):
                    deblurred[circle[i][0], circle[i][1], channel] = image[circle[i][0], circle[i][1], channel]
                continue

            g = []
            for c in circle:
                g.append(image[c[0], c[1], channel])

            if padding:
                g = np.pad(g, (length_blur_path, 0), mode='symmetric')

            r_be = r * theta
            deblurred_1d = method(g, length_blur_path, para_dic=para_default[method.__name__], r=r_be, weight=2.5)

            if padding:
                for i in range(len(circle)):
                    deblurred[circle[i][0], circle[i][1], channel] = deblurred_1d[i + length_blur_path]
            else:
                for i in range(len(circle)):
                    deblurred[circle[i][0], circle[i][1], channel] = deblurred_1d[i]

    deblurred = np.squeeze(deblurred, 2) if new_axis else deblurred

    return deblurred


def get_containedpoints(size):

    containedpoints = np.zeros(size)
    r = int(size[0] / 2) - 1

    for y in range(containedpoints.shape[0]):
        for x in range(containedpoints.shape[1]):
            if math.sqrt((y - int(size[0] / 2))**2 + (x - int(size[0] / 2))**2) < r:
                containedpoints[y, x] = 1.0

    return containedpoints


def get_metric(image, target):
    img_psnr = metrics.peak_signal_noise_ratio(image, target)
    img_ssim = metrics.structural_similarity(image, target, multichannel=True, \
                                            gaussian_weights=True, use_sample_covariance=False)
    return [img_psnr, img_ssim]


def crop(image):

    h, w = image.shape[0], image.shape[1]

    height = int(h * math.sin(45 / 180 * np.pi)) - 1
    if height % 2 == 1:
        height = height - 1
    start = int((h - height) / 2)

    if image.ndim == 2:
        return image[start:start+height, start:start+height]
    else:
        return image[start:start + height, start:start + height, :]
