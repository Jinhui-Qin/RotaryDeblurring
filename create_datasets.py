import argparse, cv2, os, copy
import numpy as np
from data.tools import get_blur, get_deblur, interpolate, scan_file
from data.tools import get_circle_matrix, add_noise, get_containedpoints
from tqdm import tqdm


step_dic = {
    'test': [70, 85, 100, 110, 125, 140],
    'train': [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]
}


def create_BSDS(data_root, save_root, phase ='train'):

    img_resolution = 320
    dtheta = 0.072
    step_list = step_dic[phase]
    containedpoints = get_containedpoints((img_resolution, img_resolution))
    containedpoints = np.tile(np.expand_dims(containedpoints, 2), (1, 1, 3))
    save_blrred_pure = os.path.join(save_root, phase, 'blur')
    save_sharp_pure = os.path.join(save_root, phase, 'sharp')
    os.makedirs(save_blrred_pure, exist_ok=True)
    os.makedirs(save_sharp_pure, exist_ok=True)
    bsds_list = scan_file(root=os.path.join(data_root, phase))
    center = (int(img_resolution / 2), int(img_resolution / 2))
    circle_dic, radius, hole_matrix = get_circle_matrix((img_resolution, img_resolution), center=center)

    for i in tqdm(range(len(bsds_list))):

        sharp_path_img = bsds_list[i]
        name_image = sharp_path_img.split('/')[-1].split('.')[0]
        sharp = cv2.imread(sharp_path_img, -1)

        if sharp.shape[0] == 481:
            sharp = sharp[81:401, 1:, :]
        else:
            sharp = sharp[1:, 81:401, :]

        for step in step_list:
            img_blurred = get_blur(copy.deepcopy(sharp), circle_dic, radius, theta=step*dtheta)
            img_blurred = add_noise(img_blurred, std=0.01)
            first_stage = get_deblur(copy.deepcopy(img_blurred), circle_dic, radius, theta=(step*dtheta))
            first_stage = interpolate(first_stage, hole_matrix, r=1)
            first_stage[containedpoints==0] = 128
            first_stage = first_stage.clip(0, 255).astype(np.uint8)

            sharp_save = copy.deepcopy(sharp)
            sharp_save[containedpoints==0] = 128
            sharp_save = sharp_save.clip(0, 255).astype(np.uint8)
            save_name = name_image + '_{}'.format(round(step*dtheta, 2)) + '_step_{}_.bmp'.format(step)

            cv2.imwrite(os.path.join(save_sharp_pure, save_name), sharp_save)
            cv2.imwrite(os.path.join(save_blrred_pure, save_name), first_stage)

    return None


def create_real(data_root, save_root, phase ='train'):

    img_resolution = 320
    dtheta = 0.072
    containedpoints = get_containedpoints((img_resolution, img_resolution))
    containedpoints = np.tile(np.expand_dims(containedpoints, 2), (1, 1, 3))
    save_blrred_pure = os.path.join(save_root, phase, 'blur')
    save_sharp_pure = os.path.join(save_root, phase, 'sharp')
    os.makedirs(save_blrred_pure, exist_ok=True)
    os.makedirs(save_sharp_pure, exist_ok=True)
    real_list = scan_file(root=os.path.join(data_root, phase, 'blur'))
    center = (int(img_resolution / 2), int(img_resolution / 2))
    circle_dic, radius, hole_matrix = get_circle_matrix((img_resolution, img_resolution), center=center)

    for i in tqdm(range(len(real_list))):

        # /data1/hunnzi/DataSets/rotary_real/real_s/test/blur/bucket1/Theta5.04_start1_to71_total_70.bmp

        blur_path_img = real_list[i]
        sharp_path_img = blur_path_img.replace('/blur/', '/sharp/')
        img_blurred = cv2.imread(blur_path_img, -1)
        sharp = cv2.imread(sharp_path_img, -1)

        step = float(blur_path_img.split('total_')[-1].split('.b')[0])
        name_image = blur_path_img.split('/')[-2] +  blur_path_img.split('/')[-1]
        img_blurred = add_noise(img_blurred, std=0.01)
        first_stage = get_deblur(copy.deepcopy(img_blurred), circle_dic, radius, theta=(step*dtheta))
        first_stage = interpolate(first_stage, hole_matrix, r=1)
        first_stage[containedpoints==0] = 128
        first_stage = first_stage.clip(0, 255).astype(np.uint8)

        sharp_save = copy.deepcopy(sharp)
        sharp_save[containedpoints==0] = 128
        sharp_save = sharp_save.clip(0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_sharp_pure, name_image), sharp_save)
        cv2.imwrite(os.path.join(save_blrred_pure, name_image), first_stage)

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='real', help='syn or real')
    parser.add_argument('--data_root', type=str, default='/data1/hunnzi/DataSets/rotary_real/real_s/')
    parser.add_argument('--save_root', type=str, default='/data1/hunnzi/DataSets/real_world_test/')
    opt = parser.parse_args()

    # if opt.dataset == 'syn':
    #     print('---------- Creating synthetic rotary motion blur datasets ----------\n')
    #     print('---------- Creating training datasets ----------\n')
    #     create_BSDS(data_root=opt.data_root, save_root=opt.save_root, phase='train')
    #     print('---------- Creating testing datasets ----------\n')
    #     create_BSDS(data_root=opt.data_root, save_root=opt.save_root, phase='test')
    # elif opt.dataset == 'real':
    #     print('---------- Creating real-world rotary motion blur datasets ----------\n')
    #     print('---------- Creating training datasets ----------\n')
    #     create_real(data_root=opt.data_root, save_root=opt.save_root, phase='test')
    #     print('---------- Creating testing datasets ----------\n')
    #     create_real(data_root=opt.data_root, save_root=opt.save_root, phase='train')
    # else:
    #     print('--dataset should be syn or real!')

    create_BSDS(data_root='/data1/hunnzi/DataSets/BSDS500/BSDS500',
                save_root='/data1/hunnzi/DataSets/BSDS500_release/', phase='test')