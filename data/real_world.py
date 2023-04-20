from data.dataset_real import Dataset
import os


class real_world(Dataset):
    def __init__(self, args, mode='train'):
        super(real_world, self).__init__(args, mode)
        self.save_dir = os.path.join(args.save_dir, 'result_images', mode)
        self.data_root =args.data_root
        self.datasets = {}
        self.sharp_list = self.scan_file(root=os.path.join(args.data_root, self.mode, 'blur'))
        self.datasets[self.mode] = self.sharp_list
        if mode == 'test':
            self.init_test_datasets()

    def scan_file(self, root=None):
        data_list = []
        for sub, dirs, files in os.walk(root):
            if not dirs:
                file_list = []
                for f in files:
                    if f.split('.')[-1] == 'bmp':
                        file_list.append(os.path.join(sub, f))
                data_list += file_list

        return data_list

    def init_test_datasets(self):
        # /data1/hunnzi/DataSets/real_world_test/test/blur/chair2Theta5.04_start1_to71_total_70.bmp
        img_list = self.scan_file(os.path.join(self.data_root, self.mode, 'blur'))
        for step in self.step_range:
            tmp_list = []
            for path in img_list:
                if(int(path.split('total_')[-1].split('.b')[0])==step):
                    if '_start1_' in path:
                        tmp_list.append(path)
            self.datasets[step] = tmp_list
        return None


    def __getitem__(self, idx):
        blur, sharp, pad_width, blur_field, idx, relpath = super(real_world, self).__getitem__(idx)
        sd = os.path.join(self.save_dir, 'theta_{}'.format(self.testangle))
        os.makedirs(sd, exist_ok=True)
        relpath = os.path.join(sd, relpath)

        return blur, sharp, pad_width, blur_field, idx, relpath
