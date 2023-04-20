from data.dataset import Dataset
import os


class BSDS500(Dataset):
    def __init__(self, args, mode='train'):
        super(BSDS500, self).__init__(args, mode)
        self.save_dir = os.path.join(args.save_dir, 'result_images', mode)
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
        img_list = self.scan_file(os.path.join(self.args.data_root, self.mode, 'blur'))
        for step in self.step_range:
            tmp_list = []
            for path in img_list:
                if str(step) == path.split('/')[-1].split('_')[3]:
                    tmp_list.append(path)
                self.datasets[step] = tmp_list
        return None

    def __getitem__(self, idx):
        blur, sharp, pad_width, blur_field, idx, relpath = super(BSDS500, self).__getitem__(idx)
        sd = os.path.join(self.save_dir, 'theta_{}'.format(self.testangle))
        os.makedirs(sd, exist_ok=True)
        relpath = os.path.join(sd, relpath)

        return blur, sharp, pad_width, blur_field, idx, relpath
