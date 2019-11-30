import os.path as osp
import torch
import torch.utils.data as data
import data.util as util


class my_test_dataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(my_test_dataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.LQ_root = opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs_LQ= {}
        if opt['name'].lower() in ['my_test']:
            subfolders_LQ = util.glob_file_list(self.LQ_root)
            for subfolder_LQ in subfolders_LQ:
                subfolder_name = osp.basename(subfolder_LQ)
                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                max_idx = len(img_paths_LQ)
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                if self.cache_data:
                    self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
                else:  # TODO
                    self.imgs_LQ[subfolder_name] = img_paths_LQ
        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

    def __getitem__(self, index):
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        if self.cache_data:
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                               padding=self.opt['padding'])
            imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            img_GT = self.imgs_GT[folder][idx]
        else:
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                               padding=self.opt['padding'])
            # imgs_LQ_path = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            imgs_LQ_path = [self.imgs_LQ[folder][i] for i in select_idx]
            imgs_LQ = util.read_img_seq(imgs_LQ_path)

        return {
            'LQs': imgs_LQ,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])
