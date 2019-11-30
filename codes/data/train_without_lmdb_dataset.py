import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import logging
import random
import numpy as np

logger = logging.getLogger('base')

class MyDataset(data.Dataset):
    """
    A video train dataset

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(MyDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.random_reverse = opt['random_reverse']
        self.interval_list = opt['interval_list']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = True if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB')
        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        if opt['name'].lower() in ['unlmdb']:
            subfolders_LQ = util.glob_file_list(self.LQ_root)
            subfolders_GT = util.glob_file_list(self.GT_root)
            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_LQ)
                max_idx_gt = len(img_paths_GT)
                # assert max_idx == len(
                #     img_paths_GT), 'Different number of images in LQ and GT folders'
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
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
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
                else: # add, if not cache_data, store the path of image
                    self.imgs_LQ[subfolder_name] = img_paths_LQ
                    self.imgs_GT[subfolder_name] = img_paths_GT
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
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        interval = random.choice(self.interval_list)
        border = self.data_info['border'][index]
        center_frame_idx = idx
        # deal with the border
        while(border == 1):
            center_frame_idx = random.randint(0, len(self.imgs_GT[folder])-1)
            border = self.data_info['border'][center_frame_idx]
        # get the neighbor_list
        neighbor_list = list(
            range(center_frame_idx - self.half_N_frames * interval,
                  center_frame_idx + self.half_N_frames * interval + 1, interval))
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()
        LQ_size_tuple = (3, 540, 960) if self.LR_input else (3, 2160, 3840)

        if self.cache_data:
            imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(neighbor_list))
            img_GT = self.imgs_GT[folder][idx]
        else:
            imgs_LQ_path = self.imgs_LQ[folder][neighbor_list[0]:neighbor_list[-1]+1]
            imgs_GT_path = [self.imgs_GT[folder][center_frame_idx]]
            # imgs_LQ = util.read_img(None,imgs_LQ_path)
            # img_GT = util.read_img(None,imgs_GT_path)
            imgs_LQ = [util.read_img(None, i_path) for i_path in imgs_LQ_path]
            img_GT = [util.read_img(None, i_path) for i_path in imgs_GT_path]
        img_LQ_l = [v for v in imgs_LQ]
        if self.opt['phase'] == 'train' and self.opt['random_crop'] == True:
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                # torch
                # img_LQ_l = [v[:, rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size] for v in img_LQ_l]
                # rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                # img_GT = img_GT[0][:,rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size]
                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[0][rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

        # aug
        # img_LQ_l.append(img_GT) when sr
        img_LQ_l.append(img_GT)
        rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])   #util.torch_aug(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
        img_LQ_l = rlt[0:-1]
        img_GT = rlt[-1]
        img_LQs = np.stack(img_LQ_l, axis=0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()  # 3 512 512
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()  # 5 3 128 128

        return {'LQs': img_LQs, 'GT': img_GT}

    def __len__(self):
        return len(self.data_info['path_GT'])
