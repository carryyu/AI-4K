import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import pickle
import lmdb
import random
import numpy as np

class VideoTestLmdbDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VideoTestLmdbDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.interval_list = opt['interval_list']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        elif opt['cache_keys']:
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')
        if self.data_type == 'lmdb':
            self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                    meminit=False)
            self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                    meminit=False)
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))


    def __getitem__(self, index):
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)
        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        select_idx = util.val_index_generation(center_frame_idx, self.opt['N_frames'],
                                           padding=self.opt['padding'])
        img_LQs = [util.read_img(self.LQ_env, '{}_{:04d}'.format(name_a, v), (3,540,960)) for v in select_idx]
        img_GT = [util.read_img(self.GT_env, key, (3, 540, 960))]
        # for deblur, random crop
        if self.opt['random_crop'] == True:
            C, H, W = (3,540,960)  # LQ size
            scale = 1
            # randomly crop
            LQ_size,GT_size = 512,512
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            # torch
            # img_LQ_l = [v[:, rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size] for v in img_LQ_l]
            # rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            # img_GT = img_GT[0][:,rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size]
            img_LQs = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQs]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[0][rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]

        img_LQs = np.stack(img_LQs, axis=0)
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()  # 3 512 512
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()  # 5 3 128 128
        return {
            'LQs': img_LQs,
            'GT': img_GT,
            'folder': name_a,
            'idx': '{}/{}'.format(center_frame_idx, 100)
        }

    def __len__(self):
        return len(self.paths_GT)
