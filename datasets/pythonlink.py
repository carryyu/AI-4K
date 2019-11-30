import os
from glob import glob
sourcepath_gt = '/data/lzy/tar/after_split/gt'
sourcepath_input = '/data/lzy/tar/after_split/input'
sourcepath_test = '/data/lzy/tar/after_split/test'
sourcepath_gt_resize = '/data/lzy/tar/after_split/gt_resize'
sourcepath_test_deblur = '/data/lzy/tar/after_split/test_deblur'
logpath = '/data/lzy/tar/logs/gt/'
logpath_test = '/data/lzy/tar/tests_logs/SDR_540p/'
distpath_gt = '/home/lzy/Downloads/vider_sr/datasets/all_data/gt'
distpath_input = '/home/lzy/Downloads/vider_sr/datasets/all_data/input'
distpath_test = '/home/lzy/Downloads/vider_sr/datasets/test_data'
distpath_gt_resize = '/home/lzy/Downloads/vider_sr/datasets/all_data/gt_resize'
distpath_test_deblur = '/home/lzy/Downloads/vider_sr/datasets/test_data_deblur'
sour = '/data/lzy/tar/videos/input/'
dist = '/home/lzy/fastdvdnet/input/'
folder_name = []

# deal with the gt and input
# for name in glob(logpath+'*.log'):
#     namebak = os.path.basename(name)
#     folder_name.append(namebak.split('.')[0])
#
# for name in folder_name:
#     os.symlink(os.path.join(sourcepath_input, name), os.path.join(distpath_input, name))
#     os.symlink(os.path.join(sourcepath_gt, name), os.path.join(distpath_gt, name))

for name in glob(logpath+'*.log'):
    namebak = os.path.basename(name)
    folder_name.append(namebak.split('.')[0])

for name in folder_name:
    os.symlink(os.path.join(sourcepath_input, name), os.path.join(distpath_input, name))