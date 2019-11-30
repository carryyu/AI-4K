import os
from glob import glob
import sys
import shutil

basepath = '/data/lzy/tar/pngs/'
save_path = '/data/lzy/tar/after_split'

name1 = 'gt'
name2 = 'input'
name3 = 'test'
path_gt = os.path.join(basepath, name1)
path_input = os.path.join(basepath, name2)
path_test = os.path.join(basepath, name3)
print('input dataset')
# for image_path in glob(path_input+'/*.png'):
#     img_path = image_path.split('/')[-1]
#     dir_name = img_path[:8]
#     new_name = img_path[8:]
#     save_path_input = os.path.join(save_path, name2)
#     new_path = os.path.join(save_path_input, dir_name, new_name)
#     print('new path: {}'.format(new_path))
#     shutil.move(image_path, new_path)
print('input dataset finished')
print('gt dataset')
# for image_path in glob(path_gt+'/*.png'):
#     img_path = image_path.split('/')[-1]
#     dir_name = img_path[:8]
#     new_name = img_path[8:]
#     save_path_gt = os.path.join(save_path, name1)
#     new_path = os.path.join(save_path_gt, dir_name, new_name)
#     print('new path: {}'.format(new_path))
#     shutil.move(image_path, new_path)
print('gt dataset finished')
print('test dataset')
for image_path in glob(path_test+'/*.png'):
    img_path = image_path.split('/')[-1]
    dir_name = img_path[:8]
    new_name = img_path[8:]
    save_path_test = os.path.join(save_path, name3)
    new_path = os.path.join(save_path_test, dir_name, new_name)
    print('new path: {}'.format(new_path))
    shutil.move(image_path, new_path)
print('test dataset finished')

