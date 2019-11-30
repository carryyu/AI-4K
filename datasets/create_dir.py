import os
from glob import glob
import sys
import shutil
basepath = '/data/lzy/tar/after_split/test/'
logpath = '/data/lzy/tar/tests_logs/SDR_540p/'
save_path = basepath
folder_name = []
for name in glob(logpath+'*.log'):
    namebak = os.path.basename(name)
    folder_name.append(namebak.split('.')[0])
for name in folder_name:
    path = os.path.join(save_path, name)
    if(not os.path.exists(path)):
        os.makedirs(path)
