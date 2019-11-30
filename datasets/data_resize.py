import cv2
import os
from glob import glob
img_path = '/data/lzy/tar/after_split/gt/'
save_path = '/data/lzy/tar/after_split/gt_resize/'
count = 0
for img_dir in os.listdir(img_path):
    print(count+1)
    count+=1
    save_dir = os.path.join(save_path, img_dir)
    if not  os.path.exists(save_dir):
        os.mkdir(save_dir)
    for imgpath in glob(os.path.join(img_path, img_dir,'*.png')):
        img = cv2.imread(imgpath)
        img_resize = cv2.resize(img, (960, 540))
        img_save_path = os.path.join(save_dir, os.path.basename(imgpath))
        cv2.imwrite(img_save_path, img_resize)