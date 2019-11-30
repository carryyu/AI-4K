import cv2
from glob import glob
import os
path = '/home/lzy/Downloads/EDVR-master/datasets/pngs/gt'
save_path = '/home/lzy/vnlnet-master/val_deblur/gt'
for img_dir in os.listdir(path):
    new_dir = os.path.join(save_path, img_dir)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for image in glob(os.path.join(path,img_dir, "*.png")):
        image_name = os.path.basename(image)
        image = cv2.imread(image)
        image = cv2.resize(image, (960, 540))
        cv2.imwrite(os.path.join(new_dir, image_name), image)

