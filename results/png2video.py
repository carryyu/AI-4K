import os
import torchvision.transforms as transform
transform.RandomHorizontalFlip
base_path = 'results/EDVR_inference/my_test/'
save_path = 'results/video_235000/'
for png_dir in os.listdir(base_path):
    print(png_dir)
for png_dir in os.listdir(base_path):
    os.system('ffmpeg -r 24000/1001 -i {}{}/%4d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}{}.mp4'.format(base_path ,png_dir,save_path,png_dir))
