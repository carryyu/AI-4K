import loss_pytorch.ssim as ssim
import cv2
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
img1 = cv2.imread('/home/lzy/Downloads/EDVR-master/datasets/pngs/gt/10091373/0001.png')
img2 = cv2.imread('/home/lzy/Downloads/EDVR-master/datasets/pngs/gt/10091373/0002.png')
tran = transforms.ToTensor()
img1 = tran(img1)
img2 = tran(img2)
img1 = img1.unsqueeze(0)
img2 = img2.unsqueeze(0)
diff = F.mse_loss(img1, img2)
psnr_value = 20*torch.log10(1.0/torch.sqrt(diff))
ssim_value = ssim.ssim(img1,img2)
print(psnr_value)
print(ssim_value)