# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:07:03 2020

@author: WardhanaG
"""

#%% Load Image
import matplotlib.pyplot as plt
from PIL import Image
import torch
from math import log10
from torch.autograd import Variable
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from model import Generator
import pytorch_ssim

def display_transform_bic():
    return Compose([
        Resize(512,interpolation=Image.BICUBIC),
        CenterCrop(512),
        ToTensor()        
    ])

def display_transform_near():
    return Compose([
        Resize(512,interpolation=Image.NEAREST),
        CenterCrop(512),
        ToTensor()        
    ])

image_set_dir="p2.jpg"
model_dir = 'network/netG_epoch_4_50_animeDS.pth'
upscale_factor = int(4)

# model loading
model = Generator(upscale_factor).eval()
model.cuda()
model.load_state_dict(torch.load(model_dir))

# image generation
hr_image = Image.open(image_set_dir)
w,h = hr_image.size
hr = (ToTensor()(hr_image)).unsqueeze(0)

lr_scale = Resize(min(w,h)//upscale_factor,interpolation=Image.BICUBIC)
lr_image = lr_scale(hr_image)
lr = (ToTensor()(lr_image)).unsqueeze(0)

lr_image_cd = lr.cuda()
sr_image = model(lr_image_cd)
sr_image = ToPILImage()(sr_image[0].data.cpu())
sr = (ToTensor()(sr_image)).unsqueeze(0)

# PSNR and SSIM
batch_mse = ((sr - hr) ** 2).data.mean()
print(batch_mse)
psnr = 10 * log10(1 / batch_mse)
print("SRGAN psnr = %5.2f DB"%(psnr))

batch_ssim = pytorch_ssim.ssim(sr, hr).item()
print("SRGAN ssim = %5.2f "%(batch_ssim))
print("="*15)

# Comparation with other technique (BICUBIC)
sr_bic = (display_transform_bic()(lr_image)).unsqueeze(0)
batch_mse_bic = ((sr_bic - hr) ** 2).data.mean()
print("Mean: %5.5f"%(batch_mse_bic.item()))
psnr_bic = 10 * log10(1 / batch_mse_bic)
print("SR BICUBIC psnr = %5.2f DB"%(psnr_bic))

batch_ssim_bic = pytorch_ssim.ssim(sr_bic, hr).item()
print("SR BICUBIC ssim = %5.2f "%(batch_ssim_bic))
print("="*15)

# Comparation with other technique (NEAREST)
sr_near = (display_transform_near()(lr_image)).unsqueeze(0)
batch_mse_near = ((sr_near - hr) ** 2).data.mean()
print(batch_mse_near)
psnr_near = 10 * log10(1 / batch_mse_near)
print("SR NEAREST psnr = %5.2f DB"%(psnr_near))

batch_ssim_near = pytorch_ssim.ssim(sr_near, hr).item()
print("SR NEAREST ssim = %5.2f "%(batch_ssim_near))
print("="*15)
# image show
# =============================================================================
# plt.figure(1)
# ax = plt.subplot(131)
# ax.set_title("LR Image")
# plt.imshow(display_transform()(lr_image))
# 
# ax = plt.subplot(132)
# ax.set_title("HR Image")
# plt.imshow(display_transform()(hr_image))
# 
# ax = plt.subplot(133)
# ax.set_title("SR Image")
# plt.imshow(display_transform()(sr_image))
# 
# plt.show()
# =============================================================================

