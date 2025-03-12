import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones(self.weight.shape))
        # 将中心位置的权重设为0，创建盲点
        self.mask[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = 0
        
    def forward(self, x):
        # 应用掩码，强制网络不能直接看到中心像素
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class Noise2SelfModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Noise2SelfModel, self).__init__()
        
        # U-Net风格的结构
        # 编码器
        self.enc1 = nn.Sequential(
            MaskedConv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2 = nn.Sequential(
            MaskedConv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc3 = nn.Sequential(
            MaskedConv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            MaskedConv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            MaskedConv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d(64, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # 编码器前向传播
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        
        # 解码器前向传播
        upconv2 = self.upconv2(enc3)
        concat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(concat2)
        
        upconv1 = self.upconv1(dec2)
        concat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(concat1)
        
        return dec1


# 灰度图像噪声数据集
class NoisyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) 
                if f.endswith('.png') or f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # 转为灰度图
        
        if self.transform:
            image = self.transform(image)
            
        return image  # 噪声图像即为输入，也是训练目标


def create_random_mask(shape, fraction=0.5):
    """创建J-不变子集掩码"""
    # 创建初始全1掩码
    mask = torch.ones(shape, device='cuda')
    
    # 随机选择一部分像素设为0（这些是我们要预测的像素）
    n_elements = np.prod(shape)
    n_masked = int(n_elements * fraction)
    
    idx = torch.randperm(n_elements, device='cuda')[:n_masked]
    flat_mask = torch.ones(n_elements, device='cuda')
    flat_mask[idx] = 0  # 要预测的像素设为0
    
    return flat_mask.reshape(shape)


def train_noise2self(model, dataloader, epochs=50, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, noisy_imgs in enumerate(dataloader):
            noisy_imgs = noisy_imgs.to(device)
            
            # 创建掩码：0表示要预测的像素，1表示输入像素
            mask = create_random_mask(noisy_imgs.shape)
            
            # 模型的输入是所有像素
            outputs = model(noisy_imgs)
            
            # 只在掩码为0的位置（即要预测的位置）计算损失
            inv_mask = 1 - mask  # 反转掩码
            loss = criterion(outputs * inv_mask, noisy_imgs * inv_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.6f}')
    
    return model


# 使用训练好的模型进行去噪
def denoise_image(model, noisy_image, device='cuda'):
    model.eval()
    with torch.no_grad():
        noisy_image = noisy_image.unsqueeze(0).to(device)  # 添加batch维度
        denoised = model(noisy_image)
        return denoised.squeeze(0).cpu()  # 移除batch维度