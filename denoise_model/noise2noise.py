import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm

class Noise2NoiseDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, transform=None, train=True):
        """
        Noise2Noise数据集
        Args:
            image_dir: 存储干净图像的目录
            patch_size: 图像块大小
            transform: 图像预处理
            train: 是否为训练集
        """
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transform = transform
        self.train = train
        
        # 获取所有图像文件
        self.image_files = []
        for file in os.listdir(image_dir):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                self.image_files.append(os.path.join(image_dir, file))
                
    def __len__(self):
        return len(self.image_files)
    
    def add_noise(self, img):
        """添加高斯噪声"""
        noise_level = random.uniform(10, 50) / 255.0  # 随机噪声水平
        noise = torch.randn_like(img) * noise_level
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0, 1)  # 确保值在[0,1]范围内
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # 读取为灰度图
        img = Image.open(img_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
        
        # 训练时随机裁剪图像块
        if self.train:
            h, w = img.shape[1], img.shape[2]
            if h > self.patch_size and w > self.patch_size:
                i = random.randint(0, h - self.patch_size)
                j = random.randint(0, w - self.patch_size)
                img = img[:, i:i+self.patch_size, j:j+self.patch_size]
        
        # 生成两个独立噪声版本
        noisy_img1 = self.add_noise(img)
        noisy_img2 = self.add_noise(img)
        
        return noisy_img1, noisy_img2
    
class DenoisingUNet(nn.Module):
    def __init__(self):
        super(DenoisingUNet, self).__init__()
        
        # 编码器
        self.enc_conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        
        # 瓶颈
        self.bottleneck = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        
        # 解码器
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv1 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv2 = nn.Conv2d(144, 96, kernel_size=3, padding=1)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv3 = nn.Conv2d(144, 64, kernel_size=3, padding=1)
        
        # 输出层
        self.output_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 编码器
        e1 = self.relu(self.enc_conv1(x))
        e1 = self.relu(self.enc_conv2(e1))
        p1 = self.pool1(e1)
        
        e2 = self.relu(self.enc_conv3(p1))
        p2 = self.pool2(e2)
        
        e3 = self.relu(self.enc_conv4(p2))
        p3 = self.pool3(e3)
        
        # 瓶颈
        b = self.relu(self.bottleneck(p3))
        
        # 解码器
        d1 = self.upsample1(b)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.relu(self.dec_conv1(d1))
        
        d2 = self.upsample2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.relu(self.dec_conv2(d2))
        
        d3 = self.upsample3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.relu(self.dec_conv3(d3))
        
        # 输出
        output = self.output_conv(d3)
        
        return output
    
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for noisy_img1, noisy_img2 in tqdm(dataloader):
        noisy_img1 = noisy_img1.to(device)
        noisy_img2 = noisy_img2.to(device)
        
        optimizer.zero_grad()
        
        # 输入一个噪声图像，预测另一个噪声图像
        output = model(noisy_img1)
        loss = criterion(output, noisy_img2)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def test_model(model, image_path, device):
    """测试单张图像的去噪效果"""
    model.eval()
    
    # 加载图像并转换为灰度图
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert('L')
    clean_img = transform(img).unsqueeze(0).to(device)
    
    # 添加噪声
    noise_level = 25/255.0
    noise = torch.randn_like(clean_img) * noise_level
    noisy_img = torch.clamp(clean_img + noise, 0, 1)
    
    # 去噪
    with torch.no_grad():
        denoised_img = model(noisy_img)
    
    # 转换为numpy进行显示
    clean_np = clean_img.squeeze().cpu().numpy()
    noisy_np = noisy_img.squeeze().cpu().numpy()
    denoised_np = denoised_img.squeeze().cpu().numpy()
    
    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean_np, cmap='gray')
    axes[0].set_title('Clean')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_np, cmap='gray')
    axes[1].set_title(f'Noisy (σ={noise_level*255:.1f})')
    axes[1].axis('off')
    
    axes[2].imshow(denoised_np, cmap='gray')
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_result.png')
    plt.show()

    # 计算PSNR
    mse_noisy = ((clean_np - noisy_np) ** 2).mean()
    mse_denoised = ((clean_np - denoised_np) ** 2).mean()
    psnr_noisy = 10 * np.log10(1.0 / mse_noisy)
    psnr_denoised = 10 * np.log10(1.0 / mse_denoised)
    
    print(f'PSNR - Noisy: {psnr_noisy:.2f} dB, Denoised: {psnr_denoised:.2f} dB')

def main():
    # 设置参数
    image_dir = './887alinda/images_png'  # 修改为您的图像目录
    batch_size = 32
    num_epochs = 100
    patch_size = 64
    learning_rate = 1e-3
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    train_dataset = Noise2NoiseDataset(
        image_dir=image_dir,
        patch_size=patch_size,
        transform=transform,
        train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # 创建模型
    model = DenoisingUNet().to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}')
        
        # 更新学习率
        scheduler.step(loss)
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'n2n_denoiser_epoch{epoch+1}.pth')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'n2n_denoiser_final.pth')
    print('Training completed!')
    
    # 测试模型
    test_image = './887alinda/images_png/图片-4_01.png'  # 修改为您的测试图像
    test_model(model, test_image, device)

if __name__ == "__main__":
    main()