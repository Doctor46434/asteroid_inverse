import torch
from noise2self import Noise2SelfModel, NoisyDataset, train_noise2self, denoise_image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# 设置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 128
batch_size = 20
epochs = 30

# 数据预处理
transform = transforms.Compose([
    # 随机裁剪而不是强制缩放
    transforms.RandomCrop(image_size) if image_size <= 200 else transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器（替换为您的噪声图像文件夹路径）
dataset = NoisyDataset(image_dir='./887alinda_part', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型
model = Noise2SelfModel(in_channels=1, out_channels=1)

# 训练模型
trained_model = train_noise2self(model, dataloader, epochs=epochs, device=device)

# 保存模型
torch.save(trained_model.state_dict(), 'noise2self_model1.pth')

# 测试去噪效果
test_image = Image.open('./887alinda_part/图片-4_01.png').convert('L')
test_tensor = transform(test_image)

# 执行去噪
denoised = denoise_image(trained_model, test_tensor, device)

# 显示结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Noisy Image')
plt.imshow(test_tensor.squeeze().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image')
plt.imshow(denoised.squeeze().numpy(), cmap='gray')
plt.axis('off')
plt.show()

# 保存图窗
plt.savefig('denoised_image.png')