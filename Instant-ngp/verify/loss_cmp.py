import torch
import matplotlib.pyplot as plt
import numpy as np

# 指定实验名称
experiment_name = 'experiment38'  # 与训练代码中的experiment_name保持一致

# 损失文件路径
loss_file_path = f'./model/{experiment_name}/loss_list.pth'

# 加载损失列表
losses = torch.load(loss_file_path)

# 将张量转换为numpy数组以便绘图
if isinstance(losses[0], torch.Tensor):
    losses_np = [loss.item() for loss in losses]
else:
    losses_np = losses

# 创建epoch索引
epochs = np.arange(len(losses_np))

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses_np)
plt.title(f'Training Loss ({experiment_name})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# 可选：使用对数刻度以便更好地可视化
# plt.yscale('log')

# 保存图像
plt.savefig(f'./model/{experiment_name}/loss_curve.png')

# 显示图像
plt.show()

# 输出一些统计信息
print(f"总训练epochs: {len(losses_np)}")
print(f"初始损失值: {losses_np[0]}")
print(f"最终损失值: {losses_np[-1]}")
print(f"最小损失值: {min(losses_np)}")
print(f"平均损失值: {np.mean(losses_np)}")