import matplotlib.pyplot as plt
import numpy as np
import math

# 生成示例数据 (一个 10x10 的矩阵)
data = np.random.rand(10, 10)

# 生成横纵坐标，范围为 -1 到 1，间隔为 0.2
x = np.linspace(-1, 1, data.shape[1])
y = np.linspace(-1, 1, data.shape[0])

# 使用 imshow 函数绘制图像，并设置 extent 参数
extent = [x.min(), x.max(), y.min(), y.max()]
plt.imshow(data, aspect='auto', cmap='viridis', extent=extent, origin='lower')

# 添加颜色条以显示数值范围
plt.colorbar()

# 添加标题和标签
plt.title("Random Data Heatmap with Custom Axes")
plt.xlabel("X axis")
plt.ylabel("Y axis")

# 设置 xy 轴网格线
plt.xticks(np.arange(-1, 1.2, 0.2))
plt.yticks(np.arange(-1, 1.2, 0.2))

# 显示图像
plt.grid(True)
plt.show()