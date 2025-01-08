import numpy as np
import matplotlib.pyplot as plt
import math

# 读取 .npz 文件
loaded_data = np.load('./asteroid_image/Geographos/Nerf_data20.npz')

fc = 9.7e9
c = 299792458
lambda0 = c/fc
omega = math.pi/900

# 访问各个数组
radar_image = loaded_data['image']
range = np.linspace(-60,60,100)
doppler = np.linspace(-12,12,100)*lambda0/omega/2/math.sin(math.pi/3)
# doppler = np.linspace(-12,12,100)
extent = [range.min(), range.max(), doppler.min(), doppler.max()]
plt.imshow(radar_image[25:75,:], aspect='auto',extent=extent, cmap='viridis')
plt.colorbar()  # 添加颜色条以显示色图的标度
plt.title("groundtruth")
plt.xlabel("range/m")
plt.ylabel("doppler/Hz")
plt.grid(True)
plt.show()
