from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os 

def sub(image_path):
# 打开并转换图像为灰度图
    image = Image.open(image_path).convert('L')
    image_array = np.array(image, dtype=np.int32)

    # 定义要减去的值
    subtract_value = 50

    # 减去该值并将负值设为0
    res = image_array - subtract_value
    result_array = np.clip(res, 0, 255)

    # 转换回图像并保存
    result_image = Image.fromarray(result_array.astype(np.uint8))

    # plt.imshow(result_image)
    # plt.show()
    # result_image.save('path/to/save/result_image.png')
    return result_image

target_folder = 'D:/研二上/实验室/NeRF/金石数据结果/2024wb/images_png_crop_rot_submean'
source_folder = 'D:/研二上/实验室/NeRF/金石数据结果/2024wb/images_png_crop_rot_denoise'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

png_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
for png_file in png_files:
    source_path = os.path.join(source_folder, png_file)
    target_path = os.path.join(target_folder, png_file)
    denoise_image = sub(source_path)
    # plt.imshow(denoise_image,cmap='gray')
    denoise_image.save(target_path,'PNG')



