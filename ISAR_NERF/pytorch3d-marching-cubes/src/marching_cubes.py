import numpy as np
import torch
import numpy as np
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
import random
import matplotlib.pyplot as plt
import re
from skimage.measure import marching_cubes


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=2, skips=[4], use_viewdirs=True):
        """
        D: 深度，多少层网络
        W: 网络内的channel 宽度
        input_ch: xyz的宽度
        input_ch_views: direction的宽度
        output_ch: 这个参数尽在 use_viewdirs=False的时候会被使用
        skips: 类似resnet的残差连接，表明在第几层进行连接
        use_viewdirs:

        网络输入已经被位置编码后的参数，输入为[64*bs,90]，输出为[64*bs，2]，一位是体积密度，一位是后向散射系数
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # 神经网络,MLP
        # 3D的空间坐标进入的网络
        # 这个跳跃连接层是直接拼接，不是resnet的那种相加
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        # 这里channel削减一半 128
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if use_viewdirs:
            # 特征
            self.feature_linear = nn.Linear(W, W)
            # 体积密度,一个值
            self.alpha_linear = nn.Linear(W, 1)
            # 后向散射系数，一个值
            self.rho_linear = nn.Linear(W // 2, 1)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # x [bs*64, 90]
        # input_pts [bs*64, 63]
        # input_views [bs*64,27]
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts

        for i, l in enumerate(self.pts_linears):

            h = self.pts_linears[i](h)
            h = F.relu(h)
            # 第四层后相加
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # alpha只与xyz有关
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            # rho与xyz和d都有关
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            sigma = self.rho_linear(h)
            alpha = self.relu(alpha) 
            sigma = self.sigmoid(sigma)
            outputs = torch.cat([alpha, sigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
    
def positon_code_xyz(xyz):
    code_len = 10
    length,dimen = xyz.shape
    xyz = xyz
    position_coding = torch.zeros_like(xyz).to(device)
    position_coding = position_coding.repeat(1,code_len*2)
    div_term = 2 ** torch.arange(0,code_len,step=1).to(device)
    position_coding[:,0::2] = torch.sin((xyz.unsqueeze(1) * math.pi * div_term.unsqueeze(1).unsqueeze(0)).view(length,-1))
    position_coding[:,1::2] = torch.cos((xyz.unsqueeze(1) * math.pi * div_term.unsqueeze(1).unsqueeze(0)).view(length,-1))
    position_coding = torch.cat((xyz,position_coding),dim=1)
    
    return position_coding

def position_code_LOS(LOS):
    code_len = 4
    batch_size,dimension = LOS.shape
    position_coding = torch.zeros_like(LOS).to(device)
    position_coding = position_coding.repeat(1,code_len*2)
    div_term = 2 ** torch.arange(0,code_len,step=1).to(device)
    position_coding[:,0::2] = torch.sin((LOS.unsqueeze(1) * div_term.unsqueeze(1).unsqueeze(0)).view(batch_size,-1))
    position_coding[:,1::2] = torch.cos((LOS.unsqueeze(1) * div_term.unsqueeze(1).unsqueeze(0)).view(batch_size,-1))
    position_coding = torch.cat((LOS,position_coding),dim=1)

    return position_coding


# def marching_cubes_batch(scalar_field, threshold, batch_size=10000):
#     """
#     使用分批次输入的方式生成等值面。
#     scalar_field: 三维标量场，形状为 (dim_x, dim_y, dim_z)
#     threshold: 等值面阈值
#     batch_size: 每次处理的点的数量

#     返回:
#     vertices: 等值面顶点坐标
#     faces: 等值面三角形面片索引
#     """
#     import numpy as np
#     from skimage.measure import marching_cubes

#     dim_x, dim_y, dim_z = scalar_field.shape
#     vertices_list = []
#     faces_list = []

#     for i in range(0, dim_x, batch_size):
#         for j in range(0, dim_y, batch_size):
#             for k in range(0, dim_z, batch_size):
#                 sub_field = scalar_field[i:i+batch_size, j:j+batch_size, k:k+batch_size]
#                 if sub_field.size == 0:
#                     continue
#                 sub_vertices, sub_faces, _, _ = marching_cubes(sub_field, level=threshold)
#                 vertices_list.append(sub_vertices + [i, j, k])
#                 faces_list.append(sub_faces + len(vertices_list) * len(sub_vertices))

#     vertices = np.concatenate(vertices_list, axis=0)
#     faces = np.concatenate(faces_list, axis=0)

#     return vertices, faces

def create_scalar_field(dim_x, dim_y, dim_z, model, batch_size=10000):
    x = torch.linspace(-dim_x/2, dim_x/2, 120*2).to(device)
    y = torch.linspace(-dim_y/2, dim_y/2, 100*2).to(device)
    z = torch.linspace(-dim_z/2, dim_z/2, 100*2).to(device)
    xyz = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
    xyz = xyz.view(-1, 3)

    scalar_field = torch.zeros((120*2, 100*2, 100*2), dtype=torch.float32).to(device)
    for i in range(0, xyz.shape[0], batch_size):
        # 打印进程百分比
        print(f"Progress: {i/xyz.shape[0]*100:.2f}%")
        batch_xyz = xyz[i:i+batch_size]
        batch_scalar = model(batch_xyz).detach().cpu().numpy()
        scalar_field.view(-1)[i:i+batch_size] = torch.tensor(batch_scalar, dtype=torch.float32)

    return scalar_field.detach().cpu().numpy()

class field_model(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.nerf = NeRF()
    def forward(self, xyz):
        LOS = torch.tensor([1, 0, 0]).unsqueeze(0).expand(xyz.shape).to(self.device)
        LOS = position_code_LOS(LOS)
        xyz = positon_code_xyz(xyz)
        xyzlos = torch.cat([xyz,LOS],dim=1)
        model = NeRF(input_ch=63, input_ch_views=27, use_viewdirs=True).to(self.device)
        model.load_state_dict(torch.load('./model_state_dict82.pth'))
        model.eval()
        scalar_field = model(xyzlos)[...,0]
        return scalar_field

def main():
    """
    主函数，用于测试Marching Cubes算法。
    """
    import torch
    import numpy as np
    distance_max = 0.6  
    distance_min = -0.6
    distance_gap = 100
    doppler_max = 0.15
    doppler_min = -0.15
    doppler_gap = 100
    n_max = 0.30
    n_min = -0.30
    n_gap = 120

    dim_x, dim_y, dim_z = 200, 150, 150
    # dim_x, dim_y, dim_z = 1.2,1.0,1.0
    threshold = 0.21
    modelnum = 82
    # 生成模型
    model = field_model(device)


    scalar_field = create_scalar_field(dim_x, dim_y, dim_z, model)
    print(f"scalar_field: {scalar_field.shape}")
    # 使用Marching Cubes算法
    from skimage.measure import marching_cubes
    vertices, faces, _, _ = marching_cubes(scalar_field, level=threshold)

    print(f"Vertices: {vertices.shape}")
    print(f"Faces: {faces.shape}")

    # 导出结果为obj文件，并将其命名为result+threshold.obj
    with open(f"result_{threshold}_{modelnum}.obj", "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()