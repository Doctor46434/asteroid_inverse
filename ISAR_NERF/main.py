import torch
import numpy as np
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
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
            outputs = torch.cat([sigma, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

def get_position_encoding(x, d_model):
    """
    Generate position encoding based on input x with shape (batch_size, seq_len) and d_model.
    """
    
    position_encoding = torch.zeros_like(x).to(device)
    position_encoding = position_encoding.repeat(d_model*2,1).t()
    div_term = 2 ** torch.arange(0, d_model, step=1).to(device)
    # Apply sin to even indices in the array; 2i
    position_encoding[:, 0::2] = torch.sin(x.unsqueeze(1) * div_term)
    # Apply cos to odd indices in the array; 2i+1
    position_encoding[:, 1::2] = torch.cos(x.unsqueeze(1) * div_term)
    
    return position_encoding

def apply_position_encoding(x):
    
    # Encode the first 3 positions with L=10 and add the original positions
    L1 = 10
    pos_encoding_first_part = get_position_encoding(x[:3], L1)
    x_first_encoded = torch.cat([x[:3].unsqueeze(1), pos_encoding_first_part], dim=1).view(-1)
    # Encode the last 3 positions with L=4 and add the original positions
    L2 = 4
    pos_encoding_last_part = get_position_encoding(x[3:], L2)
    x_last_encoded = torch.cat([x[3:].unsqueeze(1), pos_encoding_last_part], dim=1).view(-1)
    # Concatenate the encoded features
    x_encoded = torch.cat([x_first_encoded, x_last_encoded], dim=0)
    
    return x_encoded

def render(omega,LOS,model,doppler_num):
    '''
    输入变量：
    omega为一个[3]变量，指向旋转轴方向，模值为角速度
    LOS为一个[3]变量，方向为视线方向指向物体，模值为1
    model是nerf模型，将一个已经进行位置编码后的位置和视线向量输入进model,可以返回这个位置的体积密度和散射系数
    '''
    # 确定回波波长
    fc = torch.tensor([9.7e9]).to(device)
    c = torch.tensor([299792458]).to(device)
    lambda0 = c/fc
    # 确定网格参数
    distance_max = 60
    distance_min = -60
    distance_gap = 100
    doppler_max = 12
    doppler_min = -12
    doppler_gap = 100
    n_max = 60
    n_min = -60
    n_gap = 100
    # 确定投影平面
    omega_norm = torch.linalg.norm(omega)
    omega_normlize = omega/omega_norm
    Doppler_vector = torch.cross(LOS,omega)
    LOSomega_sin_angel = torch.linalg.norm(Doppler_vector)/(torch.linalg.norm(omega)*torch.linalg.norm(LOS))
    Doppler_vector = Doppler_vector/torch.linalg.norm(Doppler_vector)
    # 确定投影过程中
    distance = torch.linspace(distance_min,distance_max,distance_gap+1).to(device)
    doppler = torch.linspace(doppler_min,doppler_max,doppler_gap).to(device)*lambda0/omega_norm/2/LOSomega_sin_angel
    # 对距离向进行随机采样
    distance_random_array = torch.empty(distance_gap).to(device)
    for i in range(distance_gap):
        distance_random_array[i] = distance[i] + (distance[i+1] - distance[i]) * torch.rand(1).to(device)
    distance_random_map = LOS.unsqueeze(1)@distance_random_array.unsqueeze(0)
    doppler_map = omega_normlize.unsqueeze(1)@doppler.unsqueeze(0)
    # 确定投影平面法向量
    n = torch.cross(LOS,Doppler_vector)
    # 对法向量维进行随机采样
    n_array = torch.linspace(n_min,n_max,n_gap+1).to(device)
    n_random_array = torch.empty(n_gap).to(device)
    for i in range(n_gap):
        n_random_array[i] = n_array[i] + (n_array[i+1]-n_array[i]) * torch.rand(1).to(device)
    n_random_map = n.unsqueeze(1)@n_random_array.unsqueeze(0)
    # 计算每个随机法向量之间的间隔
    start_distance = LOS.unsqueeze(1)@torch.tensor(distance_min).float().to(device).unsqueeze(0)
    distance_random_map_temp = torch.cat((start_distance.unsqueeze(1),distance_random_map),dim=-1)
    distance_delta = torch.norm(distance_random_map_temp[:,0:-1]-distance_random_map[:,:],dim=0)
    start_n = n.unsqueeze(1)@torch.tensor(n_min).float().to(device).unsqueeze(0)
    n_random_map_temp = torch.cat((start_n.unsqueeze(1),n_random_map),dim=-1)
    n_delta = torch.norm(n_random_map_temp[:,0:-1]-n_random_map[:,:],dim=0)

    # 渲染一个多普勒下的多个distance
    alpha_box = torch.empty(distance_gap,n_gap).to(device)
    sigma_box = torch.empty(distance_gap,n_gap).to(device)
    for i in range(distance_gap):
        for j in range(n_gap):
            xyz = doppler_map[:,doppler_num] + distance_random_map[:,i]+n_random_map[:,j]
            position_code_before = torch.cat((xyz,LOS))
            position_code_after = apply_position_encoding(position_code_before)
            output = model(position_code_after)
            alpha_box[i,j] = output[0]
            sigma_box[i,j] = output[1]

    distance_profi = torch.zeros(distance_gap).to(device)
    for i in range(distance_gap):
        for j in range(n_gap):
            c = alpha_box[0:i+1,j]
            b = c**2
            a = torch.exp(-b*distance_delta[0:i+1])
            Ti = torch.cumprod(a,dim=0)[-1:]
            distance_profi[i] = distance_profi[i] + sigma_box[i,j]*alpha_box[i,j]*Ti*n_delta[j]
    
    return distance_profi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 指定文件夹路径
folder_path = './asteroid_image/Geographos'
# 获取文件夹中的所有文件和子文件夹
items = os.listdir(folder_path)
# 过滤出所有文件（排除子文件夹）
files = [item for item in items if os.path.isfile(os.path.join(folder_path, item))]
#载入数据
images = []
LOS_dirs = []
omegas = []
image_all = torch.zeros(60,100,100)
for file in files:
    full_path = folder_path+"/"+file
    data = np.load(full_path)
    image = data['image']
    LOS_dir = data['LOS']
    omega = data['rotation_axis']
    images.append(image)
    LOS_dirs.append(LOS_dir)
    omegas.append(omega)
print(images[1])
print(LOS_dirs[1])
print(omegas[1])
#载入模型
model = NeRF(input_ch = 63, input_ch_views = 27, use_viewdirs = True).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-)
# 旋转速度
omega_real = math.pi/900
losses = [] 
for epoch in range(500):
    data_num = torch.randint(1, 61, (1,)).item()
    doppler_profil_num = torch.randint(1, 101, (1,)).item()
    omega_data = (torch.from_numpy(omegas[data_num])*omega_real).to(device)
    LOS_dir_data = torch.from_numpy(LOS_dirs[data_num]).to(device)
    profile = render(omega_data,LOS_dir_data,model,doppler_profil_num)
    image_data = torch.from_numpy(image[doppler_profil_num,:]).to(device)
    loss = sum((profile-image_data)**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)
    losses.append(loss)
# LOS_dir = torch.tensor([-1,0,0]).float().to(device)
# omega = torch.tensor([0,0,omega_real]).float().to(device)
# model = NeRF(input_ch = 63, input_ch_views = 27, use_viewdirs = True).to(device)
# xyz = render(omega,LOS_dir,model,0)

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# loss = torch.sum(xyz**2)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print(xyz)