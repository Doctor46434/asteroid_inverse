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

def batchrender(omega,LOS,model,doppler_num):
    '''
    omega为一个[bs,3]变量，指向旋转轴方向，模值为角速度
    LOS为一个[bs,3]变量，方向为视线方向指向物体，模值为1
    model是nerf模型，将一个已经进行位置编码后的位置和视线向量输入进model,可以返回这个位置的体积密度和散射系数
    doppler_num为一个[bs]变量，确定了渲染后光线所在的位置
    '''
    # 确定回波波长
    fc = torch.tensor([9.7e9]).to(device)
    c = torch.tensor([299792458]).to(device)
    lambda0 = c/fc
    # 确定网格参数
    distance_max = 60
    distance_min = -60
    distance_gap = 100
    doppler_max = 15
    doppler_min = -15
    doppler_gap = 100
    n_max = 30
    n_min = -30
    n_gap = 120
    # 确定输入batch_size
    batch_size,len = omega.shape
    # 确定每个batch_size输入的投影平面
    omega_norm = torch.linalg.norm(omega,dim = 1)
    omega_normlize = omega/omega_norm.unsqueeze(1)
    Doppler_vector = torch.cross(LOS,omega,dim=1)
    LOSomega_sin_angel = torch.linalg.norm(Doppler_vector,dim=1)/(torch.linalg.norm(omega,dim=1)*torch.linalg.norm(LOS,dim=1))
    Doppler_vector = Doppler_vector/torch.linalg.norm(Doppler_vector,dim = 1).unsqueeze(1)
    # 绘制投影坐标
    distance = torch.linspace(distance_min,distance_max,distance_gap).to(device)
    distance = distance.repeat(batch_size,1)
    distance_delta = torch.tensor((distance_max-distance_min)/distance_gap).to(device)
    doppler = torch.linspace(doppler_min,doppler_max,doppler_gap).repeat(batch_size,1).to(device)
    doppler = doppler*lambda0/omega_norm.unsqueeze(1)/2/LOSomega_sin_angel.unsqueeze(1)
    distance_map = distance.unsqueeze(2)*LOS.unsqueeze(1)
    doppler_map = doppler.unsqueeze(2)*Doppler_vector.unsqueeze(1)
    # 确定投影平面法向量
    n = torch.cross(LOS,Doppler_vector,dim=1)
    n = n/torch.linalg.norm(n,dim = 1).unsqueeze(1)
    # 对投影平面法向量进行随机采样
    n_array = torch.linspace(n_min,n_max,n_gap+1).to(device)
    n_array = n_array.repeat(batch_size,distance_gap,1)
    # 非随机采样
    # n_random_array = n_array[:,:,0:-1] + (n_array[:,:,1:] - n_array[:,:,0:-1])*torch.ones(batch_size,distance_gap,n_gap).to(device)*0.5
    # # 随机采样
    n_random_array = n_array[:,:,0:-1] + (n_array[:,:,1:] - n_array[:,:,0:-1])*torch.rand(batch_size,distance_gap,n_gap).to(device)
    n_random_map = n_random_array.unsqueeze(3)*n.unsqueeze(1).unsqueeze(2)
    # 计算不同随机法向量之间的间隔
    start_n = n.unsqueeze(1).unsqueeze(2)*torch.tensor(n_min).float().to(device)
    start_n = start_n * torch.ones(batch_size,distance_gap,1,3).to(device)
    n_random_map_temp = torch.cat((start_n,n_random_map),dim=2)
    n_delta = torch.norm(n_random_map_temp[:,:,0:-1,:]-n_random_map,dim=3)

    # 计算所有需要输入网络的坐标
    code_flag = 1
    if code_flag == 1:
        xyz = doppler_map[torch.arange(batch_size),doppler_num,:].unsqueeze(1).unsqueeze(2) + distance_map.unsqueeze(2) + n_random_map

        # for i in range(0,100,10):
        #     print(i)
        #     xyz_visual = xyz[i,:,:,:].view(-1,3).detach().cpu()
        #     x, y, z = xyz_visual[:, 0], xyz_visual[:, 1], xyz_visual[:, 2]

        #     # 创建一个3D散点图
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     sc = ax.scatter(x, y, z, c=z, cmap='viridis')

        #     # 添加颜色条
        #     plt.colorbar(sc)

        #     # 设置轴标签
        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')
        #     ax.set_zlabel('Z Label')

        #     ax.axis('equal')

        #     # 显示图形
        #     plt.show()

        xyz_coding = positon_code_xyz(xyz)
        LOS_coding = position_code_LOS(LOS)
        LOS_coding = ((LOS_coding.unsqueeze(1).unsqueeze(2))*torch.ones(batch_size,distance_gap,n_gap,27).to(device)).view(-1,27)
        xyzLOS_coding = torch.cat((xyz_coding,LOS_coding),dim=1)
    else:
        xyz = doppler_map[torch.arange(batch_size),doppler_num,:].unsqueeze(1).unsqueeze(2) + distance_map.unsqueeze(2) + n_random_map
        xyz_coding = xyz.view(-1,3)
        LOS_coding = ((LOS.unsqueeze(1).unsqueeze(2))*torch.ones(batch_size,distance_gap,n_gap,3).to(device)).view(-1,3)
        print(LOS_coding)
        xyzLOS_coding = torch.cat((xyz_coding,LOS_coding),dim=1)
    output = model(xyzLOS_coding)
    output = output.view(batch_size,distance_gap,n_gap,2)
    Ti = torch.cumprod(torch.exp(-output[:,:,:,0]**2*distance_delta),dim=1)
    distance_profile = torch.sum(output[:,:,:,0]*output[:,:,:,1]*Ti*n_delta,dim=2)

    weights = output[:,:,:,0]*Ti*n_delta

    return distance_profile,weights

def sample_method(weight,n_euqal_array):
    n_euqal_array_new = n_euqal_array.detach()
    weight = weight.detach()
    batchsize, distance_gap, n_gap = n_euqal_array.shape
    # 防止出现nan
    weight = weight + 1e-5

    # 计算pdf和cdf
    pdf = weight/torch.sum(weight,dim=-1).unsqueeze(-1)
    cdf = torch.cumsum(pdf,dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],-1)

    # 等间隔采样
    u = torch.linspace(0. ,1., steps=n_gap).to(device)
    u = u.unsqueeze(0).unsqueeze(1).expand(batchsize,distance_gap,n_gap)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    cdf_g = torch.gather(cdf.unsqueeze(3).expand(batchsize, distance_gap, n_gap ,2), 2, inds_g)
    

    # 计算插值的位置
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    bins_g = torch.gather(n_euqal_array_new.unsqueeze(3).expand(batchsize, distance_gap, n_gap ,2), 2, inds_g)
    # 使用线性插值计算采样点
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
    
def render_fine(omega,LOS,model,doppler_num,weight):
    # 确定回波波长
    fc = torch.tensor([9.7e9]).to(device)
    c = torch.tensor([299792458]).to(device)
    lambda0 = c/fc
    # 确定网格参数
    distance_max = 60
    distance_min = -60
    distance_gap = 100
    doppler_max = 15
    doppler_min = -15
    doppler_gap = 100
    n_max = 30
    n_min = -30
    n_gap = 120
    # 确定输入batch_size
    batch_size,len = omega.shape
    # 确定每个batch_size输入的投影平面
    omega_norm = torch.linalg.norm(omega,dim = 1)
    omega_normlize = omega/omega_norm.unsqueeze(1)
    Doppler_vector = torch.cross(LOS,omega,dim=1)
    LOSomega_sin_angel = torch.linalg.norm(Doppler_vector,dim=1)/(torch.linalg.norm(omega,dim=1)*torch.linalg.norm(LOS,dim=1))
    Doppler_vector = Doppler_vector/torch.linalg.norm(Doppler_vector,dim = 1).unsqueeze(1)
    # 绘制投影坐标
    distance = torch.linspace(distance_min,distance_max,distance_gap).to(device)
    distance = distance.repeat(batch_size,1)
    distance_delta = torch.tensor((distance_max-distance_min)/distance_gap).to(device)
    doppler = torch.linspace(doppler_min,doppler_max,doppler_gap).repeat(batch_size,1).to(device)
    doppler = doppler*lambda0/omega_norm.unsqueeze(1)/2/LOSomega_sin_angel.unsqueeze(1)
    distance_map = distance.unsqueeze(2)*LOS.unsqueeze(1)
    doppler_map = doppler.unsqueeze(2)*Doppler_vector.unsqueeze(1)
    # 确定投影平面法向量
    n = torch.cross(LOS,Doppler_vector,dim=1)
    n = n/torch.linalg.norm(n,dim = 1).unsqueeze(1)
    # 对投影平面法向量进行等分采样
    n_array = torch.linspace(n_min,n_max,n_gap+2).to(device)
    n_array = n_array.repeat(batch_size,distance_gap,1)
    n_euqal_array = (n_array[:,:,1:] + n_array[:,:,0:-1])*0.5
    
    n_euqal_array_resample = sample_method(weight,n_euqal_array)

    n_equal_map = n_euqal_array_resample.unsqueeze(3)*n.unsqueeze(1).unsqueeze(2)
    # 计算不同随机法向量之间的间隔
    # a=torch.tensor(n_min).float().unsqueeze(0)
    # print(a.shape)
    start_n = n.unsqueeze(1).unsqueeze(2)*torch.tensor(n_min).float().to(device)
    start_n = start_n * torch.ones(batch_size,distance_gap,1,3).to(device)
    n_random_map_temp = torch.cat((start_n,n_equal_map),dim=2)
    n_delta = torch.norm(n_random_map_temp[:,:,0:-1,:]-n_equal_map,dim=3)

    # 计算所有需要输入网络的坐标
    xyz = doppler_map[torch.arange(batch_size),doppler_num,:].unsqueeze(1).unsqueeze(2) + distance_map.unsqueeze(2) + n_equal_map
    # xyz = distance_map.unsqueeze(2) + n_random_map
    # # 可视化输入坐标
    # for i in range(40):
    #     xyz_visual = xyz[i,:,:,:].view(-1,3).detach().cpu()
    #     x, y, z = xyz_visual[:, 0], xyz_visual[:, 1], xyz_visual[:, 2]

    #     # 创建一个3D散点图
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     sc = ax.scatter(x, y, z, c=z, cmap='viridis')

    #     # 添加颜色条
    #     plt.colorbar(sc)

    #     # 设置轴标签
    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')

    #     ax.axis('equal')

    #     # 显示图形
    #     plt.show()

    xyz_coding = positon_code_xyz(xyz)
    LOS_coding = position_code_LOS(LOS)
    LOS_coding = ((LOS_coding.unsqueeze(1).unsqueeze(2))*torch.ones(batch_size,distance_gap,n_gap+1,27).to(device)).view(-1,27)
    xyzLOS_coding = torch.cat((xyz_coding,LOS_coding),dim=1)
    output = model(xyzLOS_coding)
    output = output.view(batch_size,distance_gap,n_gap+1,2)
    Ti = torch.cumprod(torch.exp(-output[:,:,:,0]**2*distance_delta),dim=1)
    distance_profile = torch.sum(output[:,:,:,0]*output[:,:,:,1]*Ti*n_delta,dim=2)

    return distance_profile

def positon_code_xyz(xyz):
    code_len = 10
    batch_size,distance,n,dimension = xyz.shape
    xyz = xyz.view(-1,dimension)
    xyz = xyz/100
    position_coding = torch.zeros_like(xyz).to(device)
    position_coding = position_coding.repeat(1,code_len*2)
    div_term = 2 ** torch.arange(0,code_len,step=1).to(device)
    position_coding[:,0::2] = torch.sin((xyz.unsqueeze(1) * math.pi * div_term.unsqueeze(1).unsqueeze(0)).view(batch_size*distance*n,-1))
    position_coding[:,1::2] = torch.cos((xyz.unsqueeze(1) * math.pi * div_term.unsqueeze(1).unsqueeze(0)).view(batch_size*distance*n,-1))
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


def natural_sort_key(s):
    # 分割字符串中的数字并将它们转换为整数
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def loaddata(folder_path):
    '''
    输入文件夹路径，输出数据集
    '''
    # 获取文件夹中的所有文件和子文件夹
    items = os.listdir(folder_path)
    # 过滤出所有文件（排除子文件夹）
    files = [item for item in items if os.path.isfile(os.path.join(folder_path, item))]
    files_sorted = sorted(files, key=natural_sort_key)
    #载入数据
    images = []
    LOS_dirs = []
    omegas = []
    for file in files_sorted:
        full_path = folder_path+"/"+file
        data = np.load(full_path)
        image = torch.from_numpy(data['image']).to(device)
        LOS_dir = torch.from_numpy(data['LOS']).to(device)
        omega = torch.from_numpy(data['rotation_axis']).to(device)
        images.append(image)
        LOS_dirs.append(LOS_dir)
        omegas.append(omega)
    max_pixel = [torch.max(image) for image in images]
    max_pixel_all = max(max_pixel, key=lambda x: x.item())
    images_normalize = [image/max_pixel_all for image in images]
    return images_normalize,LOS_dirs,omegas

def random_sample(images,LOS_dirs,omegas,batch_size):
    temp = random.sample(range(30000), batch_size)
    data_num = [x//100 for x in temp]
    doppler_numbers = [x % 100 for x in temp]
    LOS_dirs_batch = [LOS_dirs[x] for x in data_num]
    omegas_batch = [omegas[x] for x in data_num]
    range_profile_batch = [images[x][y,:] for x,y in zip(data_num,doppler_numbers)]

    omegas_batch_tensor = torch.stack(omegas_batch).to(device)
    LOS_dirs_batch_tensor = torch.stack(LOS_dirs_batch).to(device)
    range_profile_batch_tensor = torch.stack(range_profile_batch).to(device)
    doppler_profil_num_tensor = torch.tensor(doppler_numbers).to(device)
    return omegas_batch_tensor,LOS_dirs_batch_tensor,range_profile_batch_tensor,doppler_profil_num_tensor

def picture_sample(images,LOS_dirs,omegas,batch_size):
    temp_num = [39]
    temp = [t*100 for t in temp_num] + np.arange(100)
    data_num = [x//100 for x in temp]
    doppler_numbers = [x % 100 for x in temp]
    
    LOS_dirs_batch = [LOS_dirs[x] for x in data_num]
    omegas_batch = [omegas[x] for x in data_num]
    range_profile_batch = [images[x][y,:] for x,y in zip(data_num,doppler_numbers)]

    omegas_batch_tensor = torch.stack(omegas_batch).to(device)
    LOS_dirs_batch_tensor = torch.stack(LOS_dirs_batch).to(device)
    range_profile_batch_tensor = torch.stack(range_profile_batch).to(device)
    doppler_profil_num_tensor = torch.tensor(doppler_numbers).long().to(device)

    # range_image = range_profile_batch_tensor.detach().cpu()
    # plt.imshow(range_image)
    # plt.show()
    
    return omegas_batch_tensor,LOS_dirs_batch_tensor,range_profile_batch_tensor,doppler_profil_num_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 载入数据
folder_path = './asteroid_image/Geographos_sinc_0sigma'
images,LOS_dirs,omegas = loaddata(folder_path)

#载入模型
model = NeRF(input_ch = 63, input_ch_views = 27, use_viewdirs = True).to(device)
model.load_state_dict(torch.load('./model_state_dict33.pth'))
model_fine = NeRF(input_ch = 63, input_ch_views = 27, use_viewdirs = True).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 调节学习率函数
def adjust_learning_rate(optimizer,epoch,lr = 5e-7):
    lr = lr * (0.8**(epoch//1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# 旋转速度
omega_real = math.pi/900
losses = []
for epoch in range(10000):
    # 对数据进行随机采样，得到给定batch_size的数据集
    omegas_batch_tensor,LOS_dirs_batch_tensor,range_profile_batch_tensor,doppler_profil_num_tensor = random_sample(images,LOS_dirs,omegas,batch_size = 40)
    # 对该方向的数据进行渲染
    distance_profile_batch,weight = batchrender(omegas_batch_tensor*omega_real,LOS_dirs_batch_tensor,model,doppler_profil_num_tensor)
    # 精细网络渲染
    distance_profile_batch_fine = render_fine(omegas_batch_tensor*omega_real,LOS_dirs_batch_tensor,model_fine,doppler_profil_num_tensor,weight)

    optimizer.zero_grad()
    adjust_learning_rate(optimizer,epoch,lr=1e-5)


    # range_image1 = distance_profile_batch.detach().cpu()
    # plt.figure()
    # plt.imshow(range_image1)
    # plt.colorbar()


    # range_image2 = range_profile_batch_tensor.detach().cpu()
    # plt.figure()
    # plt.imshow(range_image2)
    # plt.colorbar()
    # plt.show()

    # range_image3 = distance_profile_batch_fine.detach().cpu()
    # plt.figure()
    # plt.imshow(range_image3)
    # plt.colorbar()
    # plt.show()

    loss = torch.sum((distance_profile_batch-range_profile_batch_tensor)**2) + torch.sum((distance_profile_batch_fine-range_profile_batch_tensor)**2)


    if epoch % 2000 == 0:
        torch.save(model_fine.state_dict(), 'model_state_dict_finenet3.pth')
        torch.save(losses, 'loss_list_finenet3.pth')

    
    loss.backward()
    optimizer.step()


    print("Current learning rate: ", optimizer.param_groups[0]['lr'] , end=' ')
    print("Current epoch:",epoch,end=' ')
    print("Current loss:",loss)
    losses.append(loss)


