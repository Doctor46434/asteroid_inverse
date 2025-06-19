# 尝试更改了旋转速度和位置编码的版本

import torch
import numpy as np
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import math
import os
import random
import matplotlib.pyplot as plt
import re
from typing import Union

# 位置编码函数
def get_embedder(multires, include_input=True):
    """
    位置编码器
    """
    embed_kwargs = {
        'include_input': include_input,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class SirenLayer(nn.Linear):
    def __init__(self, input_dim, out_dim, *args, is_first=False, **kwargs):
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = 30
        self.c = 6
        super().__init__(input_dim, out_dim, *args, **kwargs)
        self.activation = Sine(self.w0)

    # override
    def reset_parameters(self) -> None:
        # NOTE: in offical SIREN, first run linear's original initialization, then run custom SIREN init.
        #       hence the bias is initalized in super()'s reset_parameters()
        super().reset_parameters()
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.weight.uniform_(-w_std, w_std)

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out
    

class ImplicitSurface(nn.Module):
    def __init__(self,
                 W=256,
                 D=8,
                 skips=[4],
                 W_geo_feat=256,
                 input_ch=3,
                 radius_init=1.0,
                 obj_bounding_size=2.0,
                 geometric_init=True,
                 embed_multires=6,
                 weight_norm=True,
                 use_siren=False,
                 ):
        """
        隐式表面网络，输出SDF值和几何特征
        """
        super().__init__()
        self.radius_init = radius_init
        self.register_buffer('obj_bounding_size', torch.tensor([obj_bounding_size]).float())
        self.geometric_init = geometric_init
        self.D = D
        self.W = W
        self.W_geo_feat = W_geo_feat
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
            self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool))
        self.skips = skips
        self.use_siren = use_siren
        self.embed_fn, input_ch = get_embedder(embed_multires)

        surface_fc_layers = []
        # 网络结构: D+1层
        for l in range(D+1):
            # 决定输出维度
            if l == D:
                if W_geo_feat > 0:
                    out_dim = 1 + W_geo_feat
                else:
                    out_dim = 1
            elif (l+1) in self.skips:
                out_dim = W - input_ch  # 减少跳跃连接前的输出维度
            else:
                out_dim = W
                
            # 决定输入维度
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first = (l==0))
                else:
                    # beta=100很重要，确保初始输出形成球体
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # 几何初始化
            if geometric_init and not use_siren:
                # 球形初始化，类似于SAL/IDR
                if l == D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias, -radius_init) 
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):], 0.0)
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                layer = nn.utils.parametrizations.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def forward(self, x: torch.Tensor, return_h = False):
        """前向传播，返回SDF值和几何特征"""
        x = self.embed_fn(x)
        
        h = x
        for i in range(self.D):
            if i in self.skips:
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)
        
        out = self.surface_fc_layers[-1](h)
        
        if self.W_geo_feat > 0:
            h = out[..., 1:]
            out = out[..., :1].squeeze(-1)
        else:
            out = out.squeeze(-1)
        if return_h:
            return out, h
        else:
            return out
    
    def forward_with_nablas(self, x: torch.Tensor, has_grad_bypass: bool = None):
        """计算占用场值、法向量和几何特征"""
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        
        # 将梯度计算与主计算图分离
        x_temp = x.detach().clone()  # 创建不连接到原计算图的副本
        
        # 临时启用梯度计算，仅用于计算法向量
        with torch.enable_grad():
            x_temp.requires_grad_(True)
            implicit_surface_val, h = self.forward(x_temp, return_h=True)
            
            # 计算梯度
            nabla = autograd.grad(
                outputs=implicit_surface_val,
                inputs=x_temp,
                grad_outputs=torch.ones_like(implicit_surface_val, device=x_temp.device),
                create_graph=False,  # 不创建用于二阶导数的计算图
                retain_graph=False,  # 不保留图用于多次求导
                only_inputs=True
            )[0]
        
        # 在主计算图上运行一次前向传播
        if has_grad:
            # 训练模式下，需要保留梯度关系
            original_val, original_h = self.forward(x, return_h=True)
            return original_val, nabla.to(x.device), original_h
        else:
            # 推理模式，直接复用结果
            return implicit_surface_val.detach(), nabla.detach(), h.detach()

class RadianceNet(nn.Module):
    def __init__(self,
        D=4,
        W=256,
        skips=[],
        W_geo_feat=256,
        embed_multires=6,
        embed_multires_view=4,
        use_view_dirs=True,
        weight_norm=True,
        use_siren=False,):
        """辐射网络，根据位置、视角、法线和几何特征计算颜色"""
        super().__init__()
        
        input_ch_pts = 3
        input_ch_views = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W
        self.use_view_dirs = use_view_dirs
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        if use_view_dirs:
            self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
            in_dim_0 = input_ch_pts + input_ch_views + 3 + W_geo_feat
        else:
            in_dim_0 = input_ch_pts + W_geo_feat
        
        fc_layers = []
        # 网络结构: D+1层
        for l in range(D + 1):
            # 决定输出维度
            if l == D:
                out_dim = 1  # RGB颜色
            else:
                out_dim = W
            
            # 决定输入维度
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l==0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU())
            
            if weight_norm:
                layer = nn.utils.parametrizations.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        view_dirs: torch.Tensor, 
        normals: torch.Tensor, 
        geometry_feature: torch.Tensor):
        """计算辐射场"""
        x = self.embed_fn(x)
        if self.use_view_dirs:
            view_dirs = self.embed_fn_view(view_dirs)
            radiance_input = torch.cat([x, view_dirs, normals, geometry_feature], dim=-1)
        else:
            radiance_input = torch.cat([x, geometry_feature], dim=-1)
        
        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h
    
# UniSURF模型
class UniSURF(nn.Module):
    def __init__(self, 
                 D=8, 
                 W=256, 
                 D_rgb=4, 
                 W_rgb=256, 
                 skips=[4], 
                 skips_rgb=[], 
                 embed_multires=6, 
                 embed_multires_view=4, 
                 use_view_dirs=True, 
                 weight_norm=True, 
                 use_siren=False):
        """UniSURF模型，结合ImplicitSurface和RadianceNet"""
        super().__init__()
        
        # 几何特征维度
        W_geo_feat = W_rgb if D_rgb > 0 else 0
        
        # 创建ImplicitSurface
        self.implicit_surface = ImplicitSurface(
            W=W, 
            D=D, 
            skips=skips, 
            W_geo_feat=W_geo_feat,
            embed_multires=embed_multires, 
            geometric_init=True,
            weight_norm=weight_norm,
            use_siren=use_siren
        )
        
        # 创建RadianceNet
        if D_rgb > 0:
            self.radiance_net = RadianceNet(
                D=D_rgb,
                W=W_rgb, 
                skips=skips_rgb,
                W_geo_feat=W_geo_feat, 
                embed_multires=embed_multires,
                embed_multires_view=embed_multires_view,
                use_view_dirs=use_view_dirs,
                weight_norm=weight_norm,
                use_siren=use_siren
            )
        else:
            self.radiance_net = None
    
    def forward(self, x, view_dirs=None):
        """
        UniSURF前向传播
        返回: radiances(RGB颜色), occ(占用场强度), nablas(法向量)
        """
        # 计算占用场值、法向量和几何特征
        occ, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        
        # 计算辐射强度
        if self.radiance_net is not None and view_dirs is not None:
            # 标准化法向量
            normals = nablas / (torch.norm(nablas, dim=-1, keepdim=True) + 1e-7)
            radiances = self.radiance_net(x, view_dirs, normals, geometry_feature)
        else:
            # 如果没有辐射网络或视角方向，返回默认颜色
            radiances = torch.ones_like(x)
            
        return radiances, occ, nablas
    
    @staticmethod
    def get_opacity_from_surface(imp_surface: Union[torch.Tensor, np.ndarray]):
        # DVR'logits (+)inside (-)outside; logits here, (+)outside (-)inside.
        if isinstance(imp_surface, torch.Tensor):
            odds = torch.exp(-1. * imp_surface)
            opacity = odds / (1 + odds)
        else:
            odds = np.exp(-1. * imp_surface)
            opacity = odds / (1 + odds)
        return opacity

def batchrender(omega,LOS,model,doppler_num):
    '''
    omega为一个[bs,3]变量，指向旋转轴方向，模值为角速度
    LOS为一个[bs,3]变量，方向为视线方向指向物体，模值为1
    model是nerf模型，将一个已经进行位置编码后的位置和视线向量输入进model,可以返回这个位置的体积密度和散射系数
    doppler_num为一个[bs]变量，确定了渲染后光线所在的位置
    '''

    # print('step4')
    # print(torch.cuda.memory_allocated())
    # 确定回波波长
    fc = torch.tensor([9.7e9]).to(device)
    c = torch.tensor([299792458]).to(device)
    lambda0 = c/fc
    # 确定网格参数
    distance_max = 0.60
    distance_min = -0.60
    distance_gap = 100
    doppler_max = 0.15
    doppler_min = -0.15
    doppler_gap = 100
    n_max = 0.60
    n_min = -0.60
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
    # 等间距对距离向采样
    distance = torch.linspace(distance_min,distance_max,distance_gap).to(device)
    distance = distance.repeat(batch_size,1)
    distance_delta = torch.tensor((distance_max-distance_min)/distance_gap).to(device)
    distance_map = distance.unsqueeze(2)*LOS.unsqueeze(1)
    # # 非等间距对距离向采样
    # distance_array = torch.linspace(distance_min,distance_max,distance_gap+1).to(device)
    # distance_array = distance_array.repeat(batch_size,1)
    # distance_random_array = distance_array[:,0:-1] + (distance_array[:,1:]-distance_array[:,0:-1])*torch.rand(batch_size,distance_gap).to(device)
    # distance_random_map = LOS.unsqueeze(1)*distance_random_array.unsqueeze(2)
    # start_distance = LOS.unsqueeze(1)*torch.tensor(distance_min).float().to(device)
    # start_distance = start_distance * torch.ones(batch_size,1,3).to(device)
    # distance_ran


    doppler = torch.linspace(doppler_min,doppler_max,doppler_gap).repeat(batch_size,1).to(device)
    doppler = doppler*4/LOSomega_sin_angel.unsqueeze(1)
    # print(lambda0/omega_norm.unsqueeze(1)/2)
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
    # 输入xyz坐标的shape为[batch_size,distance_gap,n_gap,3]
    xyz = doppler_map[torch.arange(batch_size),doppler_num,:].unsqueeze(1).unsqueeze(2) + distance_map.unsqueeze(2) + n_random_map
    view_dirs = LOS.unsqueeze(1).unsqueeze(2).repeat(1, distance_gap, n_gap, 1)
    radiances, occ, nablas = model(xyz,view_dirs)
    occ_normalize = model.get_opacity_from_surface(occ)
    # 在dim = 1的起始添加一个1
    shifted_transparency = torch.cat(
    [
        torch.ones([occ_normalize.shape[0], 1, occ_normalize.shape[2]], device=device),
        1.0 - occ_normalize + 1e-10,
    ], dim= 1)
    # 计算可见性权重
    visibility_weights = occ_normalize *\
        torch.cumprod(shifted_transparency, dim=1)[...,:-1, :]
    radiances = radiances.squeeze(-1)
    # 计算颜色
    distance_profile = torch.sum(visibility_weights * radiances, dim=2)

    return distance_profile

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
    files = [item for item in items if os.path.isfile(os.path.join(folder_path, item)) and item.endswith('.npz')]
    files_sorted = sorted(files, key=natural_sort_key)
    #载入数据
    images = []
    LOS_dirs = []
    omegas = []
    max_pixel = []
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

def random_sample(images,LOS_dirs,omegas,batch_size,image_hight = 100,image_width = 97,image_num = 150):
    temp = random.sample(range(image_num*image_hight), batch_size)
    data_num = [x//image_hight for x in temp]
    doppler_numbers = [x % image_hight for x in temp]
    LOS_dirs_batch = [LOS_dirs[x] for x in data_num]
    omegas_batch = [omegas[x] for x in data_num]
    range_profile_batch = [images[x][y,:] for x,y in zip(data_num,doppler_numbers)]

    omegas_batch_tensor = torch.stack(omegas_batch).to(device)
    LOS_dirs_batch_tensor = torch.stack(LOS_dirs_batch).to(device)
    range_profile_batch_tensor = torch.stack(range_profile_batch).to(device)
    doppler_profil_num_tensor = torch.tensor(doppler_numbers).to(device)
    return omegas_batch_tensor,LOS_dirs_batch_tensor,range_profile_batch_tensor,doppler_profil_num_tensor

def picture_sample(images,LOS_dirs,omegas,batch_size):
    temp_num = random.sample(range(1),1)
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

# def compute_eikonal_samples(model, batch_size=1000):
#     """单独采样点用于计算Eikonal损失"""
#     # 在[-0.6, 0.6]范围内随机采样点
#     samples = torch.rand(batch_size, 3).to(device) * 1.2 - 0.6
#     samples.requires_grad_(True)
    
#     # 随机视角方向
#     random_dirs = torch.randn(batch_size, 3).to(device)
#     random_dirs = random_dirs / torch.norm(random_dirs, dim=1, keepdim=True)
    
#     # 对点和方向进行编码
#     samples_encoded = positon_code_xyz(samples.view(1, 1, batch_size, 3)).view(batch_size, 63)
#     dirs_encoded = position_code_LOS(random_dirs).view(batch_size, 15)
    
#     # 组合输入
#     model_input = torch.cat([samples_encoded, dirs_encoded], dim=1)
    
#     # 前向传播
#     output = model(model_input)
#     density = output[:, 0]  # 只取密度值
    
#     # 计算梯度
#     gradients = torch.autograd.grad(
#         outputs=density.sum(),
#         inputs=samples,
#         create_graph=True,
#         retain_graph=True
#     )[0]
    
#     # 计算Eikonal损失
#     gradients_norm = torch.norm(gradients, dim=-1)
#     eikonal_loss = ((gradients_norm - 1) ** 2).mean()
    
#     return eikonal_loss

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 载入数据
folder_path = './dataset/sys_data/Geo/test01'

# 生成保存路径
experiment_name = 'experiment11'
if not os.path.exists('./model_unsurf/'+ experiment_name):
    os.makedirs('./model_unsurf/'+ experiment_name)

images,LOS_dirs,omegas = loaddata(folder_path)

#载入模型
model = UniSURF().to(device)

# # 指定预训练模型的路径
# pretrained_model_path = '/DATA/disk1/Instant-ngp/model_unsurf/experiment02/model_state_dict.pth'  # 修改为您的预训练模型路径

# if os.path.exists(pretrained_model_path):
#     print(f"正在加载预训练模型: {pretrained_model_path}")
#     model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
#     print("预训练模型加载成功!")
# else:
#     print(f"找不到预训练模型: {pretrained_model_path}，将使用随机初始化")

optimizer = optim.Adam(model.parameters(), lr=5e-7)
# model.load_state_dict(torch.load('./model_state_dict14.pth'))

def adjust_learning_rate(optimizer,epoch,lr = 5e-7):
    lr = lr * (0.8**(epoch//1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_eikonal_weight(epoch, initial_weight=5e-5, min_weight=1e-5):
    """随时间减小Eikonal损失的权重"""
    return max(initial_weight * (0.9 ** (epoch // 1000)), min_weight)

# 旋转速度
omega_real = math.pi/900
losses = []

# 数据
image_hight = 100
image_width = 100
image_num = 37 *1

for epoch in range(20000):
    # 对数据进行随机采样，得到给定batch_size的数据集
    omegas_batch_tensor,LOS_dirs_batch_tensor,range_profile_batch_tensor,doppler_profil_num_tensor = random_sample(images,LOS_dirs,omegas,batch_size = 40,image_num = image_num,image_hight = image_hight,image_width = image_width)
    # 对该方向的数据进行渲染
    distance_profile_batch = batchrender(omegas_batch_tensor*omega_real,LOS_dirs_batch_tensor,model,doppler_profil_num_tensor)
    # # 单独计算Eikonal损失
    # eikonal_loss = compute_eikonal_samples(model, batch_size=1000)
    # # 调整Eikonal损失的权重
    # weight_eikonal = adjust_eikonal_weight(epoch)
    optimizer.zero_grad()
    # 计算损失函数
    # distance_profile_batch_detach = distance_profile_batch.detach()
    # loss = torch.sum((distance_profile_batch-range_profile_batch_tensor)**2) + weight_eikonal * eikonal_loss
    loss = torch.sum((distance_profile_batch-range_profile_batch_tensor)**2)
    
    adjust_learning_rate(optimizer,epoch,lr=5e-4)


    if epoch % 20 == 0:
        print("Current learning rate: ", optimizer.param_groups[0]['lr'] , end=' ')
    if epoch % 2000 == 0:
        torch.save(model.state_dict(), './model_unsurf/'+ experiment_name + '/model_state_dict.pth')
        torch.save(losses, './model_unsurf/'+ experiment_name + '/loss_list.pth')
        # 生成txt文档

        # 保存一个txt文件，用于解释当前的实验参数
        with open('./model_unsurf/'+ experiment_name + '/experiment_params.txt', 'a') as f:
            f.write(f'实验参数: {experiment_name}\n')
            # 记录数据集路径
            f.write(f'数据集路径: {folder_path}\n')
        print("Model has been saved successfully!")
        
    loss.backward()
    optimizer.step()

    distance_profile_batch = None

    if epoch % 100 == 0:
        print(f"Allocated memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"Max allocated memory: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")

    # for name,param in model.named_parameters():
    #     print(param.grad)

    print("Current epoch:",epoch,end=' ')
    print("Current loss:",loss)
    losses.append(loss)

# 记得要更改的实验参数有
# 1.模型和损失函数的名称
# 2.数据集的路径
# 3.随机选择的数据集的数量
# 4.如果同时运行两个程序，还需要更改device的选择

# # 记录实验参数
# import logging

# # 配置日志记录
# logging.basicConfig(filename='experiment1.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# def log_experiment_params(params):
#     logging.info(f'实验参数: {params}')

# # 示例实验参数
# experiment_params = {
#     'experiment_name': 'ISAR_NERF_30*3image',
#     'experiment_num': '48',
# }

# 记录实验参数
# log_experiment_params(experiment_params)
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
