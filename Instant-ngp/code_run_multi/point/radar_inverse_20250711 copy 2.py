# 添加库函数
import os
import torch
import torch.nn as nn
import math
import torch.optim as optim
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_batch_individually
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from torch.autograd import gradcheck
import os

def gaussian_pdf(x, mu, sigma):
    const = 1.0
    exp = torch.exp(-0.5 * ((x - mu) / sigma)**2)
    return const * exp

class ISAR_render(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.c = torch.tensor([299792458.0], device=device)
        self.Tcoh = torch.tensor([10.0*16], device=device)
        self.PRF = torch.tensor([20.0], device=device)
        self.fc = torch.tensor([9.7e9], device=device)
        self.Tp = torch.tensor([5e-4], device=device)
        self.B = torch.tensor([30e7*16], device=device)
        self.Range_map,self.Dopple_map = torch.meshgrid(torch.linspace(-10,10,100).to(device),torch.linspace(-3,3,100).to(device), indexing='xy')
        self.lambda1 = self.c/self.fc
        self.RangeRes = self.c/self.B/2*5
        self.complex_i = torch.tensor([1j], dtype=torch.complex64, device=device)

    def forward(self, mesh, RadarLos, SpinAxis, Omega):
        # 输入
        # mesh为pytorch3d自带的结构，采样后采样点为：sampled_points [batchsize,N,3] point_normals [batchsize,N,3]
        # 雷达视线方向 RadarLos [batchsize,3]
        # 转轴 SpinAxis [batchsize,3]
        sampled_points,point_normals = sample_points_from_meshes(mesh, 8000, return_normals=True)
        DopplerAxis = torch.cross(RadarLos,SpinAxis,dim = -1)
        point_vel = torch.cross(SpinAxis.unsqueeze(1),sampled_points,dim = -1)
        point_vel_Radial = Omega * torch.sum(RadarLos.unsqueeze(1)*point_vel,dim=2)
        point_doppler = -2*point_vel_Radial/self.lambda1
        point_range = torch.sum(RadarLos.unsqueeze(1)*sampled_points,dim=2)
        point_Amp = -4*torch.sum(RadarLos.unsqueeze(1)*point_normals, dim=2)
        point_Amp = point_Amp
        point_Amp = torch.clamp(point_Amp,min=0.0,max=1.0)
        DopplerRes = self.lambda1/2/Omega/self.Tcoh

        # range_idx = torch.round
        
        # image_AllPoint = torch.sinc(1/self.RangeRes*(self.Range_map.unsqueeze(0).unsqueeze(1)-point_range.unsqueeze(2).unsqueeze(3))) * torch.sinc(1/DopplerRes*(self.Dopple_map.unsqueeze(0).unsqueeze(1)-point_doppler.unsqueeze(2).unsqueeze(3))) * torch.exp(-4*math.pi*self.complex_i/self.lambda1*point_range.unsqueeze(2).unsqueeze(3))
        # image_AllPoint = point_Amp.unsqueeze(2).unsqueeze(3)*torch.sinc(1/self.RangeRes*(self.Range_map.unsqueeze(0).unsqueeze(1)-point_range.unsqueeze(2).unsqueeze(3))) * torch.sinc(1/DopplerRes*(self.Dopple_map.unsqueeze(0).unsqueeze(1)-point_doppler.unsqueeze(2).unsqueeze(3))) * torch.exp(-4*math.pi*self.complex_i/self.lambda1*point_range.unsqueeze(2).unsqueeze(3))
        # # image_AllPoint = torch.abs(point_Amp.unsqueeze(2).unsqueeze(3)*torch.sinc(1/self.RangeRes*(self.Range_map.unsqueeze(0).unsqueeze(1)-point_range.unsqueeze(2).unsqueeze(3))) * torch.sinc(1/DopplerRes*(self.Dopple_map.unsqueeze(0).unsqueeze(1)-point_doppler.unsqueeze(2).unsqueeze(3))) * torch.exp(-4*math.pi*self.complex_i/self.lambda1*point_range.unsqueeze(2).unsqueeze(3)))
        image_AllPoint = point_Amp.unsqueeze(2).unsqueeze(3)*gaussian_pdf(point_range.unsqueeze(2).unsqueeze(3), self.Range_map.unsqueeze(0).unsqueeze(1), self.RangeRes) * gaussian_pdf(point_doppler.unsqueeze(2).unsqueeze(3), self.Dopple_map.unsqueeze(0).unsqueeze(1), DopplerRes)
        image = torch.sum(image_AllPoint,dim=1)

        return image

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(60, 0)
    ax.axis('equal')
    plt.show()

# 设置matplotlib的显示参数
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

# 设置需要更改的参数
# cuda编号
cuda_id = "cuda:3"

# Set the device
if torch.cuda.is_available():
    device = torch.device(cuda_id)
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")



# 读取路径
data_path = '/DATA/disk1/asteroid/asteroid_inverse/Instant-ngp/new_dataset/sys_data/geo40du_right'
# 数据量
batch = 60
# 雷达视线方向角度
RadarLos = torch.tensor([-math.cos(40*math.pi/180),0,math.sin(40*math.pi/180)], device=device)
# 视线方向集合
theta = torch.linspace(0,2*math.pi-2*math.pi/batch,batch).to(device)
# 输出路径
output_path = '/DATA/disk1/asteroid/asteroid_inverse/Instant-ngp/new_dataset/result/wrong_angle_point/geo40du_right'



    
# 整体放缩系数
scale_all = 0.05


# 载入初始模型方式
# 选择一种载入初始模型的方法
flag = 2
if flag == 0:
    # 载入一个已有的Mesh模型
    trg_obj = './ImageGen/3dmodel/Geographos Radar-based, low-res(1).obj'
    # trg_obj = 'dolphin.obj'
    # trg_obj = 'wx_origin.obj'
    # 读取卫星各项参数
    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale *10

    # We construct a Meshes structure for the target mesh
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])


    # 为mesh添加材质信息
    verts_rgb = torch.ones_like(trg_mesh.verts_packed())*255  # 使用纯白色作为默认颜色
    verts_rgb = verts_rgb.unsqueeze(0)
    textures = TexturesVertex(verts_features=verts_rgb)
    trg_mesh.textures = textures

    src_mesh = trg_mesh
if flag == 1:
    # 载入双球模型

    # 创建初始模，两个球体
    sphere1 = ico_sphere(4, device)
    sphere2 = ico_sphere(4, device)

    # 将第一个球体放大到1.5倍
    vert1 = sphere2.verts_packed()
    vert1 = vert1 * 1.5
    sphere2 = Meshes(verts=[vert1], faces=[sphere2.faces_packed()])

    # 计算放大后的球体的最大x坐标
    max_x1 = sphere1.verts_packed()[:, 0].max()
    # 计算第二个球体的最小x坐标
    min_x2 = sphere2.verts_packed()[:, 0].min()

    # 计算平移量，确保两球体有适当重叠
    overlap = 0.1
    shift = max_x1 - min_x2 + overlap

    # 获取第二个球体的顶点并进行x方向平移
    verts2 = sphere2.verts_packed() + torch.tensor([shift, 0, 0], device=device)

    # 获取第一个球体的顶点（已经放大）
    verts1 = sphere1.verts_packed()

    # 合并两球体的顶点
    verts = torch.cat([verts1, verts2], dim=0)

    # 可以额外添加整体平移
    verts = verts + torch.tensor([-1.5, 0, 0], device=device)

    # 整体放缩
    verts = verts * 3

    # 合并面片，并更新第二个球体的面片索引
    faces1 = sphere1.faces_packed()
    faces2 = sphere2.faces_packed() + sphere1.verts_packed().shape[0]  # 更新索引

    # 合并面片数据
    faces = torch.cat([faces1, faces2], dim=0)

    # 创建黏连的球体网格
    src_mesh = Meshes(verts=[verts], faces=[faces])

if flag == 2:
    def create_ellipsoid(level, device, scale_factors=(1.5, 1.0, 0.7)):
        """
        创建椭球体
        
        参数:
            level: ico_sphere 的细分级别
            device: 计算设备
            scale_factors: (x, y, z) 缩放因子
        """
        # 创建基础球体
        sphere = ico_sphere(level, device)
        
        # 获取球体的顶点和面
        verts = sphere.verts_packed()
        faces = sphere.faces_packed()
        
        # 对顶点进行缩放以形成椭球体
        x_scale, y_scale, z_scale = scale_factors
        scaled_verts = verts.clone()
        scaled_verts[:, 0] *= x_scale  # x方向缩放
        scaled_verts[:, 1] *= y_scale  # y方向缩放
        scaled_verts[:, 2] *= z_scale  # z方向缩放
        
        # 创建新的网格
        ellipsoid = Meshes(verts=[scaled_verts], faces=[faces])
        
        return ellipsoid
    
    sphere1 = create_ellipsoid(4, device, scale_factors=(2.0, 1.0, 1.0))

    vert = sphere1.verts_packed()
    vert = vert * 3

    src_mesh = Meshes(verts=[vert], faces=[sphere1.faces_packed()])

import re

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


# 创建渲染器
ISAR_render1 = ISAR_render(device)

# 输入数据1
# image_batch = np.load('./2024wb_peizhun/2024wb.npz')
# image_input = image_batch['image_batch']

# 输入数据2
image_input,_,_ = loaddata(data_path)
# 将列表转换为torch
image_input = torch.stack(image_input, dim=0)
image_input = image_input.to(device)
# print(image_input.shape)
image_input = torch.abs(image_input)

# plt.imshow(image_input[0,:,:].detach().cpu(),cmap='hot')
# 数据归一化
max_values, _ = torch.max(image_input, dim=2)
max_values_dim1, _ = torch.max(max_values, dim=1)
image_trg = image_input/max_values_dim1.unsqueeze(1).unsqueeze(2)
# print(image_trg.shape)
# 第零维倒序
# image_trg = torch.flip(image_trg, dims=[0])

# 生成雷达视线方向
def vec_rot(vec,axis_x,axis_y,axis_z,theta):

    axis_x = axis_x.expand(theta.shape)
    axis_y = axis_y.expand(theta.shape)
    axis_z = axis_z.expand(theta.shape)

    c = torch.cos(theta)
    s = torch.sin(theta)
    one_c = 1 - c

    Rotmat = torch.stack([
        torch.stack([axis_x**2 * one_c + c, axis_x * axis_y * one_c - axis_z * s, axis_x * axis_z * one_c + axis_y * s], dim=-1),
        torch.stack([axis_x * axis_y * one_c + axis_z * s, axis_y**2 * one_c + c, axis_y * axis_z * one_c - axis_x * s], dim=-1),
        torch.stack([axis_x * axis_z * one_c - axis_y * s, axis_y * axis_z * one_c + axis_x * s, axis_z**2 * one_c + c], dim=-1)
    ], dim=-2)

    vec_rot = torch.matmul(Rotmat,vec.unsqueeze(1)).squeeze(2)

    return vec_rot

SpinAxis = torch.tensor([0,0,1.0], device=device)
Omega = torch.tensor([0.004532090125293*1], device=device)
SpinAxis = SpinAxis.unsqueeze(0)
Omega = Omega.unsqueeze(0)

# RadarLos = torch.tensor([-1/2,0,-math.sqrt(3)/2], device=device)
# theta = torch.linspace(0,math.pi-math.pi/batch,batch).to(device)

axis_x = torch.tensor([0.0], device=device)
axis_y = torch.tensor([0.0], device=device)
axis_z = torch.tensor([1.0], device=device)
omega_vec = torch.stack((axis_x.repeat(batch),axis_y.repeat(batch),axis_z.repeat(batch)),dim = 1)
# print(omega_vec)
Round_radar_los = vec_rot(RadarLos,axis_x,axis_y,axis_z,theta)
Round_radar_los_real = vec_rot(RadarLos,axis_x,axis_y,axis_z,-theta)

# print(Round_radar_los)
# print(Round_radar_los.shape)

# # 生成测试数据
# image_src = ISAR_render1(src_mesh, Round_radar_los[0:25], SpinAxis[0:25], Omega[0:25])
# # 取模归一化
# image_src = torch.abs(image_src)
# max1,_ = torch.max(image_src,dim=2)
# max2,_ = torch.max(max1,dim=1)
# image_src = image_src/max2.unsqueeze(1).unsqueeze(2)

deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

optimizer = torch.optim.Adam([deform_verts], lr=0.05)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000,eta_min=5e-4)


# Number of optimization steps
Niter = 4000
# Weight for the image loss
w_image = 0.0005
# Weight for mesh edge loss
w_edge = 0.35
# Weight for mesh normal consistency
w_normal = 2
# Weight for mesh laplacian smoothing
w_laplacian = 5
# Plot period for the losses
plot_period = 1000
loop = tqdm(range(Niter))

image_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []

for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()
    
    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    
    # 每轮选取三个视角进行训练
    random_numbers = np.random.choice(range(0, batch), 6, replace=False)
    image_src = ISAR_render1(new_src_mesh, Round_radar_los[random_numbers,:], SpinAxis, Omega)
    # 取模归一化
    image_src = torch.abs(image_src)
    max1,_ = torch.max(image_src,dim=2)
    max2,_ = torch.max(max1,dim=1)
    image_src = image_src/max2.unsqueeze(1).unsqueeze(2)
    image_trg_sample = image_trg[random_numbers,:,:]
    
    # 计算简单的mse
    loss_image = torch.sum((image_trg_sample - image_src)**2)
    
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)
    
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    
    # Weighted sum of the losses
    loss = loss_image * w_image + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
    
    # Print the losses
    loop.set_description('total_loss = %.6f' % loss)

    # # 显示当前损失和轮数，每五轮显示一次
    # if i % 5 == 0:
    #     print('Iter: %d, Loss: %.6f' % (
    #         i, loss.item(),))
    
    # Save the losses for plotting
    image_losses.append(float(loss_image.detach().cpu()))
    edge_losses.append(float(loss_edge.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))
    
    # # Plot mesh
    # if i % plot_period == 0:
    #     plot_pointcloud(new_src_mesh, title="iter: %d" % i)
    #     plt.figure()
    #     plt.imshow(image_src[0,:,:].detach().cpu(),cmap='hot')
    #     plt.colorbar()
    #     plt.figure()
    #     plt.imshow(image_trg_sample[0,:,:].detach().cpu(),cmap='hot')
    #     plt.colorbar()
        
    # Optimization step
    loss.backward()
    optimizer.step()
    scheduler.step()

fullname = output_path

# 生成保存路径
if not os.path.exists(fullname):
    os.makedirs(fullname) 

loss_image = loss_image

fig = plt.figure(figsize=(13, 5))
ax = fig.gca()
ax.plot([x*w_image for x in image_losses], label="image losses")
ax.plot([x*w_edge for x in edge_losses], label="edge loss")
ax.plot([x*w_normal for x in normal_losses], label="normal loss")
ax.plot([x*w_laplacian for x in laplacian_losses], label="laplacian loss")                                                                                                                                  
ax.legend(fontsize="16")
ax.set_xlabel("Iteration", fontsize="16")
ax.set_ylabel("Loss", fontsize="16")
ax.set_title("Loss vs iterations", fontsize="16")

# 将图片保存在指定路径
plt.savefig(fullname + "/loss.png", dpi=300, bbox_inches='tight')

from torchvision import transforms
from PIL import Image
import os

image_src = ISAR_render1(new_src_mesh, Round_radar_los[0:6,:], SpinAxis, Omega)
# 取模归一化
image_src = torch.abs(image_src)
max1,_ = torch.max(image_src,dim=2)
max2,_ = torch.max(max1,dim=1)
image_src = image_src/max2.unsqueeze(1).unsqueeze(2)
plt.figure()
plt.imshow(image_src[0,:,:].detach().cpu(),cmap='hot')



for i in range(batch):
    image_src = ISAR_render1(new_src_mesh, Round_radar_los[i:i+1,:], SpinAxis, Omega)
    image_src = torch.abs(image_src)
    max1,_ = torch.max(image_src,dim=2)
    max2,_ = torch.max(max1,dim=1)
    image_src = image_src/max2.unsqueeze(1).unsqueeze(2)
    image_save = image_src[0,:,:].detach().cpu()

    # 确保保存路径存在
    if not os.path.exists(fullname + "/npz"):
        os.makedirs(fullname + "/npz")
    # 将图片保存为npz文件
    np.savez(fullname + "/npz/image" + str(i) + ".npz", image=image_save.numpy())

    # if i%5==0:
    #     plt.figure()
    #     plt.imshow(image_save,cmap='hot')  

    array = image_save.squeeze(0).numpy() * 255  # 转换为数组并缩放到 0-255 范围
    array = array.astype('uint8')  # 将类型转换为 8 位整数

    # 创建一个 PIL 图片对象
    # 转换为hot图像
    # 表示hot图像
    # image = Image.fromarray(array, mode='')  
    image = Image.fromarray(array, mode='L')  # 'L' 模式表示灰度图像

    # 确定保存路径
    output_folder = fullname + "/images"
    image_name = 'image' + str(i) + '.png'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, image_name)

    # 保存图片
    image.save(output_path)

# 保存模型
save_obj(fullname + "/model.obj", verts=new_src_mesh.verts_packed(), faces=new_src_mesh.faces_packed())
