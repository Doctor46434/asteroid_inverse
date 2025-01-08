import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
import time

def legendre_polynomial(l, x):
    """
    递归计算勒让德多项式 P_l(x)
    l: 阶数
    x: 自变量
    """
    if l == 0:
        return torch.ones_like(x)
    elif l == 1:
        return x
    else:
        P_l_minus_1 = legendre_polynomial(l - 1, x)
        P_l_minus_2 = legendre_polynomial(l - 2, x)
        P_l = ((2 * l - 1) * x * P_l_minus_1 - (l - 1) * P_l_minus_2) / l
        return P_l

def associated_legendre_polynomial(l, m, x):
    """
    递归计算缔合勒让德多项式 P_l^m(x)
    l: 阶数
    m: 次数
    x: 自变量
    """
    if m == 0:
        return legendre_polynomial(l, x)
    elif l == m:
        P_mm = (-1) ** m * (1 - x ** 2) ** (m / 2)
        return P_mm
    elif l == m + 1:
        P_mm = associated_legendre_polynomial(m, m, x)
        P_lm = x * (2 * m + 1) * P_mm
        return P_lm
    else:
        P_lm_1 = associated_legendre_polynomial(l - 1, m, x)
        P_lm_2 = associated_legendre_polynomial(l - 2, m, x)
        P_lm = ((2 * l - 1) * x * P_lm_1 - (l + m - 1) * P_lm_2) / (l - m)
        return P_lm


def spherical_harmonic(l, m, theta, phi):
    """
    计算球谐函数 Y_l^m(\theta, \phi)
    l: 球谐函数的阶数
    m: 球谐函数的次数
    theta: polar angle (纬度)
    phi: azimuthal angle (经度)
    """
    # 球谐函数的归一化系数
    K_lm = torch.sqrt((2 * l + 1) / (4 * math.pi) * torch.tensor(math.factorial(l - abs(m)) / math.factorial(l + abs(m)), dtype=theta.dtype, device=theta.device))
    
    # Associated Legendre polynomials
    P_lm = associated_legendre_polynomial(l, abs(m), torch.cos(theta))
    
    # 球谐函数的实部和虚部
    if m > 0:
        Y_lm = K_lm * P_lm * torch.cos(m * phi)
    elif m < 0:
        Y_lm = K_lm * P_lm * torch.sin(abs(m) * phi)
    else:
        Y_lm = K_lm * P_lm
    
    # # 可视化
    # x = Y_lm * torch.sin(theta) * torch.cos(phi)
    # y = Y_lm * torch.sin(theta) * torch.sin(phi)
    # z = Y_lm * torch.cos(theta)

    # x = x.reshape(-1)
    # y = y.reshape(-1)
    # z = z.reshape(-1)

    # vertex = torch.stack((x,y,z))
    # vertex = vertex.t()
    # vertex = vertex.detach().cpu()
    # vertex = vertex.numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_title(f'l={l}m={m}')
    # ax.scatter(vertex[:,0], vertex[:,1], vertex[:,2], c='r', marker='o')
    # ax.scatter(0, 0, 0, c='b', marker='o')
    # plt.axis('equal')
    # plt.show()
    return Y_lm

def spherical_module(sph_cofficient):
    [phi,theta] = torch.meshgrid(torch.linspace(0,2*math.pi,64),torch.linspace(0,math.pi,32), indexing='xy')
    phi = phi.to(device)
    theta = theta.to(device)
    a = 50
    b = 20
    c = 20
    x = 1/a * torch.sin(theta) * torch.cos(phi)
    y = 1/b * torch.sin(theta) * torch.sin(phi)
    z = 1/c * torch.cos(theta)
    R = torch.sqrt(1/(x**2 + y**2 + z**2)).to(device)
    l_max = round(math.sqrt(len(sph_cofficient))-1)
    for l in range(-1,l_max+1,1):
        for m in range(-l,l+1,1):
            R = R + sph_cofficient[l**2+l+m]*spherical_harmonic(l,m,theta,phi)
    
    x = R * torch.sin(theta) * torch.cos(phi)
    y = R * torch.sin(theta) * torch.sin(phi)
    z = R * torch.cos(theta)

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    vertex = torch.stack((x,y,z))
    vertex = vertex.t()
    # #可视化
    # vertex = vertex.detach().cpu()
    # vertex = vertex.numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(vertex[:,0], vertex[:,1], vertex[:,2], c='r', marker='o')
    # ax.scatter(0, 0, 0, c='b', marker='o')
    # plt.axis('equal')
    # plt.show()
    return vertex

# 找到每个点的邻域点
def find_neighbors(points, k=10):
    # 简单的暴力搜索来找到最近的k个邻居点（可以使用更高效的方法，如KD树）
    num_points = points.size(0)
    neighbors = []
    min_distance_threshold = 0.01  # 距离阈值
    for i in range(num_points):
        distances = torch.norm(points - points[i], dim=1)
        mask = distances > min_distance_threshold
        filtered_tensor = distances[mask]
        filtered_indices = torch.nonzero(mask, as_tuple=False)
        knn = torch.topk(filtered_tensor,k + 1, largest=False)[1]
        original_knn = filtered_indices[knn]
        original_knn = original_knn.t().squeeze(0)
        neighbors.append(original_knn)
    return neighbors

# 计算法向量(球谐函数版本)
def compute_normals_sph(points, neighbors):
    normals = []
    centroid = torch.tensor([0,0,0],dtype=torch.float32).to(device)
    # centroid = torch.mean(points,dim=0)
    for i, point in enumerate(points):
        neighbor_points = points[neighbors[i]]
        centroid = neighbor_points.mean(dim=0)
        cov_matrix = (neighbor_points - centroid).t().mm(neighbor_points - centroid)
        eigvals, eigvecs = torch.linalg.eig(cov_matrix)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        normal = -eigvecs[:, torch.argmin(eigvals)]
        normal = normal/torch.norm(normal)
        # normal = normal*torch.norm(point)**2*torch.acos(torch.clamp(point[2]/torch.norm(point),-0.99999,0.99999))
        # if torch.dot(normal, point - centroid) < 0:
        #     normal = -normal
        normals.append(normal)
    normals = torch.stack(normals)
    # 加权
    distances_squared = torch.sum(points ** 2, dim=1)

    # 计算每个点的角度 φ 和 sin(φ)
    # 由于点在3D空间中，φ 角可以表示为 arccos(z / |r|)， 这里 r 是从原点出发到点的矢量
    r_magnitude = torch.sqrt(distances_squared)
    phi = torch.acos(torch.clamp(torch.abs(points[:, 2] / r_magnitude),-0.99999,0.99999))  # φ = arccos(z / |r|)
    sin_phi = phi

    # 缩放法向量
    scaling_factors = distances_squared * sin_phi
    scaling_factors = scaling_factors.unsqueeze(1)  # 将 scaling_factors 从 [2048] 变为 [2048, 1]
    scaled_normals = normals * scaling_factors
    
    for i,point in enumerate(points):
        if torch.dot(scaled_normals[i,:], points[i,:]) < 0:
            scaled_normals[i,:] = -scaled_normals[i,:]
    # #可视化
    # normals = normals.detach().cpu()
    # normals = normals.numpy()
    # points = points.detach().cpu()
    # points = points.numpy()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # # 绘制点云
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=1)
    # ax.quiver(points[:, 0], points[:, 1], points[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], color='r')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.axis('equal')
    # plt.show()

    row_normal = torch.norm(scaled_normals,dim=1)
    max_normal = torch.max(row_normal)
    normal_normalized = scaled_normals/max_normal
    return normal_normalized

# 计算法向量(普通点云模型)
def compute_normals_point(points, neighbors):
    normals = []
    centroid = torch.tensor([0,0,0],dtype=torch.float32).to(device)
    # centroid = torch.mean(points,dim=0)
    for i, point in enumerate(points):
        neighbor_points = points[neighbors[i]]
        centroid = neighbor_points.mean(dim=0)
        cov_matrix = (neighbor_points - centroid).t().mm(neighbor_points - centroid)
        eigvals, eigvecs = torch.linalg.eig(cov_matrix)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        normal = -eigvecs[:, torch.argmin(eigvals)]
        normal = normal/torch.norm(normal)
        # normal = normal*torch.norm(point)**2*torch.acos(torch.clamp(point[2]/torch.norm(point),-0.99999,0.99999))
        # if torch.dot(normal, point - centroid) < 0:
        #     normal = -normal
        normals.append(normal)
    normals = torch.stack(normals)

    # 不进行缩放法向量
    scaled_normals = normals
    
    for i,point in enumerate(points):
        if torch.dot(scaled_normals[i,:], points[i,:]) < 0:
            scaled_normals[i,:] = -scaled_normals[i,:]
    # #可视化
    # normals = normals.detach().cpu()
    # normals = normals.numpy()
    # points = points.detach().cpu()
    # points = points.numpy()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # # 绘制点云
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=1)
    # ax.quiver(points[:, 0], points[:, 1], points[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], color='r')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.axis('equal')
    # plt.show()

    row_normal = torch.norm(scaled_normals,dim=1)
    max_normal = torch.max(row_normal)
    normal_normalized = scaled_normals/max_normal
    return normal_normalized

def polyfit(array_xyz,PRF):

    # 每分钟一个采样拟合到每秒一个采样
    x = array_xyz[:, 0]
    y = array_xyz[:, 1]
    z = array_xyz[:, 2]

    # 生成参数 t，用于多项式拟合
    t = np.linspace(0, 1, len(x))

    # 选择多项式的阶数（可以根据需要调整）
    degree = 5

    # 对 x(t), y(t), z(t) 分别进行多项式拟合
    poly_x = np.polyfit(t, x, degree)
    poly_y = np.polyfit(t, y, degree)
    poly_z = np.polyfit(t, z, degree)

    # 生成更多的 t 值进行插值
    t_fine = np.linspace(0, 1, len(x)*60*PRF)

    # 使用多项式计算拟合结果
    x_fine = np.polyval(poly_x, t_fine)
    y_fine = np.polyval(poly_y, t_fine)
    z_fine = np.polyval(poly_z, t_fine)
    

    # # 进行可视化
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 原数据点
    # ax.plot(x, y, z, 'ro', label='Original Data')

    # # 拟合曲线
    # ax.plot(x_fine, y_fine, z_fine, 'b-', label='Fitted Curve')

    # ax.legend()
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()

    return np.column_stack((x_fine,y_fine,z_fine))
def read_obj_file(filename):
    """
    读取一个OBJ文件中的顶点数据
    Args:
        filename (str): OBJ文件的文件名
    Returns:
        vertices (torch.ndarray): 顶点数据的数组 (N, 3)
    """
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])]) 
    return torch.tensor(vertices).to(device)

def ISAR_imaging(time_start,vextex,normal,horizon,rotation_state,num):
    c = torch.tensor([299792458]).to(device) #光速
    Tcoh = torch.tensor([10]).to(device) #合成孔径时间
    PRF = torch.tensor([20]).to(device) #脉冲频率
    fc = torch.tensor([9.7e9]).to(device) #载频
    Tp = torch.tensor([5e-4]).to(device) #脉冲宽度
    B = torch.tensor([30e6]).to(device) #带宽
    k = B/Tp #调频率
    fs  = 1.2*B #采样频率
    Tr = 1/PRF #脉冲间隔
    Na = torch.round(PRF*Tcoh)
    Na = Na + torch.remainder(Na,2)
    Tcoh = Na*Tr
    lambda0 = c/fc

    P = rotation_state[0]
    gamma = rotation_state[1]
    phi = rotation_state[2]

    axis_x = torch.sin(gamma)*torch.cos(phi)
    axis_y = torch.sin(gamma)*torch.sin(phi)
    axis_z = torch.cos(gamma)
    rotation_axis = torch.column_stack((axis_x,axis_y,axis_z))
    rotation_axis = rotation_axis.squeeze(0)
    Ry = torch.tensor([
        [torch.cos(gamma), 0, torch.sin(gamma)],
        [0, 1, 0],
        [-torch.sin(gamma), 0, torch.cos(gamma)]
    ], dtype=torch.float32).t().to(device)

    Rz = torch.tensor([
        [torch.cos(phi), -torch.sin(phi), 0],
        [torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 1]
    ], dtype=torch.float32).t().to(device)
    vextex = vextex @ Ry @ Rz
    normal = normal @ Ry @ Rz

    # #可视化
    # vextex = vextex.detach().cpu()
    # vextex = vextex.numpy()
    # rotation_axis = rotation_axis.detach().cpu()
    # rotation_axis = rotation_axis.numpy()
    # rotation_axis = rotation_axis * 100
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(vextex[:,0], vextex[:,1], vextex[:,2], c='r', marker='o')
    # ax.scatter(0, 0, 0, c='b', marker='o')
    # ax.quiver(0,0,0, rotation_axis[0], rotation_axis[1], rotation_axis[2], color='r')
    # plt.axis('equal')
    # plt.show()

    # 慢时间
    st = time_start + torch.linspace(-Tr.item()*Na.item()/2,Tr.item()*(Na.item()-1)/2,Na.item()).to(device)
    vextex = vextex.t()
    normal = normal.t()
    # #可视化
    # rotation_axis = rotation_axis.detach().cpu()
    # rotation_axis = rotation_axis.numpy()
    # rotation_axis = rotation_axis * 100

    R_box = torch.empty((Na.item(),vextex.shape[1])).to(device)
    sigma_box = torch.empty((Na.item(),vextex.shape[1])).to(device)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"代码块执行时间: {elapsed_time:.4f} 秒")
    for i in range(Na.item()):
        theta = 1/P*st[i]*math.pi/1800
        
        # 计算旋转矩阵 Rotmat
        Rotmat = torch.tensor([
            [axis_x**2+(1-axis_x**2)*torch.cos(theta),      axis_x*axis_y*(1-torch.cos(theta))-axis_z*torch.sin(theta), axis_x*axis_z*(1-torch.cos(theta))+axis_y*torch.sin(theta)],
            [axis_x*axis_y*(1-torch.cos(theta))+axis_z*torch.sin(theta), axis_y**2+(1-axis_y)*torch.cos(theta),      axis_y*axis_z*(1-torch.cos(theta))-axis_x*torch.sin(theta)],
            [axis_x*axis_z*(1-torch.cos(theta))-axis_y*torch.sin(theta), axis_y*axis_z*(1-torch.cos(theta))+axis_x*torch.sin(theta), axis_z**2+(1-axis_z**2)*torch.cos(theta)]
        ], dtype=torch.float32).to(device)
        vextex_rot = Rotmat @ vextex
        normal_rot = Rotmat @ normal

        vextex_rot = vextex_rot.detach().cpu()
        vextex_rot = vextex_rot.numpy()
        rotation_axis = rotation_axis.detach().cpu()
        rotation_axis = rotation_axis.numpy()
        rotation_axis = rotation_axis * 100
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vextex_rot[0,:], vextex_rot[1,:], vextex_rot[2,:], c='r', marker='o')
        ax.scatter(0, 0, 0, c='b', marker='o')
        ax.quiver(0,0,0, rotation_axis[0], rotation_axis[1], rotation_axis[2], color='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.axis('equal')
        plt.show()


        # 将视线方向变为单位向量
        horizon_i = horizon[i,:]
        horizon_i = horizon_i/torch.norm(horizon_i)
        # 计算斜距
        # R = torch.norm(horizon_i.repeat(vextex.shape[1],1).t() + vextex_rot,dim=0)
        # R = R - torch.norm(horizon_i.repeat(vextex.shape[1],1).t(),dim=0)
        R = horizon_i.t() @ vextex_rot
        R_box[i,:] = R
        # 计算散射系数
        
        sigma = torch.clamp(-horizon_i.t() @ normal_rot,min=0)
        # sigma = -horizon_i.t() @ normal_rot
        sigma = sigma.squeeze(0)
        sigma_box[i,:] = sigma
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"代码块执行时间: {elapsed_time:.4f} 秒")
    Dopple_domian = -2*(R_box[Na.item()-1,:]-R_box[0,:])/Tcoh/lambda0
    Range_domain = (R_box[Na.item()-1,:]+R_box[0,:])/2
    sigma = torch.mean(sigma_box,dim=0)

    # 数据点的采样频率
    fs_range_window = torch.tensor([20]).to(device)
    fs_dopple_window = torch.tensor([20]).to(device)
    # 窗长度
    N_range_window = torch.tensor([100]).to(device)
    N_dopple_window = torch.tensor([100]).to(device)
    # 距离向分辨率
    range_res = c/2/B/10
    theta = 2*math.pi*Tcoh/P/3600
    # 方位向分辨率
    dopple_res = lambda0/2/theta/10
    [Range_map,Dopple_map] = torch.meshgrid(torch.linspace(-60,60,100).to(device),torch.linspace(-12,12,100).to(device), indexing='xy')
    z = torch.zeros_like(Range_map).to(device)
    for i in range(vextex.shape[1]):
        z = z + torch.abs(sigma[i]*sinc_windowed(1/range_res*(Range_map-Range_domain[i]),fs_range_window,N_range_window)*sinc_windowed(1/dopple_res*(Dopple_map-Dopple_domian[i]),fs_dopple_window,N_dopple_window))
        # z = z + torch.abs(sigma[i]*torch.sinc(1/range_res*(Range_map-Range_domain[i]))*torch.sinc(1/dopple_res*(Dopple_map-Dopple_domian[i])))
    z = z/torch.max(z)
    # z_numpy = z.detach().cpu()
    # z_numpy = z_numpy.numpy()
    # filename = '.\nerf_data\Geographos_image' + time_start + ''
    # torch.save(z, 'Geographos_image.pt')
    # return z
    
    z = z.detach().cpu()
    z = z.numpy()

    theta_LOS = -1/P*time_start*math.pi/1800
    Rotmat = torch.tensor([
        [axis_x**2+(1-axis_x**2)*torch.cos(theta_LOS),      axis_x*axis_y*(1-torch.cos(theta_LOS))-axis_z*torch.sin(theta_LOS), axis_x*axis_z*(1-torch.cos(theta_LOS))+axis_y*torch.sin(theta_LOS)],
        [axis_x*axis_y*(1-torch.cos(theta_LOS))+axis_z*torch.sin(theta_LOS), axis_y**2+(1-axis_y)*torch.cos(theta_LOS),      axis_y*axis_z*(1-torch.cos(theta_LOS))-axis_x*torch.sin(theta_LOS)],
        [axis_x*axis_z*(1-torch.cos(theta_LOS))-axis_y*torch.sin(theta_LOS), axis_y*axis_z*(1-torch.cos(theta_LOS))+axis_x*torch.sin(theta_LOS), axis_z**2+(1-axis_z**2)*torch.cos(theta_LOS)]
    ], dtype=torch.float32).to(device)
    LOS_real = Rotmat @ horizon_i
    LOS_real = LOS_real.detach().cpu()
    LOS_real = LOS_real.numpy()
    rotation_axis = rotation_axis.detach().cpu()
    rotation_axis = rotation_axis.numpy()
    filename = './asteroid_image/Geographos/Nerf_data' + str(num)

    # np.savez(filename, image=z, LOS = LOS_real, rotation_axis = rotation_axis)

    extent = [-60, 60, -12, 12]
    plt.imshow(z, extent=extent, aspect='auto', cmap='viridis')  # 'viridis' 是一种色图，你可以根据需求更换
    plt.colorbar()  # 添加颜色条以显示色图的标度
    plt.title("groundtruth")
    plt.xlabel("range/m")
    plt.ylabel("doppler/Hz")
    plt.savefig('real_image.png')
    plt.show()

    # 进行可视化
    # R_box = R_box.detach().cpu()
    # R_box = R_box.numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    
    # # 原数据点
    # ax.plot(np.linspace(0,20,200), R_box[:,100], 'ro', label='R0_data')
    # plt.show()
    # #将散射系数转换为复系数，方便后面计算
    # sigma_box = sigma_box.to(dtype = torch.complex64)
    # 通过传统成像方法成像
    # Rmax = torch.max(R)
    # Rmin = torch.min(R)
    # Length = Rmax-Rmin
    # win1 = 0.1*Tp
    # win2 = Tp
    # Nr = torch.round((2*Length/c+win1+win2)*fs)
    # Nr = torch.round(Nr + torch.remainder(Nr,2))
    # Tng = torch.linspace((-win1+2*Rmin/c).item(),(win2+2*Rmax/c).item(),round(Nr.item())).to(device)
    # xm = torch.empty(Na.item(),round(Nr.item()))
    # for i in range(Na.item()):
    #     td = 2*R_box[i,:]/c
    #     td_tn = Tng.repeat(vextex.shape[1],1) - td.repeat(round(Nr.item()),1).t()
    #     x = rectpuls(td_tn-Tp/2,Tp)*torch.exp(1j * math.pi * k * (td_tn - Tp / 2)**2)*(torch.exp(-2j * math.pi * fc * td).repeat(round(Nr.item()),1).t())
    #     xm[i,:] = sigma_box[i,:] @ x
    #     # x = torch.
    #     # #可视化
    #     # vextex_rot = vextex_rot.detach().cpu()
    #     # vextex_rot = vextex_rot.numpy()

        
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # ax.scatter(vextex_rot[0,:], vextex_rot[1,:], vextex_rot[2,:], c='r', marker='o')
    #     # ax.scatter(0, 0, 0, c='b', marker='o')
    #     # ax.quiver(0,0,0, rotation_axis[0,0], rotation_axis[0,1], rotation_axis[0,2], color='r')
    #     # plt.axis('equal')
    #     # plt.show()
    # return xm

def rectpuls(t, width):
    return ((t >= -width / 2) & (t <= width / 2)).float()

def sinc_windowed(x,fs,N):
    # 不考虑窗函数峰值点在采样范围外的情况
    return torch.sinc(x)*b_window(x,fs,N)
    
def b_window(x,fs,N):
    return ((x * fs < N/2) & (x * fs > -N/2)).float()*(0.42 - 0.5*torch.cos(2*math.pi*(x*fs+N/2)/(N-1)) + 0.08*torch.cos(4*math.pi*(x*fs+N/2)/(N-1)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vextex = read_obj_file('./asteroid_model/Geographos Radar-based, low-res(1).obj')
vextex = vextex *20
# vextex = spherical_module(self.sph_harm_cofficient)
neighbors = find_neighbors(vextex)
normals = compute_normals_point(vextex, neighbors)
# 旋转时间
rotation_time = torch.linspace(0,1800,20).to(device)
# 视线方向
LOS_dir = torch.tensor([[-math.sqrt(3)/2,0,-1/2],[-1,0,0],[-math.sqrt(3)/2,0,1/2]]).to(device)
LOS_dir = torch.tensor([[-math.sqrt(3)/2,0,-1/2],[-1,0,0],[-math.sqrt(3)/2,0,1/2]]).to(device)
# 物体转动周期与转轴指向
rotation_state = torch.tensor([[0.5], [0], [0]]).to(device)

for j in range(3):
    for i in range(20):
        z = ISAR_imaging(rotation_time[i], vextex, normals, LOS_dir[j,:].repeat(200,1).float().to(device), rotation_state,j*20+i)