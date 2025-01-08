import vtk
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
    xyz = xyz/100
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

def createScalarField(dim_x, dim_y, dim_z, model):
    """
    创建一个不对称的标量场。
    """
    scalar_field = torch.zeros((dim_x, dim_y, dim_z), dtype=float)
    x = torch.linspace(-dim_x/2,dim_x/2,dim_x).to(device)
    y = torch.linspace(-dim_y/2,dim_y/2,dim_y).to(device)
    z = torch.linspace(-dim_z/2,dim_z/2,dim_z).to(device)
    xyz = torch.stack([x.unsqueeze(1).unsqueeze(2).expand(dim_x,dim_y,dim_z) ,y.unsqueeze(0).unsqueeze(2).expand(dim_x,dim_y,dim_z) ,z.unsqueeze(0).unsqueeze(1).expand(dim_x,dim_y,dim_z)],-1)
    xyz = xyz.view(-1,3)
    xyz = positon_code_xyz(xyz)
    LOS = torch.tensor([1, 0, 0]).unsqueeze(0).expand(dim_x*dim_y*dim_z,3).to(device)
    LOS = position_code_LOS(LOS)
    xyzlos = torch.cat([xyz,LOS],dim=1)
    scalar_field = model(xyzlos)[...,0]
    scalar_field = scalar_field.detach().cpu()
    scalar_field = scalar_field.numpy()
    scalar_field = scalar_field.reshape(dim_x,dim_y,dim_z)

    return scalar_field

def numpyToVtkImageData(scalar_field):
    """
    将NumPy数组转换为VTK的ImageData格式。
    """
    dim_x, dim_y, dim_z = scalar_field.shape
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(dim_x, dim_y, dim_z)
    vtk_data.SetSpacing(1.0, 1.0, 1.0)

    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetNumberOfValues(scalar_field.size)
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetName("ScalarField")

    for i in range(scalar_field.size):
        vtk_array.SetValue(i, scalar_field.flat[i])

    vtk_data.GetPointData().SetScalars(vtk_array)
    return vtk_data

def main():



    # 参数设置
    dim_x, dim_y, dim_z = 120, 60, 120
    iso_value = 0.02  # 等值面值

    # 创建标量场
    model = NeRF(input_ch=63, input_ch_views=27, use_viewdirs=True).to(device)
    model.load_state_dict(torch.load('./model_state_dict50.pth'))
    model.eval()
    scalar_field = createScalarField(dim_x, dim_y, dim_z, model)

    # 将标量场转换为VTK的ImageData格式
    vtk_data = numpyToVtkImageData(scalar_field)

    # 使用Marching Cubes算法提取等值面
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_data)
    mc.SetValue(0, iso_value)
    mc.Update()

    # 获取提取的多边形数据
    polydata = mc.GetOutput()

    # 将多边形数据导出为OBJ文件
    obj_writer = vtk.vtkOBJWriter()
    obj_writer.SetFileName("./models/geo2.obj")  # 设置输出文件名
    obj_writer.SetInputData(polydata)
    obj_writer.Write()

    # 创建用于渲染的mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(mc.GetOutputPort())
    mapper.ScalarVisibilityOff()

    # 创建actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.3, 0.3)  # 设置颜色

    # 创建渲染器
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)  # 设置背景颜色

    # 创建渲染窗口
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    # 添加坐标轴
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(30, 30, 30)
    renderer.AddActor(axes)

    # 创建交互渲染器
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    # 开始渲染
    render_window.Render()
    iren.Start()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

if __name__ == "__main__":
    main()

