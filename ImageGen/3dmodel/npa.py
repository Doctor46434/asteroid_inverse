import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import trimesh
import datetime
from scipy.spatial.transform import Rotation as R
import math

# 加载OBJ模型
def load_model(obj_path):
    mesh = trimesh.load(obj_path)
    return mesh

# 计算欧拉角随时间的变化（使用欧拉方程求解）
def solve_euler_equations(I_ratios, w0, tspan, dt):
    """
    求解欧拉方程
    
    参数:
    I_ratios - 惯性矩比 [Is/Il, Ii/Il, 1]
    w0 - 初始角速度 [ω₀ₛ, ω₀ᵢ, ω₀ₗ] (deg/day)
    tspan - 时间范围 (天)
    dt - 时间步长 (天)
    
    返回:
    t - 时间点
    w - 对应时间点的角速度
    """
    # 转换为弧度/天
    w0_rad = np.radians(w0)
    
    # 设置惯性矩
    I = np.array([I_ratios[0], I_ratios[1], 1.0])
    
    # 时间点
    t = np.arange(0, tspan, dt)
    n_steps = len(t)
    
    # 初始化角速度数组
    w = np.zeros((n_steps, 3))
    w[0] = w0_rad
    
    # 欧拉方程数值积分 (使用四阶龙格-库塔方法)
    for i in range(1, n_steps):
        # 当前角速度
        w_current = w[i-1]
        
        # RK4方法
        k1 = dt * euler_derivative(w_current, I)
        k2 = dt * euler_derivative(w_current + 0.5*k1, I)
        k3 = dt * euler_derivative(w_current + 0.5*k2, I)
        k4 = dt * euler_derivative(w_current + k3, I)
        
        # 更新角速度
        w[i] = w_current + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, w

def euler_derivative(w, I):
    """计算欧拉方程导数"""
    dwdt = np.zeros(3)
    dwdt[0] = ((I[1] - I[2]) * w[1] * w[2]) / I[0]
    dwdt[1] = ((I[2] - I[0]) * w[2] * w[0]) / I[1]
    dwdt[2] = ((I[0] - I[1]) * w[0] * w[1]) / I[2]
    return dwdt

# 使用角速度计算姿态变化
def compute_orientation(euler0, w, t, dt):
    """
    计算随时间的刚体方向
    
    参数:
    euler0 - 初始欧拉角 [φ₀, θ₀, ψ₀] (度)
    w - 角速度随时间变化 (弧度/天)
    t - 时间点 (天)
    dt - 时间步长 (天)
    
    返回:
    orientations - 旋转矩阵序列
    """
    n_steps = len(t)
    
    # 初始化旋转矩阵序列
    orientations = np.zeros((n_steps, 3, 3))
    
    # 初始旋转 (欧拉角 -> 旋转矩阵)
    r0 = R.from_euler('ZXZ', np.radians(euler0))
    orientations[0] = r0.as_matrix()
    
    # 当前旋转
    r_current = r0
    
    for i in range(1, n_steps):
        # 当前角速度
        w_body = w[i-1]  # 体轴系中的角速度
        
        # 角速度到旋转增量
        angle = np.linalg.norm(w_body) * dt
        if angle > 0:
            axis = w_body / np.linalg.norm(w_body)
            dr = R.from_rotvec(angle * axis)
            
            # 更新旋转
            r_current = r_current * dr
            orientations[i] = r_current.as_matrix()
        else:
            orientations[i] = orientations[i-1]
    
    return orientations

# 可视化函数
def visualize_rotation(mesh, orientations, fps=30, duration=10):
    """
    创建旋转模型的动画
    
    参数:
    mesh - trimesh对象
    orientations - 旋转矩阵序列
    fps - 每秒帧数
    duration - 动画持续时间 (秒)
    """
    # 设置图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取网格信息
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    # 计算边界框大小，用于设置坐标轴范围
    max_range = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) / 2
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
    
    # 设置坐标轴范围
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('非主轴自转可视化')
    
    # 初始化3D网格
    collection = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                                triangles=faces, cmap='viridis', alpha=0.8)
    
    # 添加惯性主轴
    axis_length = max_range / 2
    
    # 短轴 (z) - 红色
    short_axis, = ax.plot([0, 0], [0, 0], [0, axis_length], 'r-', linewidth=3)
    # 中轴 (y) - 绿色
    mid_axis, = ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=3)
    # 长轴 (x) - 蓝色
    long_axis, = ax.plot([0, axis_length], [0, 0], [0, 0], 'b-', linewidth=3)
    
    # 角速度向量 - 黄色
    ang_vel, = ax.plot([0, 0], [0, 0], [0, 0], 'y-', linewidth=2)
    
    # 总帧数
    n_frames = int(fps * duration)
    
    # 选择要显示的方向
    step = len(orientations) // n_frames
    selected_orientations = orientations[::step][:n_frames]
    
    # 计算角速度方向
    angular_velocities = compute_angular_velocities(selected_orientations, t[::step][:n_frames])
    
    def update(frame):
        # 获取当前旋转矩阵
        rot_matrix = selected_orientations[frame]
        
        # 变换顶点
        rotated_vertices = np.dot(vertices, rot_matrix.T)
        
        # 更新网格
        ax.clear()
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'非主轴自转可视化 (帧 {frame}/{n_frames-1})')
        
        # 绘制网格
        ax.plot_trisurf(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], 
                      triangles=faces, cmap='viridis', alpha=0.8)
        
        # 变换并绘制主轴
        # 短轴 (z)
        s_axis = np.dot(np.array([[0, 0, 0], [0, 0, axis_length]]), rot_matrix.T)
        ax.plot(s_axis[:, 0], s_axis[:, 1], s_axis[:, 2], 'r-', linewidth=3, label='短轴')
        
        # 中轴 (y)
        m_axis = np.dot(np.array([[0, 0, 0], [0, axis_length, 0]]), rot_matrix.T)
        ax.plot(m_axis[:, 0], m_axis[:, 1], m_axis[:, 2], 'g-', linewidth=3, label='中轴')
        
        # 长轴 (x)
        l_axis = np.dot(np.array([[0, 0, 0], [axis_length, 0, 0]]), rot_matrix.T)
        ax.plot(l_axis[:, 0], l_axis[:, 1], l_axis[:, 2], 'b-', linewidth=3, label='长轴')
        
        # 角速度向量
        if frame < len(angular_velocities):
            ang_vel_dir = angular_velocities[frame]
            if np.linalg.norm(ang_vel_dir) > 0:
                ang_vel_dir = ang_vel_dir / np.linalg.norm(ang_vel_dir) * axis_length
                av_points = np.array([[0, 0, 0], ang_vel_dir])
                ax.plot(av_points[:, 0], av_points[:, 1], av_points[:, 2], 'y-', linewidth=2, label='角速度')
        
        ax.legend()
        return []
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=n_frames, blit=True)
    
    # 显示动画
    plt.show()
    
    return ani

def compute_angular_velocities(orientations, times):
    """计算全局坐标系中的角速度向量"""
    angular_velocities = []
    
    for i in range(1, len(orientations)):
        dt = times[i] - times[i-1]
        if dt > 0:
            # 计算旋转差异
            dR = np.dot(orientations[i], orientations[i-1].T)
            
            # 转换为旋转向量
            r = R.from_matrix(dR)
            rotvec = r.as_rotvec()
            
            # 角速度 = 旋转向量 / 时间
            ang_vel = rotvec / dt
            
            # 转换到全局坐标系
            global_ang_vel = np.dot(orientations[i-1], ang_vel)
            angular_velocities.append(global_ang_vel)
        else:
            angular_velocities.append(np.zeros(3))
    
    # 添加初始角速度（假设与第一个计算值相同）
    if angular_velocities:
        angular_velocities.insert(0, angular_velocities[0])
    
    return np.array(angular_velocities)

# 主程序
if __name__ == "__main__":
    # 加载OBJ模型
    model_path = "Toutatis Radar-based.obj"  # 替换为实际模型路径
    model = load_model(model_path)
    
    # 初始参数
    phi0 = 103  # 进动角 (度)
    theta0 = 97  # 章动角 (度)
    psi0 = 134   # 自转角 (度)
    euler0 = [phi0, theta0, psi0]
    
    w0_s = 20.7  # 短轴角速度 (度/天)
    w0_i = 31.3  # 中轴角速度 (度/天)
    w0_l = 98.0  # 长轴角速度 (度/天)
    w0 = [w0_s, w0_i, w0_l]
    
    Is_Il = 3.22  # 短轴/长轴惯性比
    Ii_Il = 3.09  # 中轴/长轴惯性比
    I_ratios = [Is_Il, Ii_Il, 1.0]
    
    # 模拟时间参数
    tspan = 10  # 模拟天数
    dt = 0.01    # 时间步长 (天)
    
    # 求解欧拉方程
    t, w = solve_euler_equations(I_ratios, w0, tspan, dt)
    
    # 计算方向随时间变化
    orientations = compute_orientation(euler0, w, t, dt)
    
    # 可视化
    ani = visualize_rotation(model, orientations, fps=30, duration=10)