def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # 本次实验的名称
    parser.add_argument('--experiment_name', type=str, default='experiment_4')
    # 数据集图片张数
    parser.add_argument('--image_num', type=int, default=60)
    # 数据集图片高度
    parser.add_argument('--img_height', type=int, default=100)
    # 数据集图片宽度
    parser.add_argument('--img_width', type=int, default=100)
    # 每次训练所使用的一维距离数量
    parser.add_argument('--batchsize', type=int, default=40)

    # NERF模型参数
    # 距离网格最大值
    parser.add_argument('--max_distance', type=float, default=0.6)
    # 距离网格最小值
    parser.add_argument('--min_distance', type=float, default=-0.6)
    # 距离网格间隔
    parser.add_argument('--distance_interval', type=int, default=100)
    # 多普勒网格最大值
    parser.add_argument('--max_doppler', type=float, default=0.15)
    # 多普勒网格最小值
    parser.add_argument('--min_doppler', type=float, default=-0.15)
    # 多普勒网格间隔
    parser.add_argument('--doppler_interval', type=float, default=100)
    # NSR网格最大值
    parser.add_argument('--max_nsr', type=float, default=0.30)
    # NSR网格最小值
    parser.add_argument('--min_nsr', type=float, default=-0.30)
    # NSR网格间隔
    parser.add_argument('--nsr_interval', type=float, default=120)

    # 是否使用距离向网格随机采样
    parser.add_argument('--random_distance', type=bool, default=False)
    # 输入的图片是否是经过多普勒标校后的结果
    parser.add_argument('--is_doppler', type=bool, default=True)
    # 是否使用位置编码
    parser.add_argument('--use_positional_encoding', type=bool, default=True)
    # 是否使用ISAR-nerf渲染规则
    parser.add_argument('--use_isar_nerf', type=bool, default=True)

    # cuda设备
    parser.add_argument('--device', type=str, default='cuda')
    # 数据集路径
    parser.add_argument('--dataset_path', type=str, default='/DATA/disk1/3dmodel/3dmodel/ISAR_NERF/asteroid_image_nerf_new/luoxuan_new')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-5)
    # 模型保存路径
    parser.add_argument('--model_path', type=str, default='model/model_8.pth')                                  #运行试验必改
    # 损失保存路径
    parser.add_argument('--loss_path', type=str, default='loss/loss_list_8.pth')                                #运行试验必改
    # 参数保存路径
    parser.add_argument('--config_path', type=str, default='model_parameter/config_params_8.json')                                #运行试验必改

    return parser