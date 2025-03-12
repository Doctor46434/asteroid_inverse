def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    # 成像平面多普勒以及距离向参数
    parser.add_argument('--range_start',type=float,default=-10)
    parser.add_argument('--range_stop',type=float,default=10)
    parser.add_argument('--range_step',type=int,default=105)
    parser.add_argument('--doppler_start',type=float,default=-5)
    parser.add_argument('--doppler_stop',type=float,default=5)
    parser.add_argument('--doppler_step',type=int,default=79)

    return parser
