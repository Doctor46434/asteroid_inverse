import torch

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

            rho = self.rho_linear(h)
            outputs = torch.cat([rho, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

def get_position_encoding(x, d_model):
    """
    Generate position encoding based on input x with shape (batch_size, seq_len) and d_model.
    """
    batch_size, seq_len = x.shape
    position_encoding = torch.zeros(batch_size, seq_len, d_model*2)
    div_term = 2 ** torch.arange(0, d_model, step=1)
    temp = x.unsqueeze(2)
    # Apply sin to even indices in the array; 2i
    position_encoding[:, :, 0::2] = torch.sin(x.unsqueeze(2) * div_term)
    # Apply cos to odd indices in the array; 2i+1
    position_encoding[:, :, 1::2] = torch.cos(x.unsqueeze(2) * div_term)
    
    return position_encoding

def apply_position_encoding(x):
    batch_size, length = x.shape
    
    # Encode the first 3 positions with L=10 and add the original positions
    L1 = 10
    pos_encoding_first_part = get_position_encoding(x[:, :3], L1)
    x_first_encoded = torch.cat([x[:, :3].unsqueeze(2), pos_encoding_first_part], dim=2).view(batch_size, -1)
    
    # Encode the last 3 positions with L=4 and add the original positions
    L2 = 4
    pos_encoding_last_part = get_position_encoding(x[:, 3:], L2)
    x_last_encoded = torch.cat([x[:, 3:].unsqueeze(2), pos_encoding_last_part], dim=2).view(batch_size, -1)
    
    # Concatenate the encoded features
    x_encoded = torch.cat([x_first_encoded, x_last_encoded], dim=1)
    
    return x_encoded

# Example usage:
# x = torch.randn(2, 6) # Example input tensor with unknown batch size (bs=2 in this example)
# x_encoded = apply_position_encoding(x)
# print(x_encoded.shape) # Expected shape: [batch_size, 63+27]
# print(x_encoded) # Encoded tensor