import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency, dwt_high_frequency


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class ConvNeXt(nn.Module):
    def __init__(self, block, in_chans=3, num_classes=1000,
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x_layer1 = self.downsample_layers[0](x)
        x_layer1 = self.stages[0](x_layer1)

        x_layer2 = self.downsample_layers[1](x_layer1)
        x_layer2 = self.stages[1](x_layer2)

        x_layer3 = self.downsample_layers[2](x_layer2)
        out = self.stages[2](x_layer3)

        return x_layer1, x_layer2, out



class ConvNeXt0(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, block, in_chans=3, num_classes=1000,
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        xc = x.permute(0, 2, 3, 1)
        xc = self.norm(xc)
        g, i, c = torch.split(self.fc1(xc), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        xo = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        xo = self.drop_path(xo)
        return xo.permute(0, 3, 1, 2) + x


class BlockRGB(nn.Module):
    def __init__(self, num_ch, k_sz=3, dropout_prob=0.1):
        super(BlockRGB, self).__init__()
        self.bn = nn.BatchNorm2d(num_ch)
        self.conv1 = nn.Conv2d(num_ch, num_ch // 2, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.op1 = nn.LeakyReLU(0.2)
        self.dyndc = SELayer(num_ch // 2)
        self.conv2 = nn.Conv2d(num_ch // 2, num_ch, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.op2 = nn.LeakyReLU(0.2)

        self.rconv1 = nn.Conv2d(num_channels=num_ch, out_channels=num_ch // 2, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)

        self.a1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.a2 = nn.Parameter(torch.tensor(1.0,  dtype=torch.float32), requires_grad=True)
        self.dropout1 = nn.Dropout(dropout_prob) if dropout_prob > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_prob) if dropout_prob > 0. else nn.Identity()

    def forward(self, x):
        xf = self.bn(x)
        xf = self.op1(self.conv1(xf))
        xf = self.dropout1(xf)
        xf += self.a1 * self.rconv1(x)
        xf = self.dyndc(xf)
        xf = self.dropout2(xf)
        xf = self.op2(self.conv2(xf))
        return xf + self.a2 * x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels=16, out_channels=32, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        # self.norm_1 = norm_layer(out_channels)
        self.norm_1 = nn.BatchNorm2d(out_channels)

        self.act_1 = nn.PReLU(init=0.2)

        self.conv_2 = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        # self.norm_2 = norm_layer(out_channels)
        self.norm_2 = nn.BatchNorm2d(out_channels)

        self.act_2 = nn.PReLU(init=0.2)


    def forward(self, xl, xr):

        xl = self.act_1(self.conv_1(xl))
        xl = self.norm_1(xl)

        xr = self.act_2(self.conv_2(xr))
        xr = self.norm_2(xr)
        return xl, xr





class CCAttention(nn.Module):
    def __init__(self, nf1, nf2, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.scale = (0.5 * (nf1  + nf2)) ** -0.5

        self.Q_conv = nn.Conv2d(nf1, num_heads, kernel_size=1, stride=1, padding=0)
        self.K_conv = nn.Conv2d(nf2, num_heads, kernel_size=1, stride=1, padding=0)
        self.V_conv = nn.Conv2d(nf1, num_heads, kernel_size=1, stride=1, padding=0)

        self.bias = nn.Parameter(torch.randn((1, 1, 1, nf2)), requires_grad=True)
        self.bias_selector = nn.Linear(nf2, num_heads)

        self.fan_out = nn.Conv2d(num_heads, nf1, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.randn((1, nf1, 1, 1)), requires_grad=True)
        self.norm1 = nn.LayerNorm(nf1)
        self.norm2 = nn.LayerNorm(nf2)

    def forward(self, x1, x2):
        x1 = self.norm1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = self.norm2(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        q = self.Q_conv(x1).permute(0, 2, 3, 1) # b h w nh
        k_t = self.K_conv(x2).permute(0, 2, 1, 3) # b h, nh, w (transposed)

        attn = torch.softmax(self.scale * (q.matmul(k_t)), dim = -1) # b, h, w, w

        v = self.V_conv(x1).permute(0, 2, 3, 1) # b, h, w, c
        map = attn.matmul(v) + self.bias_selector(self.bias) # b, h, w, c
        out = x1 + self.gamma * self.fan_out(map.permute(0, 3, 1, 2))
        return out


class AgentAttention(nn.Module):
    def __init__(self, nf_x, nf_ag, nf_w, head_dim=None):
        super().__init__()

        # Q  = x
        # K  = w_l
        # V = w_h
        # Ag = HS/V

        if head_dim is None:
            self.head_dim = nf_x // 8
        else:
            self.head_dim = head_dim

        # self.scale = self.head_dim ** -0.5
        self.scale = (0.5 * (nf_x + nf_ag)) ** -0.5
        self.gamma = nn.Parameter(torch.randn((1, nf_x, 1, 1)), requires_grad=True)

        self.k_conv = nn.Conv2d(nf_x, self.head_dim, kernel_size=1, stride=1, padding=0)
        self.q_conv = nn.Conv2d(nf_w, self.head_dim, kernel_size=1, stride=1, padding=0)
        self.v_conv = nn.Conv2d(nf_x + nf_w, self.head_dim, kernel_size=1, stride=1, padding=0)
        self.ag_conv = nn.Conv2d(nf_ag, self.head_dim, kernel_size=1, stride=1, padding=0)

        self.conv_alpha = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1, stride=1, padding=0)
        self.conv_omega = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1, stride=1, padding=0)

        self.norm1 = nn.LayerNorm(nf_x)
        self.norm2 = nn.LayerNorm(nf_ag)
        self.norm3 = nn.LayerNorm(nf_w)
        # self.norm4 = nn.LayerNorm(nf_w)

        self.dw_conv = nn.Conv2d(nf_x, nf_x, kernel_size=3, padding=1, groups=nf_x)
        self.fan_out = nn.Conv2d(self.head_dim, nf_x, kernel_size=1, stride=1, padding=0)

        self.alpha_proj = nn.Linear(self.head_dim, 1, bias=False)
        self.omega_proj = nn.Linear(self.head_dim, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x, ag, w_l, w_h=None):
        # Q  = w_l
        # K  = w_h
        # V = x
        # Ag = HS/V

        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        agn = self.norm2(ag.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        w_ln = self.norm3(w_l.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        xv_ln = self.v_conv(torch.cat((xn, w_ln), dim=1))
        xk_ln = self.k_conv(xn)
        w_ln = self.q_conv(w_ln)
        agn = self.ag_conv(agn)

        x_alpha = self.conv_alpha(xk_ln)
        x_omega = self.conv_omega(w_ln)

        b, _, h, w = xn.size()
        # xk_ln = xk_ln.view(b, self.head_dim, h * w)
        # agn = agn.view(b, self.head_dim, h * w)
        # xv_ln = xv_ln.view(b, self.head_dim, h * w)
        # w_ln = w_ln.view(b, self.head_dim, h * w)
        xk_ln = xk_ln.permute(0, 2, 3, 1) # b, h , w, nh
        agn = agn.permute(0, 2, 3, 1)
        w_ln = w_ln.permute(0, 2, 3, 1)
        xv_ln = xv_ln.permute(0, 2, 3, 1)

        alpha = self.avg_pool(x_alpha)
        omega = self.avg_pool(x_omega)

        #b, h, w, w
        attn_1 = torch.softmax(self.scale * (xk_ln.matmul(torch.transpose(agn, 2, 3))), dim=-1)

        # b, h, w, w
        attn_2 = torch.softmax(self.scale * (agn.matmul(torch.transpose(w_ln, 2, 3))), dim=-1)


        # (b, h, w, nh) =  (b, h, w, w)  x (b, h, w, nh)
        alpha  = self.alpha_proj(alpha.permute(0, 2, 3, 1))
        omega = self.omega_proj(omega.permute(0, 2, 3, 1))

        map = (alpha * attn_1 + omega * attn_2) @ xv_ln
        out = self.gamma * self.fan_out(map.permute(0, 3, 1, 2))  + self.dw_conv(xn)
        return out


class Blender(nn.Module):
    def __init__(self, in_channels, fan_in=0.25):
        super().__init__()

        self.ca = ChannelAttention(int(in_channels * fan_in))
        self.proj_in = nn.Conv2d(in_channels, int(in_channels * fan_in), kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(int(in_channels * fan_in), in_channels // 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.proj_out(self.ca(self.proj_in(x)))

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_weight=0.01):
        super().__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x = self.conv(x)
        x = self.project(self.upsample(x))
        return  x



class UpsampleLayer(nn.Module):
    def __init__(self, in_channels=96, out_channels=198):
        super().__init__()

        self.bl = Blender(in_channels)
        self.br = Blender(in_channels)

        self.ca_l = ChannelAttention(in_channels // 4)
        self.ca_r = ChannelAttention(in_channels // 4)

        self.sa_l = SpatialAttention(in_channels // 4)
        self.sa_r = SpatialAttention(in_channels // 4)

        self.conv_1_1 = nn.Conv2d(in_channels // 4,
                                  in_channels // 4,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  groups=in_channels // 4)

        self.conv_1_2 = nn.Conv2d(in_channels // 4,
                                  in_channels // 4,
                                  kernel_size=1,
                                  stride=1)

        self.conv_1 = nn.Conv2d(in_channels // 16,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.norm_1 = nn.BatchNorm2d(out_channels)

        self.act_1 = nn.PReLU(init=0.2)

        self.conv_2_1 = nn.Conv2d(in_channels // 4,
                                  in_channels // 4,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  groups=in_channels // 4)

        self.conv_2_2 = nn.Conv2d(in_channels // 4,
                                  in_channels // 4,
                                  kernel_size=1,
                                  stride=1)

        self.conv_2 = nn.Conv2d(in_channels // 16,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)


        self.norm_2 = nn.BatchNorm2d(out_channels)

        self.act_2 = nn.PReLU(init=0.2)

        self.upsample1 = nn.PixelShuffle(upscale_factor=2)
        self.upsample2 = nn.PixelShuffle(upscale_factor=2)

        self.shortcut_l = Shortcut(in_channels, out_channels)
        self.shortcut_r = Shortcut(in_channels, out_channels)

    def forward(self, xl, xr):
        xl_res = xl
        xr_res = xr

        xl = self.bl(xl)
        xr = self.br(xr)

        res_1 = self.act_1(self.conv_1_1(xl))
        res_1 = res_1 + xl
        res_1 = self.conv_1_2(res_1)
        res_1 = self.sa_l(self.ca_l(res_1))
        xl = (xl + res_1)
        xl = self.conv_1(self.upsample1(xl)) # .permute(0, 2, 3, 1)
        xl = self.norm_1(xl) # .permute(0, 3, 1, 2)

        res_2 = self.act_2(self.conv_2_1(xr))
        res_2 = res_2 + xr
        res_2 = self.conv_2_2(res_2)
        res_2 = self.sa_r(self.ca_r(res_2))
        xr = (xr + res_2)
        xr = self.conv_2(self.upsample2(xr)) # .permute(0, 2, 3, 1)
        xr = self.norm_2(xr) # .permute(0, 3, 1, 2)

        return xl + self.shortcut_r(xl_res), xr + self.shortcut_r(xr_res)


class ChannelAttention(nn.Module):
    def __init__(self, nf):
        super().__init__()

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf, nf // 8, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // 8, nf, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.ca(x)


class SpatialAttention(nn.Module):
    def __init__(self, nf):
        super().__init__()

        self.sa = nn.Sequential(
            nn.Conv2d(nf, nf // 8, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // 8, nf, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.sa(x)


class InnerBlock(nn.Module):
    def __init__(self, num_channels, wave_channels, scale):
        super().__init__()

        self.scale = scale

        self.l_way = GatedCNNBlock(num_channels)
        self.r_way = SELayer(num_channels) # GatedCNNBlock(num_channels)

        distiller_l = []
        distiller_r = []
        in_ch = 3

        nc2 = num_channels // 4
        for i in range(scale):
            out_channels = int((i + 1) / self.scale * nc2)
            distiller_l.append(nn.Conv2d(in_ch, out_channels, kernel_size=2, stride=2))
            distiller_l.append(nn.PReLU(init=0.2))
            distiller_l.append(nn.BatchNorm2d(out_channels))

            distiller_r.append(nn.Conv2d(in_ch, out_channels, kernel_size=2, stride=2))
            distiller_r.append(nn.PReLU(init=0.2))
            distiller_r.append(nn.BatchNorm2d(out_channels))

            in_ch = out_channels

        self.distiller_l = nn.Sequential(*distiller_l)
        self.distiller_r = nn.Sequential(*distiller_r)


        self.ca_l = AgentAttention(num_channels, in_ch, wave_channels)
        self.ca_r = AgentAttention(num_channels, in_ch, wave_channels)

        self.alpha_l = nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha_r = nn.Parameter(torch.ones(1), requires_grad=True)

        # self.splitter = Splitter(in_channels=3)
        self.splitter = Splitter2(in_channels=3)



    def forward(self, xl, wl_l, xr, wr_l, wl_h, wr_h, i):
        il, ir = self.splitter(i)
        il = self.distiller_l(il)
        ir = self.distiller_r(ir)

        xla = self.ca_l(xl, il, wl_l, wr_h)
        xra = self.ca_r(xr, ir, wr_l, wl_h)

        xl = self.alpha_l * xl + self.l_way(xla)
        xr = self.alpha_r * xr + self.r_way(xra)
        return xl, xr



class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.n_feats = n_fea_out

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, 2 * n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_map[:, :self.n_feats, : , :], illu_map[:, self.n_feats:, : , :]


class Splitter(nn.Module):
    def __init__(self, in_channels=3, kernel_size=5):
        super(Splitter, self).__init__()

        self.weights = nn.Parameter(1 / (kernel_size * kernel_size) * torch.ones((in_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)

    def forward(self, x):
        xl = torch.nn.functional.conv2d(x, self.weights, bias=None, stride=1, padding='same')
        xh = x - xl

        return xl, xh

class Splitter2(nn.Module):
    def __init__(self, in_channels=3, kernel_size=5):
        super(Splitter2, self).__init__()

        self.conv1 = nn.Conv2d(1, in_channels, kernel_size=kernel_size, padding='same', bias=True)
        self.conv2 = nn.Conv2d(2, in_channels, kernel_size=kernel_size, padding='same', bias=True)

    def rgb2hsl(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsl_h[cmax_idx == 3] = 0.
        hsl_h /= 6.

        hsl_l = (cmax + cmin) / 2.
        hsl_s = torch.empty_like(hsl_h)
        hsl_s[hsl_l == 0] = 0
        hsl_s[hsl_l == 1] = 0
        hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
        hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
        hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
        hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
        hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
        return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)

    def rgb2lab(self, rgb: torch.Tensor) -> torch.Tensor:
        xlab = kornia.color.rgb_to_lab(rgb)
        xlab[:, 0, :, :] /= 100.0
        xlab[:, 1, :, :] += 128.0
        xlab[:, 2, :, :] += 128.0
        xlab[:, 1, :, :] /= 255.0
        xlab[:, 2, :, :] /= 255.0
        return xlab

    def forward(self, x):
        x_hsl = self.rgb2hsl(x)
        # x_hsl = self.rgb2lab(x)

        xl = self.conv1(x_hsl[:, 2, :, :].unsqueeze(1))
        xr = self.conv2(x_hsl[:, :2, :, :])
        return xl, xr


class CC38(nn.Module):
    def __init__(self, num_feats=16, num_blocks=16):
        super(CC38, self).__init__()

        self.num_blocks = num_blocks

        self.encoder = ConvNeXt(Block, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                                drop_path_rate=0, layer_scale_init_value=1e-6, head_init_scale=1.0)
        checkpoint = torch.load('./weights/convnext_tiny_22k_1k_384.pth')
        self.encoder.load_state_dict(checkpoint["model"])

        self.splitter = Illumination_Estimator(num_feats)

        self.tail_l = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(28, 3, kernel_size=7, padding=0))
        self.tail_r = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(28, 3, kernel_size=7, padding=0))

        self.dwn1 = DownsampleLayer(3, 64)
        self.dwn2 = DownsampleLayer(64 + 16, 128)
        self.dwn3 = DownsampleLayer(128 + 32, 256)
        self.dwn4 = DownsampleLayer(256 + 64, 512)


        self.up1 = UpsampleLayer(512 + 128 + 384, 256)
        self.up2 = UpsampleLayer(512 + 192 + 64, 128)
        self.up3 = UpsampleLayer(256 + 96 + 32, 64)
        self.up4 = UpsampleLayer(128 + 16, 28)


        inner_stage = []
        for i in range(num_blocks):
            inner_stage.append(InnerBlock(512, 128, scale=4))

        self.inner_stage = nn.ModuleList(inner_stage)

        # dual domain branch
        self.dwtl_1 = DWT_transform(3, 16)
        self.dwtr_1 = DWT_transform(3, 16)

        self.dwtl_2 = DWT_transform(64, 32)
        self.dwtr_2 = DWT_transform(64, 32)

        self.dwtl_3 = DWT_transform(128, 64)
        self.dwtr_3 = DWT_transform(128, 64)

        self.dwtl_4 = DWT_transform(256, 128)
        self.dwtr_4 = DWT_transform(256, 128)

    def forward(self, input):
        x_f1, x_f2, x_f3 = self.encoder(input)
        # h//4, h // 8, h // 16
        # 256   512   1024
        xl, xr = self.splitter(input)
        xl1, xr1 = self.dwn1(xl, xr)

        wl_l, wh_l = self.dwtl_1(xl)
        wl_r, wh_r = self.dwtr_1(xr)
        xl2, xr2 = self.dwn2(torch.cat((xl1, wl_l), dim=1), torch.cat((xr1, wl_r), dim=1))

        wl_l1, wh_l1 = self.dwtl_2(xl1)
        wl_r1, wh_r1 = self.dwtr_2(xr1)
        xl3, xr3 = self.dwn3(torch.cat((xl2, wl_l1), dim=1), torch.cat((xr2, wl_r1), dim=1))

        wl_l2, wh_l2 = self.dwtl_3(xl2)
        wl_r2, wh_r2 = self.dwtr_3(xr2)
        xl4, xr4 = self.dwn4(torch.cat((xl3, wl_l2), dim=1), torch.cat((xr3, wl_r2), dim=1))

        wl_l3, wh_l3 = self.dwtl_4(xl3)
        wl_r3, wh_r3 = self.dwtr_4(xr3)

        for m in self.inner_stage:
            xl4, xr4 = m(xl4, wl_l3, xr4, wl_r3, wh_l3, wh_r3, input)

        xl_4, xr_4 = self.up1(torch.cat((xl4, x_f3, wh_l3), dim=1), torch.cat((xr4, x_f3, wh_r3), dim=1))
        xl_3, xr_3 = self.up2(torch.cat((xl_4, xl3, x_f2, wh_l2), dim=1), torch.cat((xr_4, xr3, x_f2, wh_r2), dim=1))
        xl_2, xr_2 = self.up3(torch.cat((xl_3, xl2, x_f1, wh_l1), dim=1), torch.cat((xr_3, xr2, x_f1, wh_r1), dim=1))
        xl_1, xr_1 = self.up4(torch.cat((xl_2, xl1, wh_l), dim=1), torch.cat((xr_2, xr1, wh_r), dim=1))

        l = self.tail_l(xl_1)
        r = self.tail_r(xr_1)
        out = (l + xl) * (r + xr)

        return F.sigmoid(out)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    name="CC38"
    with torch.cuda.device(-1):
        net = CC38(num_blocks=4)
        net = net.cuda()
        macs, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        # print(net.flops())
        print('{} - {:<30}  {:<8}'.format(name, 'Computational complexity: ', macs))
        print('{} - {:<30}  {:<8}'.format(name, 'Number of parameters: ', params))
        print('{} - {:<30}  {:<8}'.format(name, 'Number of parameters: ', params))


    # s = Shortcut(16, 4)
    # inp = torch.rand((1, 16, 8, 8))
    # out = s(inp)
    # print(out.shape)

    # att = AgentAttention(128, 36, 64)
    # x = torch.randn(1, 128, 64, 64)
    # ag = torch.randn(1, 36, 64, 64)
    # w_h = torch.randn(1, 64, 64, 64)
    # w_l = torch.randn(1, 64, 64, 64)
    # out = att(x, ag, w_h, w_l)
    # print(out.shape)


