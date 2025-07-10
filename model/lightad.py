import sys

import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck, BasicBlock

from model import get_model
from model import MODEL
from model.lib_mamba.vmambanew import SS2D
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from pytorch_wavelets import DWTForward
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, LayerNorm2d
import pywt
import pywt.data
from timm.layers import DropPath
from functools import partial
from dropblock import DropBlock2D

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False,
                              dilation=dilation)

class DWConv33(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(DWConv33, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DWPatchExpand2D(nn.Module):
    def __init__(self, in_dim, out_dim, scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale = scale

        self.expand = nn.Linear(in_dim, (scale ** 2) * out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=self.scale, p2=self.scale, c=self.out_dim)
        x = self.norm(x)
        return x



class DWConv2d_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super(DWConv2d_BN_ReLU, self).__init__()
        self.conv1bx = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU(),
        )
        self.conv1ax = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU(),
        )
        self.convxx = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False,
                      groups=in_channels),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1ax(self.convxx(self.conv1bx(x)))
        return x


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # self.add_module('bn', torch.nn.BatchNorm2d(b))
        self.add_module('bn', torch.nn.InstanceNorm2d(b, affine=True))
        # torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        # torch.nn.init.constant_(self.bn.bias, 0)


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.1):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=16, w=9):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02  # 实部 + 虚部
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2)
        return x


class FC_Block(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d", h_w=16):
        super().__init__()
        self.dim = dim
        self.global_channels = int(global_ratio * dim)
        if local_ratio * dim == 0:
            self.local_channels = 0
        else:
            self.local_channels = dim - self.global_channels
        if self.local_channels != 0:
            self.local_op = DWConv2d_BN_ReLU(self.local_channels, self.local_channels, kernels)
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op_1 = GlobalFilter(self.global_channels, h=h_w, w=h_w//2+1)
            self.global_op_2 = GlobalFilter(self.global_channels, h=h_w, w=h_w//2+1)
        else:
            self.global_op = nn.Identity()

        self.post_norm_global_1 = nn.Sequential(
            conv1x1(self.global_channels, self.global_channels),
            nn.InstanceNorm2d(self.global_channels, affine=True),
            nn.SiLU()
        )

        self.post_norm_global_2 = nn.Sequential(
            conv1x1(self.global_channels, self.dim),
            nn.InstanceNorm2d(self.dim, affine=True),
            nn.SiLU()
        )
        self.post_norm_x2 = nn.Sequential(
            conv1x1(self.local_channels, self.dim),
            nn.InstanceNorm2d(self.dim, affine=True),
            nn.SiLU()
        )
        self.proj = nn.Sequential(
            conv1x1(self.dim * 2, self.dim * 2),
            nn.InstanceNorm2d(self.dim * 2, affine=True),
            nn.SiLU(),
            conv1x1(self.dim * 2, self.dim),
        )

    def forward(self, x):  # x (B,C,H,W)
        if self.global_channels == 0 and self.local_channels == 0:
            return x

        x1, x2 = torch.split(x, [self.global_channels, self.local_channels], dim=1)
        if self.global_channels != 0:
            x1 = self.post_norm_global_1(self.global_op_1(x1))
            x1 = self.post_norm_global_2(self.global_op_2(x1))
        if self.local_channels != 0:
            x2 = self.post_norm_x2(self.local_op(x2))

        if self.global_channels != 0 and self.local_channels != 0:
            x = self.proj(torch.cat([x1, x2], dim=1))
        elif self.global_channels == 0:
            x = self.proj(torch.cat([x2, x2], dim=1))
        elif self.local_channels == 0:
            x = self.proj(torch.cat([x1, x1], dim=1))
        return x


class FC(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, h_w=16):
        super().__init__()
        self.dim = dim
        self.h_w = h_w
        self.attn = FC_Block(dim, global_ratio=global_ratio, local_ratio=local_ratio,
                            kernels=kernels,  h_w=self.h_w)

    def forward(self, x):
        x = self.attn(x)
        return x


class FAFI(torch.nn.Module):
    def __init__(self,
                 ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5,  drop_path=0., has_skip=True,
                 in_dim=512, out_dim=256, h_w=16,
                 upsample=None):
        super().__init__()
        self.h_w = h_w
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))
        self.mixer = FC(ed, global_ratio=global_ratio, local_ratio=local_ratio,
                                                       kernels=kernels, h_w=self.h_w)



        if upsample is not None:
            self.upsample = upsample(in_dim=in_dim, out_dim=out_dim)
        else:
            self.upsample = None

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        if self.upsample is not None:
            x = rearrange(x, 'b c h w -> b h w c')
            x = self.upsample(x)
            x = rearrange(x, 'b h w c -> b c h w')

        shortcut = x
        x = self.mixer(self.ffn0(self.dw0(x)))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class LightNet(nn.Module):
    def __init__(self,
                 stages=['s', 's', 's'],
                 dims=[512, 256, 128, 64],
                 global_ratio=[0.6, 0.7, 0.8],
                 local_ratio=[0.3, 0.2, 0.2],
                 depth=[2, 2, 1],
                 kernels=[3, 5, 7],
                 adopt=True,
                 drop_path=0.1,):
        super().__init__()
        self.embed_dim = dims[1:]
        self.first_dim = dims[0]
        self.layers_up = nn.ModuleList()
        self.adopted = nn.ModuleList([nn.Identity()])
        if adopt:
            self.adopted = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels=self.first_dim, out_channels=self.first_dim * 2, kernel_size=1, bias=False),
                    nn.InstanceNorm2d(self.first_dim * 2),
                    nn.SiLU(),
                    nn.Conv2d(in_channels=self.first_dim * 2, out_channels=self.first_dim, kernel_size=1, bias=False),
                    nn.InstanceNorm2d(self.first_dim),
                )
            ])

        self.layers_up.append(self.adopted)
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depth))]
        for i, (ed, dpth, gr, lr) in enumerate(
                zip(self.embed_dim, depth, global_ratio, local_ratio)):
            dpr = dprs[sum(depth[:i]):sum(depth[:i + 1])]
            layer_mid_up = nn.ModuleList()
            h_w = {64: 64, 128: 32, 256: 16,}
            for d in range(dpth):
                layer = FAFI(ed, gr, lr, kernels[i],
                                         dpr[d],
                                         h_w=h_w[dims[i+1]],
                                         in_dim=dims[i] if i != 0 else dims[i],
                                         out_dim=dims[i+1],
                                         upsample=DWPatchExpand2D if (d == 0) else None,
                                         )
                layer_mid_up.append(layer)
            self.layers_up.append(layer_mid_up)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        out_features = []
        for i, layer_mid in enumerate(self.layers_up):
            for layer in layer_mid:
                x = layer(x)
            if i != 0:
                out_features.insert(0, x)
        return out_features

# ========== MFF & OCE ==========

class make_layers(nn.Module):
    def __init__(self, dims=[512, 256, 128, 64], stride=1, p=0.2):
        super(make_layers, self).__init__()
        self.dims = dims
        self.layer1 = nn.Sequential(
            DWConv33(self.dims[1], self.dims[1] * 2, stride),
            nn.InstanceNorm2d(self.dims[1] * 2, affine=True),
            nn.SiLU(),
            nn.Dropout2d(p=p),
            DWConv33(self.dims[1] * 2, self.dims[0], 1),
            nn.InstanceNorm2d(self.dims[0], affine=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


class MFF_OCE_DWConv(nn.Module):
    def __init__(self, dims=[512, 256, 128, 64], norm_layer=None, dropout=0.2):
        super(MFF_OCE_DWConv, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d  # 使用nn.InstanceNorm2d对visa数据集更好
        self._norm_layer = norm_layer
        self.dims = dims
        self.dropout = nn.Dropout2d(p=dropout)
        self.bn_layer = make_layers(dims, stride=2, p=dropout)

        self.conv1 = DWConv33(self.dims[3], self.dims[2], 2)
        self.bn1 = norm_layer(self.dims[2], affine=True)
        self.conv2 = DWConv33(self.dims[2], self.dims[1], 2)
        self.bn2 = norm_layer(self.dims[1], affine=True)
        self.conv21 = nn.Conv2d(self.dims[2], self.dims[2], 1)
        self.bn21 = norm_layer(self.dims[2], affine=True)
        self.conv31 = nn.Conv2d(self.dims[1], self.dims[1], 1)
        self.bn31 = norm_layer(self.dims[1], affine=True)
        self.convf = nn.Conv2d(self.dims[1], self.dims[1], 1)
        self.bnf = norm_layer(self.dims[1], affine=True)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        fpn0 = self.relu(self.bn1(self.conv1(x[0])))
        fpn1 = self.relu(self.bn21(self.conv21(x[1]))) + fpn0
        fpn1 = self.dropout(fpn1)
        sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(x[2])))
        sv_features = self.relu(self.bnf(self.convf(sv_features)))
        sv_features = self.dropout(sv_features)
        sv_features = self.bn_layer(sv_features)

        return sv_features.contiguous()


class Creat_model(nn.Module):
    def __init__(self):
        super(Creat_model, self).__init__()
        self.model = timm.create_model(
            'resnet18',
            # 'resnet34',
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3],
        )

    def forward(self, x):
        return self.model(x)


class LightAD(nn.Module):
    def __init__(self, model_t, model_s):
        super(LightAD, self).__init__()
        self.net_t = get_model(model_t)
        # self.net_t = Creat_model()
        self.mff_oce = MFF_OCE_DWConv(dims=model_s['dims'], dropout=model_s['dropout'])
        self.net_s = LightNet(dims=model_s['dims'], global_ratio=model_s['global_ratio'],
                             local_ratio=model_s['local_ratio'], depth=model_s['depth'],
                             kernels=model_s['kernels'], adopt=model_s['adopt'])

        self.frozen_layers = ['net_t']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        feats_s = self.net_s(self.mff_oce(feats_t))
        return feats_t, feats_s


@MODEL.register_module
def lightad(pretrained=False, **kwargs):
    model = LightAD(**kwargs)
    return model


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from util.util import get_timepc, get_net_params
    import timm

    bs = 8

    model_t = dict()
    model_s = dict(
        dims=[256, 256, 128, 64],
        global_ratio=[0.5, 0.5, 0.5],
        local_ratio=[0.5, 0.5, 0.5],
        depth=[2, 3, 2],
        kernels=[1, 3, 5],
        dropout=0.1,
        adopt=True,
    )
    vmunet = LightAD(model_t, model_s)
    x = torch.randn(bs, 3, 256, 256).cuda()

    net = vmunet.cuda()
    net.eval()
    y = net(x)
    Flops = FlopCountAnalysis(net, x)
    print(flop_count_table(Flops, max_depth=5))
    flops = Flops.total() / bs / 1e9
    params = parameter_count(net)[''] / 1e6
    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = net(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = net(x)
        t_e = get_timepc()
    print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params, bs * cnt / (t_e - t_s)))
