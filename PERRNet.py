import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import sys

from einops import rearrange
from mmcv.cnn import build_norm_layer

sys.path.insert(0, '../../')
from mmcv.ops.carafe import CARAFEPack
from pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_li, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from FADC import AdaptiveDilatedConv
from DCNv3 import DCNv3_pytorch as DCNv3
import torch
import torch.nn as nn
import numpy as np
import typing as t
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from functools import partial

import math
import typing as t

import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule

class EnhancedImageFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(EnhancedImageFeatureExtractor, self).__init__()
        # 多尺度特征提取
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(in_channels, out_channels//4, kernel_size=7, padding=3)
        self.conv7 = nn.Conv2d(in_channels, out_channels//4, kernel_size=9, padding=4)
        
        # 边缘检测分支
        self.edge_detect = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1)
        )
        
        # 融合层
        self.fusion = nn.Conv2d(out_channels+out_channels//4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 多尺度特征
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        feat7 = self.conv7(x)
        multi_scale = torch.cat([feat1, feat3, feat5, feat7], dim=1)
        
        # 边缘特征
        edge_feat = self.edge_detect(x)
        
        # 融合
        fused = torch.cat([multi_scale, edge_feat], dim=1)
        out = self.relu(self.bn(self.fusion(fused)))
        return out
def get_freq_indices(method: str) -> t.Tuple:
    """Get the frequency indices according to the method."""
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                            6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                            5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                            3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                            4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                            3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                            3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(nn.Module):
    """the implementation of FCA"""
    def __init__(self, channel: int, dct_h: int, dct_w: int, reduction: int = 16, freq_sel_method: str = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        # Ensure that frequencies of different sizes have the same representation in the identical 7x7 frequency space.
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # multi-spectral information aggregate
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry. :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)
class EnhancedMultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(EnhancedMultiSpectralAttentionLayer, self).__init__()
        # 原有的多光谱注意力层
        self.msa = MultiSpectralAttentionLayer(channel, dct_h, dct_w, reduction, freq_sel_method)
        
        # 处理原始图像的分支
        self.img_conv = nn.Conv2d(3, channel//4, kernel_size=3, padding=1)
        self.img_pool = nn.AdaptiveAvgPool2d((dct_h, dct_w))
        
        # 融合原始图像特征和通道注意力
        self.fusion = nn.Sequential(
            nn.Conv2d(channel + channel//4, channel, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数
        
    def forward(self, x, orig_img=None):
        # 获取原始的通道注意力
        channel_att = self.msa(x)
        if orig_img is not None:
            # 处理原始图像
            img_feat = self.img_conv(orig_img)
            img_feat = self.img_pool(img_feat)
            
            # 调整大小以匹配特征图
            if img_feat.size(2) != x.size(2) or img_feat.size(3) != x.size(3):
                img_feat = F.interpolate(img_feat, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            # 融合原始图像特征和通道注意力
            combined = torch.cat([channel_att.expand_as(x), img_feat], dim=1)
            enhanced_att = self.fusion(combined)
            
            # 使用可学习参数控制原始图像的影响
            final_att = channel_att + self.gamma * enhanced_att
            return final_att
        
        return channel_att
class AdaptiveFrequencySelectionLayer(nn.Module):
    def __init__(self, channel, dct_h, dct_w, mapper_x, mapper_y):
        super(AdaptiveFrequencySelectionLayer, self).__init__()
        self.channel = channel
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.mapper_x = mapper_x
        self.mapper_y = mapper_y
        self.num_freq = len(mapper_x)

        # 扩张卷积操作，捕捉不同尺度的频率信息
        self.dilated_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=2, dilation=2)
        
        # 自适应学习频率的权重
        self.attention = nn.Sequential(
            nn.Conv2d(channel, channel // 16, kernel_size=1),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, kernel_size=1),  # 恢复维度
            nn.Sigmoid()  # 权重是 [0, 1] 之间的数
        )

    def forward(self, x):
        # 使用扩张卷积捕捉不同尺度的频率信息
        x = self.dilated_conv(x)

        # 计算频率注意力权重
        freq_attention = self.attention(x)  # 获取自适应的注意力权重

        # 乘以频率注意力权重
        x = x * freq_attention
        return x

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters with adaptive frequency selection
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # 初始化DCT过滤器
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # 自适应频率选择
        self.adaptive_freq_selector = AdaptiveFrequencySelectionLayer(channel, height, width, mapper_x, mapper_y)

    def forward(self, x):
        assert len(x.shape) == 4, 'x must be 4D, but got ' + str(len(x.shape))

        # 应用DCT滤波器
        x = x * self.weight
        
        # 应用自适应频率选择
        x = self.adaptive_freq_selector(x)

        # 计算最终的输出
        result = torch.sum(x, dim=[2, 3])  # 聚合空间维度
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter


class DoubleUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleUpsampling, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.up(x)
        return x

class DoubleDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleDownsampling, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.down(x)
        return x

class FeatureDecoder(nn.Module):
    def __init__(self, in_channels):
        super(FeatureDecoder, self).__init__()
        eucb_ks = 3 # kernel size for eucb
        
        self.up = EUCB(in_channels, in_channels, kernel_size=eucb_ks, stride=eucb_ks//2)
        self.down = DoubleDownsampling(in_channels, in_channels)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels*2)
    
    def forward(self, x5, x4, x3):
        # UP(X5) ⊗ X4 ⊗ Down(X3)
        Xmid = self.up(x5) * x4 * self.down(x3)
        # Xd5 = X5 ⊗ Down(Xmid)
        Xd5 = x5 * self.down(Xmid)
        # Xd3 = X5 ⊗ UP(Xmid)
        Xd3 = x3 * self.up(Xmid)
        # Xd4 = Xmid ⊗ Down(Xd5) ⊗ UP(Xd3)
        Xd4 = Xmid * self.up(Xd5) * self.down(Xd3)
        
        # Concatenated operation with a 3x3 convolutional layer and Batch Normalization
        XD_out = torch.cat((self.conv3x3(Xd4),self.up(Xd5)), dim=1)
        XD_out = self.bn(XD_out)
        return XD_out
    
    
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
        

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    elif act == 'sigmoid':
        return nn.Sigmoid()  # 添加对 sigmoid 激活函数的支持
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
	        nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

import torch.nn.functional as F

class SpatialCoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, bias=False, kernel_size=3, act=nn.ReLU(), dropout_rate=0.):
        super(SpatialCoordAttention, self).__init__()

        assert kernel_size in (3,7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # Multi-scale convolution layers for global attention
        self.global_att = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=bias),  # 3x3 kernel
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=bias)   # 7x7 kernel
        ])

        # Local attention layers (1x1 convolutions)
        self.local_att1 = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=bias)
        self.local_att2 = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=bias)

        # Apply Dropout after ReLU activations to prevent overfitting
        self.local_h = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels*2 // reduction, kernel_size=kernel_size, padding=padding, bias=bias),  
            act,
            nn.Dropout(dropout_rate),  # Adding Dropout here
            nn.Conv1d(in_channels*2 // reduction, in_channels, kernel_size=kernel_size, padding=padding, bias=bias),  
            nn.Sigmoid()
        )
        self.local_w = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels*2 // reduction, kernel_size=kernel_size, padding=padding, bias=bias),  
            act,
            nn.Dropout(dropout_rate),  # Adding Dropout here
            nn.Conv1d(in_channels*2 // reduction, in_channels, kernel_size=kernel_size, padding=padding, bias=bias),  
            nn.Sigmoid()
        )

        self.beta = nn.Parameter(torch.ones(1))
        # 添加处理原始图像的卷积层
        self.orig_conv = nn.Conv2d(3, in_channels, kernel_size=1, bias=bias)  # 假设原图是3通道
        self.fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的参数控制原图权重
    def forward(self, x, orig_img=None):
        # Global attention (multi-scale)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        global_out = torch.cat([avg_out, max_out], dim=1)

        # Apply multi-scale convolution
        global_out_scale1 = self.global_att[0](global_out)  # 3x3 kernel
        global_out_scale2 = self.global_att[1](global_out)  # 7x7 kernel

        # Combine outputs from both scales (concatenation or addition)
        global_out = global_out_scale1 + global_out_scale2  # You could also use torch.cat if you prefer concatenation
        global_out = torch.sigmoid(global_out)

        # Local attention (height-wise)
        avg_h = torch.mean(x, dim=3, keepdim=True)
        max_h, _ = torch.max(x, dim=3, keepdim=True)
        local_h = torch.cat([avg_h, max_h], dim=3)
        local_h = local_h.permute(0, 3, 1, 2)
        local_h = self.local_att1(local_h)
        local_h = local_h.permute(0, 2, 3, 1)

        # Local attention (width-wise)
        avg_w = torch.mean(x, dim=2, keepdim=True)
        max_w, _ = torch.max(x, dim=2, keepdim=True)
        local_w = torch.cat([avg_w, max_w], dim=2)
        local_w = local_w.permute(0, 2, 1, 3)
        local_w = self.local_att2(local_w)
        local_w = local_w.permute(0, 2, 1, 3)

        # Combine local attention
        local_out = local_h.expand_as(x) * local_w.expand_as(x)

        # Combine global and local attentions
        return local_out * self.beta + global_out

class EnhancedSpatialCoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, bias=False, kernel_size=3, act=nn.ReLU(), dropout_rate=0.):
        super(EnhancedSpatialCoordAttention, self).__init__()
        # 保留原有的空间注意力机制
        self.spatial_att = SpatialCoordAttention(in_channels, reduction, bias, kernel_size, act, dropout_rate)
        
        # 添加处理原始图像的边缘检测分支
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 添加处理原始图像的纹理分析分支
        self.texture_analyzer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 融合边缘和纹理信息
        self.fusion = nn.Conv2d(2, 1, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1))  # 可学习的权重参数
        self.beta = nn.Parameter(torch.ones(1))   # 可学习的权重参数
        
    def forward(self, x, orig_img=None):
        # 获取原始的空间注意力
        spatial_att = self.spatial_att(x, orig_img)
        
        if orig_img is not None:
            # 提取边缘信息
            edge_map = self.edge_detector(orig_img)
            
            # 提取纹理信息
            texture_map = self.texture_analyzer(orig_img)
            
            # 调整大小以匹配特征图
            if edge_map.size(2) != x.size(2) or edge_map.size(3) != x.size(3):
                edge_map = F.interpolate(edge_map, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
                texture_map = F.interpolate(texture_map, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            
            # 融合边缘和纹理信息
            img_info = self.fusion(torch.cat([edge_map, texture_map], dim=1))
            
            # 结合原始空间注意力和图像信息
            enhanced_att = spatial_att * (1 + self.alpha * img_info)
            return enhanced_att
        
        return spatial_att

class CBAMLayer(nn.Module):
    def __init__(self, channel, h, w, bias=False, reduction=16, spatial_kernel=7, act=nn.ReLU()):
        super(CBAMLayer, self).__init__()
        
        # 原始图像特征提取器
        self.img_feature_extractor = EnhancedImageFeatureExtractor(in_channels=3, out_channels=channel)
        
        # 增强的通道注意力
        self.channel_att = EnhancedMultiSpectralAttentionLayer(channel, h, w)
        
        # 增强的空间注意力
        self.spatial_att = EnhancedSpatialCoordAttention(channel, reduction, bias, spatial_kernel, act)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channel),
            act
        )
        
        # 可学习的权重参数
        self.lambda_channel = nn.Parameter(torch.ones(1))
        self.lambda_spatial = nn.Parameter(torch.ones(1))
        self.lambda_orig = nn.Parameter(torch.zeros(1))  # 原始图像的权重，初始化为0
        
    def forward(self, x, orig_img):
        # 提取原始图像特征
        if orig_img is not None:
            # 调整原始图像大小以匹配特征图
            if orig_img.size(2) != x.size(2) or orig_img.size(3) != x.size(3):
                orig_img = F.interpolate(orig_img, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            
            img_feat = self.img_feature_extractor(orig_img)
            
            # 确保特征维度匹配
            if img_feat.size(1) != x.size(1):
                img_feat = F.conv2d(img_feat, torch.ones(x.size(1), img_feat.size(1), 1, 1).to(x.device) / img_feat.size(1))
        else:
            img_feat = torch.zeros_like(x)
        
        # 应用增强的通道注意力
        channel_out = self.channel_att(x, orig_img)
        
        # 应用增强的空间注意力
        spatial_out = self.spatial_att(channel_out, orig_img)
        
        # 融合特征
        combined_feat = torch.cat([x * spatial_out, img_feat], dim=1)
        fused_feat = self.fusion(combined_feat)
        
        # 使用可学习的权重参数控制各部分的贡献
        output = self.lambda_channel * channel_out + self.lambda_spatial * spatial_out * x + self.lambda_orig * fused_feat
        
        return output

class IFM(nn.Module):
    def __init__(self, groups=32, channels=128, c_scale=2):
        super(IFM, self).__init__()
        self.up_c = nn.Conv2d(channels, channels * c_scale, kernel_size=1, stride=1, padding=0)
        self.groups = groups
        self.out_max, self.out_mean = [], []
        self.conv = nn.Conv2d(groups * 2, channels, kernel_size=3, stride=1, padding=1)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        ori = x
        x = self.up_c(x)
        self.out_max, self.out_mean = [], []
        x = self.channel_shuffle(x, self.groups)
        x_groups = x.chunk(self.groups, 1)
        for x_i in x_groups:
            self.out_max.append(torch.max(x_i, dim=1)[0].unsqueeze(1))
            self.out_mean.append(torch.mean(x_i, dim=1).unsqueeze(1))
        out_max = torch.cat(self.out_max, dim=1)
        out_mean = torch.cat(self.out_mean, dim=1)
        out = torch.cat((out_max, out_mean), dim=1)
        x = self.conv(out) + ori
        return x


class SARNet(nn.Module):
    def __init__(self, fun_str1='pvt_v2_b3'):
        super().__init__()
        self.backbone, embedding_dims = eval(fun_str1)()

        self.rsm3 = RSM(embedding_dims[3] // 4, 256, focus_background=True,
                        opr_kernel_size=7, iterations=1)
        self.rsm2 = RSM(embedding_dims[1], embedding_dims[3] // 4, focus_background=True,
                        opr_kernel_size=7, iterations=1)
        self.rsm1 = RSM(embedding_dims[1], embedding_dims[1], focus_background=True,
                        opr_kernel_size=7,
                        iterations=1)
        self.rsm0 = RSM(embedding_dims[1], embedding_dims[1], focus_background=False,
                        opr_kernel_size=7, iterations=1)
 
        self.mfm0 = MFM(cur_in_channels=embedding_dims[0], low_in_channels=embedding_dims[1],
                        out_channels=embedding_dims[1], cur_scale=1, low_scale=2)
        self.mfm1 = MFM(cur_in_channels=embedding_dims[1], low_in_channels=embedding_dims[2],
                        out_channels=embedding_dims[1], cur_scale=1, low_scale=2)
        self.mfm2 = MFM(cur_in_channels=embedding_dims[2], low_in_channels=embedding_dims[3],
                        out_channels=embedding_dims[1], cur_scale=1, low_scale=2)

        self.cbr1 = CBR(in_channels=384, out_channels=256,
                        kernel_size=3, stride=1,
                        dilation=1, padding=1)
        self.cbr2 = CBR(in_channels=384, out_channels=256,
                        kernel_size=3, stride=1,
                        dilation=1, padding=1)
        self.predict_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1, stride=1))
        
        self.block = nn.Sequential(
            ConvBNR(256, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        self.mfm3 = MFM(cur_in_channels=128, low_in_channels=256,
                        out_channels=embedding_dims[1], cur_scale=2,
                        low_scale=4)  # 16

        reduction=4
        bias=False
        act=nn.ReLU(inplace=True)
        self.c0 = CBAMLayer(embedding_dims[0], 96, 96)
        self.c1 = CBAMLayer(embedding_dims[1], 48, 48)
        self.c2 = CBAMLayer(embedding_dims[2], 24, 24)
        self.c3 = CBAMLayer(embedding_dims[3], 12, 12)
        self.c4 = CBAMLayer(embedding_dims[1]*2, 24, 24)
        self.c01 = CBAMLayer(embedding_dims[0], 96, 96)
        self.c11 = CBAMLayer(embedding_dims[1], 48, 48)
        self.c21 = CBAMLayer(embedding_dims[2], 24, 24)
        self.c31 = CBAMLayer(embedding_dims[3], 12, 12)
        self.c41 = CBAMLayer(embedding_dims[1]*2, 24, 24)

        eucb_ks = 3 # kernel size for eucb
        in_channels=128
        self.up = EUCB(in_channels, in_channels, kernel_size=eucb_ks, stride=eucb_ks//2)
        self.down = DoubleDownsampling(in_channels, in_channels)
    def forward(self, x):
        # byxhz
        layer = self.backbone(x)
        x0 = F.interpolate(x, size=layer[0].size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(x, size=layer[1].size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x, size=layer[2].size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x, size=layer[3].size()[2:], mode='bilinear', align_corners=True)

        f0 = self.c01(self.c0(layer[0],x0),x0)
        f1 = self.c11(self.c1(layer[1],x1),x1)
        f2 = self.c21(self.c2(layer[2],x2),x2)
        f3 = self.c31(self.c3(layer[3],x3),x3)
        
        u2 = self.mfm0(f0, f1)
        u3 = self.mfm1(f1, f2)
        u4 = self.mfm2(f2, f3)
        u5 = torch.cat((self.down(u2),u3,self.up(u4)), dim=1)
        """
        u5 = self.fdc(u4,u3,u2)
        u5 = self.at(u5)
        u5 = self.cbr(u5)
        
        edge4 = self.block(u5)
        """
        u51 = self.cbr1(u5)
        u52 = self.cbr2(u5)
        predict4 = self.predict_conv(u51)
        edge4 = self.block(u52)
        u1 = self.mfm3(u2, u51)
        """
        predict4 = F.interpolate(predict4, scale_factor=0.25, mode='bilinear', align_corners=False)
        u5 = F.interpolate(u5, scale_factor=0.25, mode='bilinear', align_corners=False)
        #edge4 = F.interpolate(edge4, scale_factor=0.25, mode='bilinear', align_corners=False)
        fgc3, predict3 = self.fgc3(u4, u5, predict4)
        #fgc3 = self.fm0(fgc3)
        fgc2, predict2 = self.fgc2(u3, fgc3, predict3)
        #fgc2 = self.fm0(fgc2)
        fgc1, predict1 = self.fgc1(u2, fgc2, predict2)
        #fgc1 = self.fm0(fgc1)
        fgc0, predict0 = self.fgc0(u1, fgc1, predict1)
        
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        """
        # focus
        predict4 = F.interpolate(predict4, scale_factor=0.25, mode='bilinear', align_corners=False)
        u51 = F.interpolate(u51, scale_factor=0.25, mode='bilinear', align_corners=False)
        edge4 = F.interpolate(edge4, scale_factor=0.25, mode='bilinear', align_corners=False)
        fgc3, predict3, edge3 = self.rsm3(u4, u51, predict4, edge4,4)
        #fgc3 = self.fm0(fgc3)
        fgc2, predict2, edge2 = self.rsm2(u3, fgc3, predict3, edge3,3)
        #fgc2 = self.fm0(fgc2)
        fgc1, predict1, edge1 = self.rsm1(u2, fgc2, predict2, edge2,2)
        #fgc1 = self.fm0(fgc1)
        fgc0, predict0, edge0 = self.rsm0(u1, fgc1, predict1, edge1,1)

        # rescale
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge4 = F.interpolate(edge4, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge1 = F.interpolate(edge1, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge0 = F.interpolate(edge0, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        return predict4, predict3, predict2, predict1, predict0, edge4, edge3, edge2, edge1, edge0
    

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride,
                      padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        return x


class MFM(nn.Module):
    def __init__(self, cur_in_channels=64, low_in_channels=32, out_channels=16, cur_scale=2, low_scale=1):
        super(MFM, self).__init__()
        self.cur_in_channels = cur_in_channels
        self.cur_conv = nn.Sequential(
            nn.Conv2d(in_channels=cur_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

        self.cur_scale = cur_scale
        self.low_scale = low_scale

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels= int(out_channels/(low_scale*low_scale)) + int(out_channels/(cur_scale*cur_scale)), out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
        
        # 使用 PixelShuffle 来进行上采样
        self.cur_pixelshuffle = nn.PixelShuffle(upscale_factor=cur_scale)
        self.low_pixelshuffle = nn.PixelShuffle(upscale_factor=low_scale)

    def forward(self, x_cur, x_low):
        # 当前通道的卷积
        x_cur = self.cur_conv(x_cur)
        #print(x_cur.shape)
        # 当前通道的 PixelShuffle 上采样
        x_cur = self.cur_pixelshuffle(x_cur)
        #print(x_cur.shape)
        # 低频通道的卷积
        x_low = self.low_conv(x_low)
        #print(x_low.shape)
        # 低频通道的 PixelShuffle 上采样
        x_low = self.low_pixelshuffle(x_low)
        #print(x_low.shape)
        # 拼接两者
        x = torch.cat((x_cur, x_low), dim=1)
        #print(x.shape)
        # 输出卷积
        x = self.out_conv(x)
        
        return x


import numpy as np
import cv2

from thop import profile
def get_open_map(input, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations),
                        input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()


class Basic_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class EM(nn.Module):
    def __init__(self, groups=32, channels=128, c_scale=2):
        super(EM, self).__init__()
        self.up_c = nn.Conv2d(channels, channels * c_scale, kernel_size=1, stride=1, padding=0)
        self.groups = groups
        self.out_max, self.out_mean = [], []
        self.conv = nn.Conv2d(groups * 2, channels, kernel_size=3, stride=1, padding=1)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        ori = x
        x = self.up_c(x)
        self.out_max, self.out_mean = [], []
        x = self.channel_shuffle(x, self.groups)
        x_groups = x.chunk(self.groups, 1)
        for x_i in x_groups:
            self.out_max.append(torch.max(x_i, dim=1)[0].unsqueeze(1))
            self.out_mean.append(torch.mean(x_i, dim=1).unsqueeze(1))
        out_max = torch.cat(self.out_max, dim=1)
        out_mean = torch.cat(self.out_mean, dim=1)
        out = torch.cat((out_max, out_mean), dim=1)
        x = self.conv(out) + ori
        return x
    
# 深度可分离卷积类
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pointwise(x)
        return x
    
def edge_prediction(map):
    laplace = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype=np.float32)
    laplace = laplace[np.newaxis, np.newaxis, ...]
    laplace = torch.Tensor(laplace).cuda()
    edge = F.conv2d(map, laplace, padding=1)
    edge = F.relu(torch.tanh(edge))
    return edge

class RSM(nn.Module):
    def __init__(self, channel1, channel2, focus_background=True, opr_kernel_size=3, iterations=1):
        super(RSM, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.focus_background = focus_background
        
        self.up = nn.Sequential(
            nn.Conv2d(channel2, channel1, 7, 1, 3),  # 原来的卷积
            nn.BatchNorm2d(channel1),
            nn.ReLU(),
            EUCB(channel1, channel1, kernel_size=3, activation='relu'))

        self.input_map1 = EUCB(1, 1, kernel_size=3, activation='sigmoid')
        self.input_map2 = EUCB(1, 1, kernel_size=3, activation='sigmoid')
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.beta = nn.Parameter(torch.ones(1))
        self.beta1 = nn.Parameter(torch.ones(1))
        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1, stride=1)

        # 替换为深度可分离卷积
        self.depthwise_conv1 = DepthwiseSeparableConv(self.channel1 , self.channel1, kernel_size=3, padding=1)  # 小尺度卷积
        self.depthwise_conv2 = DepthwiseSeparableConv(self.channel1, self.channel1, kernel_size=5, padding=2)  # 中尺度卷积
        self.depthwise_conv3 = DepthwiseSeparableConv(self.channel1, self.channel1, kernel_size=7, padding=3)  # 大尺度卷积
        self.conv_out = nn.Conv2d(self.channel1*4, self.channel1, kernel_size=1, stride=1, padding=0)

        self.opr_kernel_size = opr_kernel_size
        self.iterations = iterations
        self.block = nn.Sequential(
            ConvBNR(128, 256, 3),
            ConvBNR(256, 128, 3),
            nn.Conv2d(128, 1, 1)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(channel1, channel1, kernel_size=opr_kernel_size, stride=1, padding=opr_kernel_size//2, bias=True),
            nn.BatchNorm2d(channel1)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(channel1, channel1, kernel_size=opr_kernel_size, stride=1, padding=opr_kernel_size//2, bias=True),
            nn.BatchNorm2d(channel1)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(channel1, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        activation='relu'
        self.activation = act_layer(activation, inplace=True)
        self.em0 = EM(groups=32, channels=channel1, c_scale=2)
        self.em1 = EM(groups=32, channels=channel1, c_scale=2)
        self.conv_layer = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1) 
        #self.edge = LearnableEdge(1)
        
    def get_dilation_erosion_kernels(self, stage):
        # 根据不同的阶段返回膨胀和腐蚀核的大小
        if stage == 1 or stage == 2:
            dilation_kernel = torch.ones(1, 1, 7, 7)  # 较大的膨胀核
            erosion_kernel = None  # 只膨胀
        elif stage == 3:
            dilation_kernel = torch.ones(1, 1, 5, 5)  # 中等大小的膨胀核
            erosion_kernel = None  # 只膨胀
        elif stage == 4:
            dilation_kernel = None  # 不膨胀
            erosion_kernel = torch.ones(1, 1, 5, 5)  # 较大的腐蚀核
        else:
            dilation_kernel = torch.ones(1, 1, 3, 3)  # 默认膨胀核大小
            erosion_kernel = None
        
        return dilation_kernel, erosion_kernel

    def forward(self, cur_x, dep_x, in_map, in_edge, stage):
        dep_x = self.up(dep_x)
        input_map = self.input_map1(in_map)
        in_edge = self.input_map2(in_edge)
        cur_x = self.em0(cur_x)
        # 获取膨胀核和腐蚀核
        dilation_kernel, erosion_kernel = self.get_dilation_erosion_kernels(stage)

        dilation_kernel = dilation_kernel.to(input_map.device) if dilation_kernel is not None else None
        erosion_kernel = erosion_kernel.to(in_edge.device) if erosion_kernel is not None else None

        # 进行膨胀和腐蚀操作
        # 膨胀操作
        if dilation_kernel is not None:
            dilated_in_edge = F.conv2d(in_edge, dilation_kernel, padding=dilation_kernel.size(2) // 2)  # 确保填充一致
            dilated_in_edge = dilated_in_edge.clamp(0, 1)  # 限制为 [0, 1] 范围内
            combined_map = dilated_in_edge + input_map

        # 腐蚀操作
        if erosion_kernel is not None:
            eroded_input_map = F.conv2d(input_map, erosion_kernel, padding= erosion_kernel.size(2) // 2)  # 确保填充一致
            eroded_input_map = eroded_input_map.clamp(0, 1)  # 限制为 [0, 1] 范围内
            combined_map = in_edge + eroded_input_map

        #combined_map = dilated_in_edge + eroded_input_map  # 膨胀与腐蚀的合成
        b_feature = cur_x * combined_map  # 最终特征
        # 继续处理 b_feature
        fn = self.beta * self.conv2(b_feature)
        g1 = self.W_g(dep_x)
        x1 = self.W_x(fn)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        refine = fn * psi + dep_x
        refine1 = self.depthwise_conv1(refine)
        refine2 = self.depthwise_conv2(refine)
        refine3 = self.depthwise_conv3(refine)
        fused_refine = torch.cat((refine1, refine2, refine3), dim=1)
        fused_refine = self.conv_layer(fused_refine)
        fused_refine = self.em1(fused_refine)
        output_map = self.output_map(fused_refine)
        edge_map = edge_prediction(output_map)

        return fused_refine, output_map, edge_map