# models/ms_hrnet_ablation.py
"""
SSAF消融实验变体
通过开关控制各子模块的启用/禁用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hrnet import HRNetBranch, FuseLayer


class SSAFAblation(nn.Module):
    """
    可配置的SSAF消融模块
    通过布尔开关独立控制每个子组件
    """
    
    def __init__(self, num_bands=4, reduction=2, spatial_scales=[1, 2, 4],
                 use_spectral=True,
                 use_channel=True,
                 use_band_interaction=True,
                 use_multiscale_spatial=True,
                 use_dynamic_gate=True):
        super().__init__()
        self.num_bands = num_bands
        self.use_spectral = use_spectral
        self.use_channel = use_channel
        self.use_band_interaction = use_band_interaction
        self.use_multiscale_spatial = use_multiscale_spatial
        self.use_dynamic_gate = use_dynamic_gate
        
        # 1. 光谱注意力
        if use_spectral:
            self.spectral_encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_bands, max(1, num_bands // reduction), 1),
                nn.BatchNorm2d(max(1, num_bands // reduction)),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, num_bands // reduction), num_bands, 1)
            )
            self.temperature = nn.Parameter(torch.ones(1) * 10.0)
        
        # 2. 通道注意力 (SE Block)
        if use_channel:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_bands, max(1, num_bands // reduction), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, num_bands // reduction), num_bands, 1),
                nn.Sigmoid()
            )
        
        # 3. 波段交互
        if use_band_interaction:
            self.band_interaction = nn.Sequential(
                nn.Conv2d(num_bands, num_bands, 3, padding=1,
                         groups=num_bands, bias=False),
                nn.BatchNorm2d(num_bands),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_bands, num_bands * 2, 1, bias=False),
                nn.BatchNorm2d(num_bands * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_bands * 2, num_bands, 1, bias=False),
                nn.BatchNorm2d(num_bands)
            )
        
        # 4. 多尺度空间注意力
        if use_multiscale_spatial:
            self.spatial_branches = nn.ModuleList([
                self._make_spatial_branch(num_bands, scale)
                for scale in spatial_scales
            ])
            self.spatial_fusion = nn.Sequential(
                nn.Conv2d(len(spatial_scales), 1, 1),
                nn.Sigmoid()
            )
        else:
            # 退化为单尺度空间注意力（d=1）
            self.spatial_single = nn.Sequential(
                nn.Conv2d(num_bands, num_bands // 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_bands // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_bands // 4, 1, 1),
                nn.Sigmoid()
            )
        
        # 5. 动态门控
        if use_dynamic_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_bands * 2, num_bands, 1),
                nn.Sigmoid()
            )
        
        self._init_weights()
    
    def _make_spatial_branch(self, in_channels, scale):
        dilation = scale
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3,
                     padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        identity = x
        out = x
        
        # 1. 光谱注意力
        if self.use_spectral:
            logits = self.spectral_encoder(out)
            temp = self.temperature.abs().clamp(min=1.0, max=10.0)
            spectral_w = F.softmax(logits / temp, dim=1)
            out = out * spectral_w * self.num_bands
        
        # 2. 通道注意力
        if self.use_channel:
            channel_w = self.channel_attention(out)
            out = out * channel_w
        
        # 3. 波段交互
        if self.use_band_interaction:
            interact = self.band_interaction(out)
            out = out + interact  # 残差
        
        # 4. 空间注意力（多尺度或单尺度）
        if self.use_multiscale_spatial:
            spatial_maps = [branch(out) for branch in self.spatial_branches]
            spatial_w = self.spatial_fusion(torch.cat(spatial_maps, dim=1))
            out = out * spatial_w
        else:
            spatial_w = self.spatial_single(out)
            out = out * spatial_w
        
        # 5. 动态门控（否则用固定残差）
        if self.use_dynamic_gate:
            gate_input = torch.cat([identity, out], dim=1)
            gate_w = self.gate(gate_input)
            out = gate_w * out + (1 - gate_w) * identity
        else:
            out = 0.5 * out + 0.5 * identity  # 固定权重残差
        
        return out, {}


class MSHRNetAblation(nn.Module):
    """
    用于消融实验的MS-HRNet变体
    """
    
    VARIANTS = {
        # 完整模型
        'full': dict(use_spectral=True,  use_channel=True,
                     use_band_interaction=True, use_multiscale_spatial=True,
                     use_dynamic_gate=True),
        # 去掉各单个组件
        'wo_spectral':   dict(use_spectral=False, use_channel=True,
                              use_band_interaction=True, use_multiscale_spatial=True,
                              use_dynamic_gate=True),
        'wo_channel':    dict(use_spectral=True,  use_channel=False,
                              use_band_interaction=True, use_multiscale_spatial=True,
                              use_dynamic_gate=True),
        'wo_band_inter': dict(use_spectral=True,  use_channel=True,
                              use_band_interaction=False, use_multiscale_spatial=True,
                              use_dynamic_gate=True),
        'wo_ms_spatial': dict(use_spectral=True,  use_channel=True,
                              use_band_interaction=True, use_multiscale_spatial=False,
                              use_dynamic_gate=True),
        'wo_dyn_gate':   dict(use_spectral=True,  use_channel=True,
                              use_band_interaction=True, use_multiscale_spatial=True,
                              use_dynamic_gate=False),
        # 无SSAF（与原始HRNet等价）
        'no_ssaf':       None,  # 特殊处理
    }
    
    def __init__(self, variant='full', in_channels=4, num_classes=1,
                 base_channels=48):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.variant = variant
        
        # SSAF模块
        if variant != 'no_ssaf':
            cfg = self.VARIANTS[variant]
            self.ssaf = SSAFAblation(num_bands=in_channels, **cfg)
        else:
            self.ssaf = None
        
        # HRNet Backbone（与原始完全相同）
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        
        self.layer1 = HRNetBranch(4, 64, base_channels)
        
        self.transition1 = nn.ModuleList([
            None,
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels*2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True))
        ])
        self.stage2_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels*2, base_channels*2)
        ])
        self.fuse2 = FuseLayer(2, [base_channels, base_channels*2])
        
        self.transition2 = nn.ModuleList([
            None, None,
            nn.Sequential(
                nn.Conv2d(base_channels*2, base_channels*4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True))
        ])
        self.stage3_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels*2, base_channels*2),
            HRNetBranch(4, base_channels*4, base_channels*4)
        ])
        self.fuse3 = FuseLayer(3, [base_channels, base_channels*2, base_channels*4])
        
        self.transition3 = nn.ModuleList([
            None, None, None,
            nn.Sequential(
                nn.Conv2d(base_channels*4, base_channels*8, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True))
        ])
        self.stage4_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels*2, base_channels*2),
            HRNetBranch(4, base_channels*4, base_channels*4),
            HRNetBranch(4, base_channels*8, base_channels*8)
        ])
        self.fuse4 = FuseLayer(4, [base_channels, base_channels*2,
                                    base_channels*4, base_channels*8])
        
        total_ch = base_channels * 15
        self.aggregate = nn.Sequential(
            nn.Conv2d(total_ch, base_channels*4, 1, bias=False),
            nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # SSAF
        if self.ssaf is not None:
            x, _ = self.ssaf(x)
        
        # HRNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        
        x_list = [x, self.transition1[1](x)]
        x_list = [b(x_list[i]) for i, b in enumerate(self.stage2_branches)]
        x_list = self.fuse2(x_list)
        
        x_list_new = x_list + [self.transition2[2](x_list[-1])]
        x_list = [b(x_list_new[i]) for i, b in enumerate(self.stage3_branches)]
        x_list = self.fuse3(x_list)
        
        x_list_new = x_list + [self.transition3[3](x_list[-1])]
        x_list = [b(x_list_new[i]) for i, b in enumerate(self.stage4_branches)]
        x_list = self.fuse4(x_list)
        
        x0 = x_list[0]
        feats = torch.cat([
            x0,
            F.interpolate(x_list[1], x0.shape[2:], mode='bilinear', align_corners=True),
            F.interpolate(x_list[2], x0.shape[2:], mode='bilinear', align_corners=True),
            F.interpolate(x_list[3], x0.shape[2:], mode='bilinear', align_corners=True),
        ], dim=1)
        
        out = self.final_conv(self.aggregate(feats))
        return F.interpolate(out, input_size, mode='bilinear', align_corners=True)