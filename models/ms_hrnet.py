import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用原有的 HRNet 组件
from .hrnet import (
    HRNetBranch, FuseLayer
)

class SpectralSpatialAttentionFusion(nn.Module):
    """
    改进的 SSAF 模块
    
    关键改进:
    1. Softmax 光谱注意力 - 强制波段选择性
    2. 通道-空间解耦注意力
    3. 多尺度空间融合
    4. 动态门控机制
    """
    
    def __init__(self, num_bands=4, reduction=2, spatial_scales=[1, 2, 4]):
        super(SpectralSpatialAttentionFusion, self).__init__()
        self.num_bands = num_bands
        self.spatial_scales = spatial_scales
        
        # 1. 光谱注意力编码器
        self.spectral_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, num_bands // reduction, 1),
            nn.BatchNorm2d(num_bands // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_bands // reduction, num_bands, 1)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)
        
        # 2. 通道注意力 (SE Block)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, num_bands // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_bands // reduction, num_bands, 1),
            nn.Sigmoid()
        )
        
        # 3. 波段交互
        self.band_interaction = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, 3, padding=1, groups=num_bands, bias=False),
            nn.BatchNorm2d(num_bands),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_bands, num_bands * 2, 1, bias=False),
            nn.BatchNorm2d(num_bands * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_bands * 2, num_bands, 1, bias=False),
            nn.BatchNorm2d(num_bands)
        )
        
        # 4. 多尺度空间注意力
        self.spatial_branches = nn.ModuleList([
            self._make_spatial_branch(num_bands, scale) 
            for scale in spatial_scales
        ])
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(len(spatial_scales), 1, 1),
            nn.Sigmoid()
        )
        
        # 5. 动态门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands * 2, num_bands, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _make_spatial_branch(self, in_channels, scale):
        if scale == 1:
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 1, 1)
            )
        else:
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # 1. Softmax 光谱注意力
        spectral_logits = self.spectral_encoder(x)
        temp = self.temperature.abs().clamp(min=1.0, max=10.0)
        spectral_weights = F.softmax(spectral_logits / temp, dim=1)
        x_spectral = x * spectral_weights * C
        
        # 2. 通道注意力
        channel_weights = self.channel_attention(x_spectral)
        x_channel = x_spectral * channel_weights
        
        # 3. 波段交互
        x_interact = self.band_interaction(x_channel)
        x_fused = x_channel + x_interact
        
        # 4. 多尺度空间注意力
        spatial_maps = [branch(x_fused) for branch in self.spatial_branches]
        spatial_concat = torch.cat(spatial_maps, dim=1)
        spatial_weights = self.spatial_fusion(spatial_concat)
        x_spatial = x_fused * spatial_weights
        
        # 5. 动态门控
        gate_input = torch.cat([identity, x_spatial], dim=1)
        gate_weights = self.gate(gate_input)
        x_output = gate_weights * x_spatial + (1 - gate_weights) * identity
        
        # 返回注意力图用于可视化和分析
        attention_maps = {
            'spectral_weights': spectral_weights,
            'channel_weights': channel_weights,
            'spatial_weights': spatial_weights,
            'gate_weights': gate_weights,
            'temperature': temp.item()
        }
        
        return x_output, attention_maps


class MSHRNet(nn.Module):
    """
    MS-HRNet V2: 使用改进的 SSAF 模块
    
    主要改进:
    1. ImprovedSSAF 替代原 SSAF
    2. 返回更丰富的注意力图用于分析
    """
    
    def __init__(self, in_channels=4, num_classes=1, base_channels=48, 
                 use_minimal_ssaf=False):
        super(MSHRNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        
        # ============ 改进的 SSAF ============
        if use_minimal_ssaf:
            from .ssaf_improved import MinimalSSAF
            self.ssaf = MinimalSSAF(num_bands=in_channels)
        else:
            self.ssaf = SpectralSpatialAttentionFusion(num_bands=in_channels, reduction=2)
        
        # ============ HRNet Backbone (与原版相同) ============
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = HRNetBranch(4, 64, base_channels)
        
        # Transitions and Stages
        self.transition1 = nn.ModuleList([
            None,
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.stage2_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels * 2, base_channels * 2)
        ])
        self.fuse2 = FuseLayer(2, [base_channels, base_channels * 2])
        
        self.transition2 = nn.ModuleList([
            None, None,
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.stage3_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels * 2, base_channels * 2),
            HRNetBranch(4, base_channels * 4, base_channels * 4)
        ])
        self.fuse3 = FuseLayer(3, [base_channels, base_channels * 2, base_channels * 4])
        
        self.transition3 = nn.ModuleList([
            None, None, None,
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.stage4_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels * 2, base_channels * 2),
            HRNetBranch(4, base_channels * 4, base_channels * 4),
            HRNetBranch(4, base_channels * 8, base_channels * 8)
        ])
        self.fuse4 = FuseLayer(4, [base_channels, base_channels * 2, 
                                    base_channels * 4, base_channels * 8])
        
        # Feature aggregation
        total_channels = base_channels * 15
        self.aggregate = nn.Sequential(
            nn.Conv2d(total_channels, base_channels * 4, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # ============ 改进的 SSAF ============
        x, attention_maps = self.ssaf(x)
        
        # ============ HRNet Backbone ============
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        
        # Stage 2
        x_list = [x]
        x_list.append(self.transition1[1](x))
        x_list = [branch(x_list[i]) for i, branch in enumerate(self.stage2_branches)]
        x_list = self.fuse2(x_list)
        
        # Stage 3
        x_list_new = x_list.copy()
        x_list_new.append(self.transition2[2](x_list[-1]))
        x_list = [branch(x_list_new[i]) for i, branch in enumerate(self.stage3_branches)]
        x_list = self.fuse3(x_list)
        
        # Stage 4
        x_list_new = x_list.copy()
        x_list_new.append(self.transition3[3](x_list[-1]))
        x_list = [branch(x_list_new[i]) for i, branch in enumerate(self.stage4_branches)]
        x_list = self.fuse4(x_list)
        
        # Aggregation
        x0 = x_list[0]
        x1 = F.interpolate(x_list[1], size=x0.shape[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x_list[2], size=x0.shape[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x_list[3], size=x0.shape[2:], mode='bilinear', align_corners=True)
        
        feats = torch.cat([x0, x1, x2, x3], dim=1)
        feats = self.aggregate(feats)
        
        # Final prediction
        out = self.final_conv(feats)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        if self.training:
            return out, attention_maps
        else:
            return out