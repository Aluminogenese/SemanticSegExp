import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedSSAF(nn.Module):
    """
    改进的 SSAF 模块
    
    关键改进:
    1. Softmax 光谱注意力 - 强制波段选择性
    2. 通道-空间解耦注意力
    3. 多尺度空间融合
    4. 动态门控机制
    """
    
    def __init__(self, num_bands=4, reduction=2, spatial_scales=[1, 2, 4]):
        super(ImprovedSSAF, self).__init__()
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
        self.temperature = nn.Parameter(torch.ones(1) * 5.0)
        
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


class MinimalSSAF(nn.Module):
    """轻量级 SSAF - 参数量更少"""
    
    def __init__(self, num_bands=4):
        super(MinimalSSAF, self).__init__()
        self.num_bands = num_bands
        
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, num_bands, 1),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 3.0)
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_bands, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        logits = self.spectral_attention(x)
        temp = self.temperature.abs().clamp(min=1.0)
        spectral_w = F.softmax(logits / temp, dim=1)
        x_spectral = x * spectral_w * self.num_bands
        
        spatial_w = self.spatial_attention(x_spectral)
        x_spatial = x_spectral * spatial_w
        
        alpha = torch.sigmoid(self.alpha)
        output = alpha * x_spatial + (1 - alpha) * x
        
        return output, {'spectral_weights': spectral_w, 'temperature': temp.item()}