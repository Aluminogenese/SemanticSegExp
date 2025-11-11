"""
Advanced Spectral-Spatial Attention Fusion (SSAF)

参考论文：
1. SKNet (CVPR 2019): Selective Kernel Networks - 多分支融合
2. ECANet (CVPR 2020): Efficient Channel Attention - 自适应卷积核
3. FcaNet (ICCV 2021): Frequency Channel Attention - 频域分析
4. Remote Sensing 领域: Spectral-Spatial Feature Fusion

核心改进：
1. 显式建模光谱指数（NDVI, NDBI 等）
2. 多尺度光谱注意力（全局+局部）
3. 跨波段交互建模
4. 物理先验约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralIndexModule(nn.Module):
    """
    显式计算常用光谱指数
    
    参考遥感领域的经典指数：
    - NDVI = (NIR - Red) / (NIR + Red)  # 植被指数
    - NDBI = (SWIR - NIR) / (SWIR + NIR) # 建筑指数（简化为 Red-NIR）
    - NDWI = (Green - NIR) / (Green + NIR) # 水体指数
    
    这些指数在建筑物分割中有明确的物理意义
    """
    
    def __init__(self, num_bands=4):
        super(SpectralIndexModule, self).__init__()
        self.num_bands = num_bands
        
        if num_bands == 4:  # RGB + NIR
            # 可学习的指数权重（初始化为标准公式）
            self.ndvi_weight = nn.Parameter(torch.tensor([0.0, 0.0, -1.0, 1.0]))  # (NIR-R)/(NIR+R)
            self.ndbi_weight = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, -1.0]))  # (R-NIR)/(R+NIR)
            self.ndwi_weight = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, -1.0]))  # (G-NIR)/(G+NIR)
            
            # 融合卷积
            self.index_fusion = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),  # 3个指数
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, num_bands, 1),
                nn.Sigmoid()
            )
    
    def compute_index(self, x, weight):
        """计算归一化差值指数"""
        numerator = (x * weight.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        denominator = x.abs().sum(dim=1, keepdim=True) + 1e-8
        return numerator / denominator
    
    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W]
        Returns:
            indices: [B, 4, H, W] 基于光谱指数的权重
        """
        if self.num_bands != 4:
            return torch.ones_like(x)
        
        # 计算三个光谱指数
        ndvi = self.compute_index(x, self.ndvi_weight)  # [B, 1, H, W]
        ndbi = self.compute_index(x, self.ndbi_weight)
        ndwi = self.compute_index(x, self.ndwi_weight)
        
        # 拼接并融合
        indices = torch.cat([ndvi, ndbi, ndwi], dim=1)  # [B, 3, H, W]
        weights = self.index_fusion(indices)  # [B, 4, H, W]
        
        return weights


class MultiScaleSpectralAttention(nn.Module):
    """
    多尺度光谱注意力
    
    参考 SKNet 的思想：
    1. 多个分支处理不同尺度
    2. 通过注意力融合
    3. 自适应选择
    """
    
    def __init__(self, num_bands=4, reduction=2):
        super(MultiScaleSpectralAttention, self).__init__()
        self.num_bands = num_bands
        
        # 三个尺度的全局池化
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局
        self.pool1 = nn.AdaptiveAvgPool2d(4)  # 4x4 区域
        self.pool2 = nn.AdaptiveAvgPool2d(8)  # 8x8 区域
        
        # 三个分支的处理
        self.branch_global = nn.Sequential(
            nn.Conv2d(num_bands, num_bands // reduction, 1),
            nn.LayerNorm([num_bands // reduction, 1, 1]),
            nn.ReLU(inplace=True),
        )
        
        self.branch_local1 = nn.Sequential(
            nn.Conv2d(num_bands, num_bands // reduction, 1),
            nn.LayerNorm([num_bands // reduction, 4, 4]),
            nn.ReLU(inplace=True),
        )
        
        self.branch_local2 = nn.Sequential(
            nn.Conv2d(num_bands, num_bands // reduction, 1),
            nn.LayerNorm([num_bands // reduction, 8, 8]),
            nn.ReLU(inplace=True),
        )
        
        # 融合三个尺度
        self.fusion = nn.Sequential(
            nn.Conv2d(num_bands // reduction * 3, num_bands, 1),
            nn.Softmax(dim=1)  # 在通道维度做 Softmax
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            weights: [B, C, 1, 1]
        """
        # 三个尺度的特征
        feat_global = self.branch_global(self.gap(x))  # [B, C//r, 1, 1]
        feat_local1 = self.branch_local1(self.pool1(x))  # [B, C//r, 4, 4]
        feat_local2 = self.branch_local2(self.pool2(x))  # [B, C//r, 8, 8]
        
        # 上采样到相同尺寸
        feat_local1 = F.adaptive_avg_pool2d(feat_local1, 1)
        feat_local2 = F.adaptive_avg_pool2d(feat_local2, 1)
        
        # 拼接融合
        feat_cat = torch.cat([feat_global, feat_local1, feat_local2], dim=1)
        weights = self.fusion(feat_cat)  # [B, C, 1, 1], Softmax归一化
        
        return weights


class CrossBandInteraction(nn.Module):
    """
    跨波段交互建模（通道间注意力，避免 HW×HW 的显存爆炸）

    将注意力从“空间位置间”改为“波段通道间”计算：
    - 先将输入展平为 [B, C, HW]，用通道间相关性构建注意力矩阵 [B, C, C]
    - 再用该矩阵对 Value（同为通道维度）进行加权，最后还原到 [B, C, H, W]

    复杂度与显存：O(C^2 + C·H·W)，对 C=4 的多光谱数据非常友好。
    """

    def __init__(self, num_bands=4):
        super(CrossBandInteraction, self).__init__()
        self.num_bands = num_bands

        # Value / Output 投影（保持与原版接口类似）
        self.value_conv = nn.Conv2d(num_bands, num_bands, 1)
        self.output_conv = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, 1, bias=False),
            nn.BatchNorm2d(num_bands)
        )

        # 残差权重
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] with cross-band interaction
        """
        B, C, H, W = x.shape

        # 展平空间维度，计算通道间相似度（归一化以稳定数值）
        x_flat = x.view(B, C, -1)  # [B, C, HW]
        q = F.normalize(x_flat, dim=2)  # [B, C, HW]
        k = F.normalize(x_flat, dim=2)  # [B, C, HW]

        # 通道间注意力 [B, C, C]
        attn = torch.bmm(q, k.transpose(1, 2))  # [B, C, C]
        attn = F.softmax(attn, dim=-1)

        # 对 Value 做加权并还原形状
        v = self.value_conv(x).view(B, C, -1)  # [B, C, HW]
        out = torch.bmm(attn, v).view(B, C, H, W)

        out = self.output_conv(out)
        out = self.gamma * out + x
        return out


class SpatialPyramidAttention(nn.Module):
    """
    空间金字塔注意力
    
    参考 PSPNet 的金字塔池化 + Attention 机制
    捕获不同尺度的空间上下文
    """
    
    def __init__(self, in_channels, pool_sizes=[1, 2, 4, 8]):
        super(SpatialPyramidAttention, self).__init__()
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            spatial_weight: [B, 1, H, W]
        """
        h, w = x.size(2), x.size(3)
        
        # 多尺度特征
        pyramid_feats = []
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            pyramid_feats.append(feat)
        
        # 拼接
        pyramid_feats = torch.cat(pyramid_feats, dim=1)  # [B, C, H, W]
        
        # 生成空间注意力图
        spatial_weight = self.fusion(pyramid_feats)
        
        return spatial_weight


class AdvancedSSAF(nn.Module):
    """
    Advanced Spectral-Spatial Attention Fusion
    
    集成四个模块：
    1. SpectralIndexModule: 利用物理先验（光谱指数）
    2. MultiScaleSpectralAttention: 多尺度光谱注意力
    3. CrossBandInteraction: 跨波段交互
    4. SpatialPyramidAttention: 空间金字塔注意力
    
    设计原则：
    - 充分利用遥感领域知识
    - 多尺度特征提取
    - 显式建模波段交互
    - 强正则化（防止过拟合）
    """
    
    def __init__(self, num_bands=4, reduction=2, use_index=True, use_interaction=True):
        super(AdvancedSSAF, self).__init__()
        self.num_bands = num_bands
        self.use_index = use_index
        self.use_interaction = use_interaction
        
        # ============ 1. 光谱指数模块（物理先验）============
        if use_index and num_bands == 4:
            self.spectral_index = SpectralIndexModule(num_bands)
        else:
            self.spectral_index = None
        
        # ============ 2. 多尺度光谱注意力 ============
        self.spectral_attention = MultiScaleSpectralAttention(num_bands, reduction)
        
        # ============ 3. 跨波段交互（可选）============
        if use_interaction:
            self.band_interaction = CrossBandInteraction(num_bands)
        else:
            self.band_interaction = None
        
        # ============ 4. 空间金字塔注意力 ============
        self.spatial_attention = SpatialPyramidAttention(num_bands, pool_sizes=[1, 2, 4, 8])
        
        # ============ 5. 门控融合 ============
        # 自适应融合光谱指数权重和注意力权重
        if self.spectral_index is not None:
            self.gate = nn.Sequential(
                nn.Conv2d(num_bands * 2, num_bands, 1),
                nn.Sigmoid()
            )
        
        # 残差权重（防止注意力过强）
        self.alpha = nn.Parameter(torch.tensor(0.3))
        
        # Dropout（防止过拟合）
        self.dropout = nn.Dropout2d(0.15)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - 输入多光谱图像
        
        Returns:
            enhanced: [B, C, H, W] - 增强后的特征
            weights_dict: dict - 各种权重（用于可视化）
        """
        B, C, H, W = x.shape
        identity = x
        
        weights_dict = {}
        
        # ============ 1. 光谱指数增强（物理先验）============
        if self.spectral_index is not None:
            index_weights = self.spectral_index(x)  # [B, C, H, W]
            weights_dict['index_weights'] = index_weights.mean(dim=[2, 3])  # [B, C]
        
        # ============ 2. 多尺度光谱注意力 ============
        spectral_weights = self.spectral_attention(x)  # [B, C, 1, 1]
        # 保持 [B, C, 1, 1] 形状，便于外部可视化
        weights_dict['spectral_weights'] = spectral_weights
        
        # 融合光谱指数和光谱注意力
        if self.spectral_index is not None:
            # 门控融合：自适应选择物理先验和学习权重
            index_weights_broadcast = index_weights  # [B, C, H, W]
            spectral_weights_broadcast = spectral_weights.expand(-1, -1, H, W)  # [B, C, H, W]
            
            gate_input = torch.cat([index_weights_broadcast, spectral_weights_broadcast], dim=1)
            gate = self.gate(gate_input)  # [B, C, H, W]
            
            final_spectral_weights = gate * index_weights + (1 - gate) * spectral_weights_broadcast
        else:
            final_spectral_weights = spectral_weights.expand(-1, -1, H, W)
        
        # 应用光谱权重
        x_spectral = x * final_spectral_weights * C  # 乘 C 补偿归一化
        
        # ============ 3. 跨波段交互 ============
        if self.band_interaction is not None:
            x_interact = self.band_interaction(x_spectral)
            x_interact = self.dropout(x_interact)  # 正则化
        else:
            x_interact = x_spectral
        
        # ============ 4. 空间注意力 ============
        spatial_weights = self.spatial_attention(x_interact)  # [B, 1, H, W]
        weights_dict['spatial_weights'] = spatial_weights
        
        x_spatial = x_interact * spatial_weights
        
        # ============ 5. 自适应残差连接 ============
        alpha = torch.sigmoid(self.alpha).clamp(max=0.5)
        x_final = alpha * x_spatial + (1 - alpha) * identity
        
        weights_dict['alpha'] = alpha
        
        return x_final, weights_dict


# ============ 轻量级版本（快速实验）============
class LightAdvancedSSAF(nn.Module):
    """
    轻量级版本 - 只保留核心模块
    
    用于快速验证设计是否有效
    """
    
    def __init__(self, num_bands=4):
        super(LightAdvancedSSAF, self).__init__()
        self.num_bands = num_bands
        
        # 光谱指数（物理先验）
        self.spectral_index = SpectralIndexModule(num_bands) if num_bands == 4 else None
        
        # 简化的多尺度光谱注意力
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, num_bands, 1),
            nn.Softmax(dim=1)
        )
        
        # 简化的空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_bands, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 残差
        self.alpha = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, x):
        identity = x
        
        # 光谱
        if self.spectral_index is not None:
            index_weights = self.spectral_index(x)
        spectral_weights = self.spectral_attention(x)
        
        if self.spectral_index is not None:
            spectral_weights = 0.5 * (index_weights.mean(dim=[2,3], keepdim=True) + spectral_weights)
        
        x_spec = x * spectral_weights * self.num_bands
        
        # 空间
        spatial_weights = self.spatial_attention(x_spec)
        x_out = x_spec * spatial_weights
        
        # 残差
        alpha = torch.sigmoid(self.alpha).clamp(max=0.5)
        x_final = alpha * x_out + (1 - alpha) * identity
        
        return x_final, {'spectral_weights': spectral_weights.squeeze()}


# ============ 测试代码 ============
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Testing Advanced SSAF")
    print("="*60)
    
    # 测试完整版
    model = AdvancedSSAF(num_bands=4, reduction=2, 
                        use_index=True, use_interaction=True).to(device)
    x = torch.randn(4, 4, 256, 256).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    # 前向传播
    out, weights_dict = model(x)
    
    print(f"Output shape: {out.shape}")
    print(f"\nLearned weights:")
    
    if 'spectral_weights' in weights_dict:
        spec_w = weights_dict['spectral_weights'].mean(dim=0).cpu()
        print(f"  Spectral - Blue: {spec_w[0]:.4f}, Green: {spec_w[1]:.4f}, "
              f"Red: {spec_w[2]:.4f}, NIR: {spec_w[3]:.4f}")
        print(f"  Sum: {spec_w.sum():.4f}")
    
    if 'index_weights' in weights_dict:
        idx_w = weights_dict['index_weights'].mean(dim=0).cpu()
        print(f"  Index - Blue: {idx_w[0]:.4f}, Green: {idx_w[1]:.4f}, "
              f"Red: {idx_w[2]:.4f}, NIR: {idx_w[3]:.4f}")
    
    if 'alpha' in weights_dict:
        print(f"  Residual alpha: {weights_dict['alpha'].item():.4f}")
    
    # 参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    # 测试轻量级版本
    print("\n" + "="*60)
    print("Testing Light Advanced SSAF")
    print("="*60)
    
    light_model = LightAdvancedSSAF(num_bands=4).to(device)
    out_light, weights_light = light_model(x)
    
    params_light = sum(p.numel() for p in light_model.parameters())
    print(f"Light version parameters: {params_light:,}")
    print(f"Parameter reduction: {(1 - params_light/params)*100:.1f}%")
    
    spec_w_light = weights_light['spectral_weights'].mean(dim=0).cpu()
    print(f"\nLight version spectral weights:")
    print(f"  Blue: {spec_w_light[0]:.4f}, Green: {spec_w_light[1]:.4f}, "
          f"Red: {spec_w_light[2]:.4f}, NIR: {spec_w_light[3]:.4f}")