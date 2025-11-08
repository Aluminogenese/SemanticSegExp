
"""
MS-HRNet: Multi-Spectral High-Resolution Network for Building Segmentation

1. Spectral-Spatial Attention Fusion (SSAF) - 多光谱注意力融合
2. Multi-Scale Boundary Refinement (MSBR) - 多尺度边界细化
3. Object-Contextual Representations (OCR) - 上下文增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralSpatialAttentionFusion(nn.Module):
    """
    Spectral-Spatial Attention Fusion (SSAF)
    
    1. 波段重要性自适应学习 (Spectral Attention)
    2. 波段间交互建模 (Inter-Band Interaction)
    3. 空间注意力引导 (Spatial Attention)
    
    专门针对 RGB+NIR 4波段遥感影像设计
    """
    def __init__(self, num_bands=4, reduction=2, num_heads=2, init_temperature=5.0):
        super(SpectralSpatialAttentionFusion, self).__init__()
        self.num_bands = num_bands
        self.num_heads = num_heads

        self.spectral_attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_bands, num_bands // reduction, 1, bias=True),
                nn.LayerNorm([num_bands // reduction, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_bands // reduction, num_bands, 1, bias=True),
                # 注意：这里不用 Sigmoid，后面会用 Softmax
            )
            for _ in range(num_heads)
        ])

        self.temperature = nn.Parameter(torch.ones(num_heads) * init_temperature)

        self.band_interaction = nn.Sequential(
            # 深度可分离卷积
            nn.Conv2d(num_bands, num_bands * 2, 3, padding=1, 
                     groups=num_bands, bias=False),
            nn.BatchNorm2d(num_bands * 2),
            nn.ReLU(inplace=True),
            
            # 点卷积
            nn.Conv2d(num_bands * 2, num_bands, 1, bias=False),
            nn.BatchNorm2d(num_bands)
        )
        # NIR 特征增强（针对建筑物分割）
        if num_bands == 4:
            self.nir_enhance = nn.Sequential(
                nn.Conv2d(1, num_bands // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_bands // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_bands // 2, num_bands, 1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.nir_enhance = None
        
        self.spatial_branches = nn.ModuleList([
            self._make_spatial_branch(num_bands, dilation=d)
            for d in [1, 2, 3]
        ])
        
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.Sigmoid()
        )
        
        # ============ 4. 门控融合 ============
        # 控制注意力的强度（防止过强）
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的残差权重

        self._init_weights()

    def _make_spatial_branch(self, in_channels, dilation=1):
        """创建单个空间注意力分支"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 
                     padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 关键：初始化 bias 为 0，让初始权重接近均匀分布
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W] - 4波段输入 (RGB + NIR)
        Returns:
            enhanced: [B, 4, H, W] - 增强后的特征
            spectral_weights: [B, 4, 1, 1] - 波段权重（用于可视化）
        """
        B, C, H, W = x.shape
        identity = x
        
        # ============ 1. 多头光谱注意力（Softmax 归一化）============
        spectral_logits_list = []
        for head in self.spectral_attention_heads:
            logits = head(x)  # [B, C, 1, 1]
            spectral_logits_list.append(logits)
        
        # 多头融合：对每个头使用温度缩放的 Softmax
        spectral_weights_list = []
        for i, logits in enumerate(spectral_logits_list):
            # 温度缩放 + Softmax（在通道维度）
            temp = self.temperature[i].abs().clamp(min=1.0)  # 确保温度 >= 1
            weights = F.softmax(logits / temp, dim=1)  # [B, C, 1, 1]
            spectral_weights_list.append(weights)
        
        # 平均多个头的权重
        spectral_weights = torch.stack(spectral_weights_list).mean(dim=0)
        
        # 归一化到和为 1（理论上已经是，但为了数值稳定）
        spectral_weights = spectral_weights / (spectral_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 应用光谱注意力
        x_spectral = x * spectral_weights * C  # 乘以 C 补偿 Softmax 的缩放
        
        # ============ 2. 波段交互建模 ============
        x_interact = self.band_interaction(x_spectral)
        
        # NIR 增强
        if self.nir_enhance is not None and C == 4:
            nir_band = x[:, 3:4, :, :]
            nir_weight = self.nir_enhance(nir_band)
            x_interact = x_interact * nir_weight
        
        # ============ 3. 多尺度空间注意力 ============
        spatial_maps = [branch(x_interact) for branch in self.spatial_branches]
        spatial_maps = torch.cat(spatial_maps, dim=1)
        spatial_weights = self.spatial_fusion(spatial_maps)
        
        x_final = x_interact * spatial_weights
        
        # ============ 4. 自适应残差连接 ============
        # alpha 控制注意力的强度（0.5 表示 50% 原始 + 50% 注意力）
        alpha_clamped = torch.sigmoid(self.alpha)  # 限制在 [0, 1]
        x_output = alpha_clamped * x_final + (1 - alpha_clamped) * identity
        
        return x_output, spectral_weights



class MultiScaleBoundaryRefinement(nn.Module):
    """
    Multi-Scale Boundary Refinement (MSBR)

    1. 多尺度边界特征提取 (dilation rates: 1, 2, 4)
    2. 边界感知的特征增强
    3. 自适应边界权重学习
    
    相比原来的 EdgeAttention:
    - 原来: 单尺度 3x3 卷积
    - 现在: 多尺度空洞卷积 + 特征融合
    """
    def __init__(self, in_channels, reduction=4):
        super(MultiScaleBoundaryRefinement, self).__init__()
        mid_channels = in_channels // reduction
        
        # Multi-scale boundary detection
        self.boundary_branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.boundary_branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.boundary_branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Boundary feature fusion
        self.boundary_fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Boundary weight generation
        self.boundary_weight = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            out: [B, C, H, W] 边界增强后的特征
            boundary_map: [B, 1, H, W] 边界概率图
        """
        # Multi-scale boundary detection
        b1 = self.boundary_branch1(x)
        b2 = self.boundary_branch2(x)
        b3 = self.boundary_branch3(x)
        
        # Fusion
        boundary_feat = torch.cat([b1, b2, b3], dim=1)
        boundary_feat = self.boundary_fusion(boundary_feat)
        
        # Generate boundary map
        boundary_map = self.boundary_weight(boundary_feat)
        
        # Feature enhancement: 边界区域加强
        enhanced = self.feature_enhance(x)
        out = x + enhanced * boundary_map
        
        return out, boundary_map


class SpatialOCR(nn.Module):
    """
    Object-Contextual Representations (OCR)
    
    技术贡献: 首次应用于多光谱建筑物分割
    优势: 捕获密集建筑物场景中的长程依赖
    """
    def __init__(self, in_channels, key_channels, out_channels, num_classes=1):
        super(SpatialOCR, self).__init__()
        self.object_context = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pixel_context = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(key_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aux_head = nn.Conv2d(in_channels, num_classes, 1)
        
    def forward(self, feats):
        # 生成软分割用于区域池化
        aux_pred = self.aux_head(feats)
        soft_regions = torch.sigmoid(aux_pred)
        
        # 对象上下文
        obj_context = self.object_context(feats)
        B, C, H, W = obj_context.shape
        obj_context = obj_context.view(B, C, -1)
        soft_regions_flat = soft_regions.view(B, 1, -1)
        
        # 加权池化
        region_context = torch.bmm(obj_context, soft_regions_flat.transpose(1, 2))
        region_context = region_context / (soft_regions_flat.sum(dim=2, keepdim=True) + 1e-6)
        region_context = region_context.unsqueeze(-1).expand(-1, -1, H, W)
        
        # 像素上下文
        pix_context = self.pixel_context(feats)
        
        # 融合
        concat = torch.cat([pix_context, region_context], dim=1)
        output = self.fusion(concat)
        
        return output, aux_pred


# ==================== HRNet Backbone Components ====================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class HRNetBranch(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels):
        super(HRNetBranch, self).__init__()
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, downsample=downsample))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class FuseLayer(nn.Module):
    def __init__(self, num_branches, in_channels_list):
        super(FuseLayer, self).__init__()
        self.num_branches = num_branches
        self.in_channels_list = in_channels_list
        
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels_list[j], in_channels_list[i], 1, bias=False),
                        nn.BatchNorm2d(in_channels_list[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=True)
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            convs.append(nn.Sequential(
                                nn.Conv2d(in_channels_list[j], in_channels_list[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(in_channels_list[i])
                            ))
                        else:
                            convs.append(nn.Sequential(
                                nn.Conv2d(in_channels_list[j], in_channels_list[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(in_channels_list[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*convs))
            self.fuse_layers.append(fuse_layer)
    
    def forward(self, x):
        out = []
        for i in range(self.num_branches):
            y = 0
            for j in range(self.num_branches):
                if self.fuse_layers[i][j] is not None:
                    y += self.fuse_layers[i][j](x[j])
                else:
                    y += x[j]
            out.append(F.relu(y))
        return out


# ==================== MS-HRNet Main Architecture ====================
class MSHRNetOCR(nn.Module):
    """
    MS-HRNet: Multi-Spectral High-Resolution Network
    
    创新组合:
    1. SSAF: 输入端的多光谱自适应融合
    2. HRNet: 高分辨率特征提取
    3. OCR: 上下文增强
    4. MSBR: 边界细化
    """
    def __init__(self, in_channels=4, num_classes=1, base_channels=48):
        super(MSHRNetOCR, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        
        # ============ SSAF (输入端) ============
        self.ssaf = SpectralSpatialAttentionFusion(
            num_bands=in_channels, 
            reduction=2
        )
        
        # ============ HRNet Backbone ============
        # Stage 1
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
        self.fuse4 = FuseLayer(4, [base_channels, base_channels * 2, base_channels * 4, base_channels * 8])
        
        # Feature aggregation
        total_channels = base_channels * 15
        self.aggregate = nn.Sequential(
            nn.Conv2d(total_channels, base_channels * 4, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # ============ OCR ============
        self.ocr = SpatialOCR(
            in_channels=base_channels * 4,
            key_channels=base_channels * 2,
            out_channels=base_channels * 2,
            num_classes=num_classes
        )
        
        # ============ MSBR ============
        self.msbr = MultiScaleBoundaryRefinement(base_channels * 2, reduction=4)
        
        # Final classifier
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1, bias=False),
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
        
        # ============ SSAF: 多光谱融合 ============
        x, spectral_weights = self.ssaf(x)
        
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
        
        # ============ OCR: 上下文增强 ============
        feats, aux_pred = self.ocr(feats)
        
        # ============ MSBR: 边界细化 ============
        feats, boundary_map = self.msbr(feats)
        
        # Final prediction
        out = self.final_conv(feats)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        if self.training:
            # 返回多个输出用于深度监督
            aux_pred = F.interpolate(aux_pred, size=input_size, mode='bilinear', align_corners=True)
            boundary_map = F.interpolate(boundary_map, size=input_size, mode='bilinear', align_corners=True)
            return out, aux_pred, boundary_map, spectral_weights
        else:
            return out
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = MSHRNetOCR(in_channels=4, num_classes=1, base_channels=48)
    model.to(device)
    model.train()
    
    # 测试输入
    x = torch.randn(2, 4, 512, 512).to(device)
    
    print("Testing MSHRNetOCRImproved...")
    print(f"Input shape: {x.shape}")
    
    # 前向传播
    out, aux_pred, boundary_map, spectral_weights = model(x)
    
    print(f"Output shape: {out.shape}")
    print(f"Aux pred shape: {aux_pred.shape}")
    print(f"Boundary map shape: {boundary_map.shape}")
    print(f"Spectral weights shape: {spectral_weights.shape}")
    
    print("\nSpectral weights (mean across batch):")
    mean_weights = spectral_weights.mean(dim=0).squeeze()
    print(mean_weights)
    print(f"Variance: {torch.var(mean_weights).item():.4f}")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")