"""
HRNet + OCR (Object-Contextual Representations) for High-Resolution Building Segmentation
结合了：
1. HRNet的多尺度特征融合
2. OCR的上下文增强
3. 边缘注意力机制
4. 深度监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.stride = stride

    def forward(self, x):
        residual = x # torch.Size([8, 64, 128, 128])
        out = self.conv1(x) # torch.Size([8, 48, 128, 128])
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) # torch.Size([8, 48, 128, 128])
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x) # torch.Size([8, 48, 128, 128])
        out += residual
        out = self.relu(out)
        return out


class EdgeAttention(nn.Module):
    """边缘注意力模块 - 专门增强建筑物边界"""
    def __init__(self, in_channels):
        super(EdgeAttention, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        edge_weight = self.edge_conv(x)
        return x * edge_weight


class SpatialOCR(nn.Module):
    """Object-Contextual Representations - 上下文增强"""
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
        obj_context = self.object_context(feats) # torch.Size([8, 96, 128, 128])
        B, C, H, W = obj_context.shape
        obj_context = obj_context.view(B, C, -1) # torch.Size([8, 96, 16384])
        soft_regions_flat = soft_regions.view(B, 1, -1) # torch.Size([8, 1, 16384])
        
        # 加权池化
        region_context = torch.bmm(obj_context, soft_regions_flat.transpose(1, 2)) # torch.Size([8, 96, 1])
        region_context = region_context / (soft_regions_flat.sum(dim=2, keepdim=True) + 1e-6)
        region_context = region_context.unsqueeze(-1).expand(-1, -1, H, W) # torch.Size([8, 96, 128, 128])
        
        # 像素上下文
        pix_context = self.pixel_context(feats) # torch.Size([8, 96, 128, 128])
        
        # 融合
        concat = torch.cat([pix_context, region_context], dim=1) # torch.Size([8, 192, 128, 128])
        output = self.fusion(concat) # torch.Size([8, 96, 128, 128])
        
        return output, aux_pred


class HRNetBranch(nn.Module):
    """HRNet的单个分支"""
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
    """多尺度特征融合层"""
    def __init__(self, num_branches, in_channels_list):
        super(FuseLayer, self).__init__()
        self.num_branches = num_branches
        self.in_channels_list = in_channels_list
        
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    # 上采样
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels_list[j], in_channels_list[i], 1, bias=False),
                        nn.BatchNorm2d(in_channels_list[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=True)
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # 下采样
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


class HRNetOCR(nn.Module):
    """
    高分辨率网络 + OCR，专为建筑物分割优化
    """
    def __init__(self, in_channels=4, num_classes=1, base_channels=48):
        super(HRNetOCR, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        
        # Stage 1: 初始特征提取
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 2: 两个分支 [1x, 2x下采样]
        # 第一个分支处理stage1的输出(64通道)
        self.layer1 = HRNetBranch(4, 64, base_channels)
        
        # Transition: 从stage1到stage2，创建两个分支
        self.transition1 = nn.ModuleList([
            None,  # 第一个分支直接使用layer1的输出
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Stage2的branches
        self.stage2_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels * 2, base_channels * 2)
        ])
        self.fuse2 = FuseLayer(2, [base_channels, base_channels * 2])
        
        # Transition2: 从stage2到stage3，添加第三个分支
        self.transition2 = nn.ModuleList([
            None,
            None,
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Stage 3: 三个分支 [1x, 2x, 4x下采样]
        self.stage3_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels * 2, base_channels * 2),
            HRNetBranch(4, base_channels * 4, base_channels * 4)
        ])
        self.fuse3 = FuseLayer(3, [base_channels, base_channels * 2, base_channels * 4])
        
        # Transition3: 从stage3到stage4，添加第四个分支
        self.transition3 = nn.ModuleList([
            None,
            None,
            None,
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Stage 4: 四个分支 [1x, 2x, 4x, 8x下采样]
        self.stage4_branches = nn.ModuleList([
            HRNetBranch(4, base_channels, base_channels),
            HRNetBranch(4, base_channels * 2, base_channels * 2),
            HRNetBranch(4, base_channels * 4, base_channels * 4),
            HRNetBranch(4, base_channels * 8, base_channels * 8)
        ])
        self.fuse4 = FuseLayer(4, [base_channels, base_channels * 2, base_channels * 4, base_channels * 8])
        
        # 特征聚合
        total_channels = base_channels * 15  # 48+96+192+384
        self.aggregate = nn.Sequential(
            nn.Conv2d(total_channels, base_channels * 4, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # OCR模块
        self.ocr = SpatialOCR(
            in_channels=base_channels * 4,
            key_channels=base_channels * 2,
            out_channels=base_channels * 2,
            num_classes=num_classes
        )
        
        # 边缘注意力
        self.edge_attention = EdgeAttention(base_channels * 2)
        
        # 最终分类头
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
        input_size = x.size()[2:] # torch.size([512, 512])
        
        # Stage 1: 初始特征提取
        x = self.conv1(x) # torch.Size([8, 64, 256, 256])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x) # torch.Size([8, 64, 128, 128])
        x = self.bn2(x)
        x = self.relu(x)  # /4, 64通道
        
        # Layer1: 处理成base_channels
        x = self.layer1(x)  # /4, base_channels  torch.Size([8, 48, 128, 128])
        
        # Transition1: 创建两个分支
        x_list = [x]  # 第一个分支
        x_list.append(self.transition1[1](x))  # 第二个分支，下采样 torch.Size([8, 96, 64, 64])
        
        # Stage 2
        x_list = [branch(x_list[i]) for i, branch in enumerate(self.stage2_branches)]
        x_list = self.fuse2(x_list) # torch.Size([8, 48, 128, 128]), torch.Size([8, 96, 64, 64])
        
        # Transition2: 添加第三个分支
        x_list_new = x_list.copy()
        x_list_new.append(self.transition2[2](x_list[-1]))  # 从最后一个分支下采样 2:torch.Size([8, 192, 32, 32])
        
        # Stage 3
        x_list = [branch(x_list_new[i]) for i, branch in enumerate(self.stage3_branches)]
        x_list = self.fuse3(x_list) # torch.Size([8, 48, 128, 128]), torch.Size([8, 96, 64, 64]), torch.Size([8, 192, 32, 32])
        
        # Transition3: 添加第四个分支
        x_list_new = x_list.copy()
        x_list_new.append(self.transition3[3](x_list[-1]))  # 从最后一个分支下采样
        
        # Stage 4
        x_list = [branch(x_list_new[i]) for i, branch in enumerate(self.stage4_branches)]
        x_list = self.fuse4(x_list) # torch.Size([8, 48, 128, 128]), torch.Size([8, 96, 64, 64]), torch.Size([8, 192, 32, 32]), torch.Size([8, 384, 16, 16])
        
        # 特征聚合：所有尺度上采样到最高分辨率
        x0 = x_list[0] # torch.Size([8, 48, 128, 128])
        x1 = F.interpolate(x_list[1], size=x0.shape[2:], mode='bilinear', align_corners=True) # torch.Size([8, 96, 128, 128])
        x2 = F.interpolate(x_list[2], size=x0.shape[2:], mode='bilinear', align_corners=True) # torch.Size([8, 192, 128, 128])
        x3 = F.interpolate(x_list[3], size=x0.shape[2:], mode='bilinear', align_corners=True) # torch.Size([8, 384, 128, 128])
        
        feats = torch.cat([x0, x1, x2, x3], dim=1) # torch.Size([8, 720, 128, 128])
        feats = self.aggregate(feats) # torch.Size([8, 192, 128, 128])
        
        # OCR增强
        feats, aux_pred = self.ocr(feats) # torch.Size([8, 96, 128, 128]), torch.Size([8, 1, 128, 128])
        
        # 边缘注意力
        feats = self.edge_attention(feats) # torch.Size([8, 96, 128, 128])
        
        # 最终预测
        out = self.final_conv(feats) # torch.Size([8, 1, 128, 128])
        
        # 上采样到原始尺寸
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True) # torch.Size([8, 1, 512, 512])
        
        if self.training:
            aux_pred = F.interpolate(aux_pred, size=input_size, mode='bilinear', align_corners=True)
            return out, aux_pred
        else:
            return out
