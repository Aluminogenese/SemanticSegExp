"""
HRNet for High-Resolution Building Segmentation

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


class HRNet(nn.Module):
    """
    纯净版 HRNet - 无任何创新模块
    用于对比实验的Baseline
    """
    def __init__(self, in_channels=4, num_classes=1, base_channels=48):
        super(HRNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        
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
        
        # 简单的特征聚合和分类头
        total_channels = base_channels * 15
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_channels, base_channels * 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
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
        
        # Stage 1
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
        out = self.final_conv(feats)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out
