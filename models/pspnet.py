import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """金字塔池化模块"""
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        
        for stage in self.stages:
            pyramid = stage(x)
            pyramid = F.interpolate(pyramid, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(pyramid)
        
        output = torch.cat(pyramids, dim=1)
        output = self.bottleneck(output)
        return output


class PSPNet(nn.Module):
    """
    PSPNet 实现
    
    论文: "Pyramid Scene Parsing Network"
    年份: 2017, CVPR
    
    特点:
    - 金字塔池化模块捕获多尺度上下文
    - 使用ResNet作为backbone
    - 辅助损失
    """
    def __init__(self, in_channels=4, num_classes=1, backbone='resnet50'):
        super(PSPNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        
        # 简化的backbone (类ResNet结构)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 128, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, 256, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, 512, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, 512, blocks=3, stride=1)  # 降低stride
        
        # PSP模块
        self.psp = PyramidPoolingModule(512, pool_sizes=[1, 2, 3, 6])
        
        # 最终分类层
        self.final = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # 辅助损失分支
        self.aux = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        x = self.layer0(x)  # 1/4
        x = self.layer1(x)  # 1/4
        x = self.layer2(x)  # 1/8
        x_aux = self.layer3(x)  # 1/16
        x = self.layer4(x_aux)  # 1/16
        
        x = self.psp(x)
        x = self.final(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        if self.training:
            aux = self.aux(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
            return x, aux
        else:
            return x