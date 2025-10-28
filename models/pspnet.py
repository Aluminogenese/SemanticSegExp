import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """ResNet Bottleneck Block"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)
    

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
    def __init__(self, in_channels=4, num_classes=1, output_stride=16):
        super(PSPNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-101 layers: [3, 4, 23, 3]
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)     # 256 channels
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)    # 512 channels
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=2)   # 1024 channels
        if output_stride == 16:
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1, dilation=2)
        else:
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        # 输出: 2048 channels
        
        # PSP模块 (输入 2048 channels)
        self.psp = PyramidPoolingModule(2048, pool_sizes=[1, 2, 3, 6])
        
        # 最终分类层
        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        # 辅助损失分支 (来自 layer3: 1024 channels)
        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        self._init_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """构建 ResNet layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # ResNet-101 Encoder
        x = self.conv1(x)      # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # /4
        
        x = self.layer1(x)     # /4, 256 channels
        x = self.layer2(x)     # /8, 512 channels
        x_aux = self.layer3(x) # /16, 1024 channels
        x = self.layer4(x_aux) # /16, 2048 channels
        
        # PSP
        x = self.psp(x)
        x = self.final(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        if self.training:
            aux = self.aux(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
            return x, aux
        else:
            return x