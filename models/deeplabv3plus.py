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
    
class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        self.aspp = nn.ModuleList()
        
        # 1x1 conv
        self.aspp.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 3x3 conv with different dilation rates
        for dilation in dilation_rates:
            self.aspp.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, 
                         dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Image pooling
        self.aspp.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Concat projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilation_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.aspp:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                pool_out = conv(x)
                pool_out = F.interpolate(pool_out, size=x.shape[2:], 
                                        mode='bilinear', align_corners=True)
                res.append(pool_out)
            else:
                res.append(conv(x))
        
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 实现
    
    论文: "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    年份: 2018, ECCV
    
    特点:
    - Atrous Spatial Pyramid Pooling (ASPP)
    - Encoder-Decoder结构
    - 低层特征融合
    """
    def __init__(self, in_channels=4, num_classes=1, output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.inplanes = 64
        
        # ==================== 关键修改：ResNet-101 结构 ====================
        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-101 layers: [3, 4, 23, 3]
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)     # 256 channels
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)    # 512 channels
        
        # 根据 output_stride 调整 layer3 和 layer4
        if output_stride == 16:
            self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=2)   # 1024 channels
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1, dilation=2)  # 2048 channels
            dilation_rates = [6, 12, 18]
        elif output_stride == 8:
            self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=1, dilation=2)
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1, dilation=4)
            dilation_rates = [12, 24, 36]
        else:
            self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=2)
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
            dilation_rates = [6, 12, 18]
        # ===================================================================
        
        # Low-level features (来自 layer1: 256 channels)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # ASPP (输入 2048 channels)
        self.aspp = ASPPModule(2048, 256, dilation_rates)
        
        # Decoder
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
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
        x = self.conv1(x)         # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # /4
        
        low_level_feat = self.layer1(x)  # /4, 256 channels
        x = self.layer2(low_level_feat)  # /8, 512 channels
        x = self.layer3(x)        # /16 or /8, 1024 channels
        x = self.layer4(x)        # /16 or /8, 2048 channels
        
        # ASPP
        x = self.aspp(x)  # 256 channels
        
        # Decoder
        x = F.interpolate(x, size=low_level_feat.shape[2:], 
                         mode='bilinear', align_corners=True)
        low_level_feat = self.low_level_conv(low_level_feat)  # 48 channels
        
        x = torch.cat([x, low_level_feat], dim=1)  # 304 channels
        x = self.decoder_conv2(x)  # 256 channels
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x