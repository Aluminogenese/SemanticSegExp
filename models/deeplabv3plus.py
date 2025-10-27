import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Low-level features
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # High-level features (simplified ResNet-like)
        self.layer1 = self._make_layer(256, 256, blocks=3)
        self.layer2 = self._make_layer(256, 512, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 1024, blocks=6, stride=2)
        
        # ASPP
        if output_stride == 16:
            dilation_rates = [6, 12, 18]
        elif output_stride == 8:
            dilation_rates = [12, 24, 36]
        else:
            dilation_rates = [6, 12, 18]
        
        self.aspp = ASPPModule(1024, 256, dilation_rates)
        
        # Decoder
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
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
        
        # Encoder
        x = self.conv1(x)  # 1/2
        low_level_feat = x
        x = self.maxpool(x)  # 1/4
        low_level_feat = self.low_level_conv(x)  # 1/8, 用于decoder
        
        x = self.layer1(low_level_feat)  # 1/8
        x = self.layer2(x)  # 1/16
        x = self.layer3(x)  # 1/32
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = F.interpolate(x, size=low_level_feat.shape[2:], 
                         mode='bilinear', align_corners=True)
        low_level_feat = self.decoder_conv1(low_level_feat)
        
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder_conv2(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x