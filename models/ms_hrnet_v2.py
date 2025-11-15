import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssaf_improved import ImprovedSSAF  # 或 MinimalSSAF


# 复用原有的 HRNet 组件
from .ms_hrnet import (
    BasicBlock, HRNetBranch, FuseLayer, SpatialOCR
)


class MSHRNetV2(nn.Module):
    """
    MS-HRNet V2: 使用改进的 SSAF 模块
    
    主要改进:
    1. ImprovedSSAF 替代原 SSAF
    2. 返回更丰富的注意力图用于分析
    """
    
    def __init__(self, in_channels=4, num_classes=1, base_channels=48, 
                 use_minimal_ssaf=False):
        super(MSHRNetV2, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        
        # ============ 改进的 SSAF ============
        if use_minimal_ssaf:
            from .ssaf_improved import MinimalSSAF
            self.ssaf = MinimalSSAF(num_bands=in_channels)
        else:
            self.ssaf = ImprovedSSAF(num_bands=in_channels, reduction=2)
        
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
        
        # OCR
        self.ocr = SpatialOCR(
            in_channels=base_channels * 4,
            key_channels=base_channels * 2,
            out_channels=base_channels * 2,
            num_classes=num_classes
        )
        
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
        
        # OCR
        feats, aux_pred = self.ocr(feats)
        
        # Final prediction
        out = self.final_conv(feats)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        if self.training:
            aux_pred = F.interpolate(aux_pred, size=input_size, 
                                     mode='bilinear', align_corners=True)
            return out, aux_pred, attention_maps
        else:
            return out