import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# UNetFormer 简化实现
# 参考: Wang et al., "UNetFormer", ISPRS J. Photogramm. Remote Sens., 2022
# 核心思想: CNN encoder + Transformer decoder + 高效自注意力
# 下面是适配你的4波段输入和二分类输出的简化版本
# =============================================================================

class EfficientSelfAttention(nn.Module):
    """
    高效自注意力（线性复杂度）
    来自UNetFormer，用于高分辨率特征图
    """
    def __init__(self, channels, num_heads=8, sr_ratio=2):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5
        
        self.q  = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)
        
        # 空间降维，降低KV的分辨率
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(channels, channels, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(channels)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0,2,1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0,2,1)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)
        
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, channels, num_heads=8, sr_ratio=2, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = EfficientSelfAttention(channels, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp   = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels)
        )
    
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class UNetFormer(nn.Module):
    """
    UNetFormer: CNN Encoder + Transformer Decoder
    适配4波段输入、二分类输出
    
    参考: Wang et al., ISPRS J. Photogramm. Remote Sens., 2022
    代码简化自: https://github.com/WangLibo1995/GeoSeg
    """
    
    def __init__(self, in_channels=4, num_classes=1,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16]):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes  = num_classes
        
        # ── CNN Encoder (ResNet风格) ──
        self.enc1 = nn.Sequential(
            ConvBNReLU(in_channels, embed_dims[0], k=7, s=2, p=3),
            ConvBNReLU(embed_dims[0], embed_dims[0])
        )  # /2
        self.enc2 = nn.Sequential(
            ConvBNReLU(embed_dims[0], embed_dims[1], s=2),
            ConvBNReLU(embed_dims[1], embed_dims[1])
        )  # /4
        self.enc3 = nn.Sequential(
            ConvBNReLU(embed_dims[1], embed_dims[2], s=2),
            ConvBNReLU(embed_dims[2], embed_dims[2])
        )  # /8
        self.enc4 = nn.Sequential(
            ConvBNReLU(embed_dims[2], embed_dims[3], s=2),
            ConvBNReLU(embed_dims[3], embed_dims[3])
        )  # /16
        
        # ── Transformer Bottleneck ──
        self.patch_embed = nn.Conv2d(embed_dims[3], embed_dims[3], 1)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], sr_ratio=1)
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(embed_dims[3])
        
        # ── Decoder ──
        self.dec4 = self._make_dec(embed_dims[3] + embed_dims[2], embed_dims[2])
        self.dec3 = self._make_dec(embed_dims[2] + embed_dims[1], embed_dims[1])
        self.dec2 = self._make_dec(embed_dims[1] + embed_dims[0], embed_dims[0])
        self.dec1 = self._make_dec(embed_dims[0], embed_dims[0] // 2)
        
        self.head = nn.Conv2d(embed_dims[0] // 2, num_classes, 1)
        
        self._init_weights()
    
    def _make_dec(self, in_ch, out_ch):
        return nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        input_size = x.shape[2:]
        
        # Encoder
        e1 = self.enc1(x)   # [B, 64,  H/2,  W/2]
        e2 = self.enc2(e1)  # [B, 128, H/4,  W/4]
        e3 = self.enc3(e2)  # [B, 256, H/8,  W/8]
        e4 = self.enc4(e3)  # [B, 512, H/16, W/16]
        
        # Transformer bottleneck
        H4, W4 = e4.shape[2:]
        t = self.patch_embed(e4)
        t = t.flatten(2).transpose(1, 2)  # [B, H*W, C]
        for blk in self.transformer:
            t = blk(t, H4, W4)
        t = self.norm(t).transpose(1, 2).reshape(B, -1, H4, W4)
        
        # Decoder (UNet style with skip connections)
        d4 = self.dec4(torch.cat([
            F.interpolate(t,  e3.shape[2:], mode='bilinear', align_corners=True),
            e3
        ], dim=1))
        
        d3 = self.dec3(torch.cat([
            F.interpolate(d4, e2.shape[2:], mode='bilinear', align_corners=True),
            e2
        ], dim=1))
        
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, e1.shape[2:], mode='bilinear', align_corners=True),
            e1
        ], dim=1))
        
        d1 = self.dec1(F.interpolate(d2, input_size, mode='bilinear', align_corners=True))
        
        return self.head(d1)