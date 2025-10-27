import torch
from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNetOCR, MSHRNetOCR

# ============================================================================
# 模型测试
# ============================================================================

batch_size = 8
in_channels = 4
img_size = 512

x = torch.randn(batch_size, in_channels, img_size, img_size)

print("Testing models...")
print("="*60)

# 1. UNet
print("\n1. UNet")
model = UNet(in_channels=in_channels, num_classes=1)
output = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params/1e6:.2f}M")

# 2. UNet++
print("\n2. UNet++")
model = UNetPlusPlus(in_channels=in_channels, num_classes=1)
output = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params/1e6:.2f}M")

# 3. PSPNet
print("\n3. PSPNet")
model = PSPNet(in_channels=in_channels, num_classes=1)
model.eval()
output = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params/1e6:.2f}M")

# 4. DeepLabV3+
print("\n4. DeepLabV3+")
model = DeepLabV3Plus(in_channels=in_channels, num_classes=1)
output = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params/1e6:.2f}M")

# 5. HRNet + OCR
print("\n5. HRNet + OCR")
model = HRNetOCR(in_channels=in_channels, num_classes=1)
output = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output[0].shape}")
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params/1e6:.2f}M")

# 6. MS-HRNet + OCR
print("\n6. MS-HRNet + OCR")
model = MSHRNetOCR(in_channels=in_channels, num_classes=1)
output = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output[0].shape}")
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params/1e6:.2f}M")

print("\n" + "="*60)
print("All models tested successfully!")