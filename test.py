# 在你的项目里运行这段代码，对比三种算法的 Dice
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import AdvancedDataset
from models import HRNet
from eval import eval_net
import pandas as pd

device = torch.device('cuda')
net = HRNet(in_channels=4, num_classes=1, base_channels=48)
net.load_state_dict(torch.load('checkpoints_potsdam/BEST_hrnet_combined_potsdam.pth'))
net.to(device).eval()

val_ds = AdvancedDataset(
    '/home/lucianlu/data/data_potsdam/val/images/',
    '/home/lucianlu/data/data_potsdam/val/labels/',
    augment=False, crop_size=512
)

# 方式1：训练时用的方式（drop_last=True）
loader_drop = DataLoader(val_ds, batch_size=4, shuffle=False,
                         num_workers=4, pin_memory=True, drop_last=True)
score_drop = eval_net(net, loader_drop, device)

# 方式2：drop_last=False
loader_full = DataLoader(val_ds, batch_size=4, shuffle=False,
                         num_workers=4, pin_memory=True, drop_last=False)
score_full = eval_net(net, loader_full, device)

print(f"训练时方式 (drop_last=True):  Dice = {score_drop:.4f}")
print(f"完整验证  (drop_last=False): Dice = {score_full:.4f}")
print(f"测试脚本  (全图推理):         Dice = 0.9270")
print(f"\n验证集图片数: {len(val_ds)}")
print(f"drop_last=True 实际用了: {len(loader_drop) * 4} 张")
print(f"drop_last=False 实际用了: {len(val_ds)} 张")