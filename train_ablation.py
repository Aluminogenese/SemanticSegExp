# train_ablation.py
"""
SSAF子组件消融实验训练脚本
"""
import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval import eval_net
from utils.dataset import AdvancedDataset
from models.ms_hrnet_ablation import MSHRNetAblation
from train import CombinedLoss, parse_loss_weights

dir_checkpoint = 'checkpoints_ssaf_ablation/'


def train_ablation(variant, device, train_img, train_mask, val_img, val_mask,
                   in_channels=4, epochs=400, batch_size=4, lr=1e-3,
                   loss_weights=None, warmup_epochs=5):
    
    net = MSHRNetAblation(variant=variant, in_channels=in_channels, num_classes=1)
    net.to(device)
    
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    logging.info(f'Variant: {variant}, Params: {total_params:.2f}M')
    
    train_ds = AdvancedDataset(train_img, train_mask, augment=True,  crop_size=512)
    val_ds   = AdvancedDataset(val_img,   val_mask,   augment=False, crop_size=512)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=True)
    
    if loss_weights is None:
        loss_weights = {'bce': 1.0, 'dice': 1.0, 'focal': 0.5, 'boundary': 0.3}
    
    criterion = CombinedLoss(weights=loss_weights).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(
            np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    writer = SummaryWriter(comment=f'_ssaf_ablation_{variant}')
    
    best_dice = 0.0
    global_step = 0
    n_train = len(train_ds)
    
    for epoch in range(epochs):
        net.train()
        for batch in train_loader:
            imgs      = batch['image'].to(device, dtype=torch.float32)
            true_masks = batch['mask'].to(device, dtype=torch.float32)
            
            pred = net(imgs)
            loss = criterion(pred, true_masks)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
            writer.add_scalar(f'Loss/{variant}', loss.item(), global_step)
            global_step += 1
            
            if global_step % max(1, n_train // (5 * batch_size)) == 0:
                val_score = eval_net(net, val_loader, device)
                writer.add_scalar(f'Dice/{variant}', val_score, global_step)
                logging.info(f'[{variant}] Epoch {epoch+1}, Step {global_step},'
                             f' Val Dice: {val_score:.4f}')
                
                if val_score > best_dice:
                    best_dice = val_score
                    os.makedirs(dir_checkpoint, exist_ok=True)
                    torch.save(net.state_dict(),
                               f'{dir_checkpoint}/BEST_ssaf_{variant}.pth')
                    logging.info(f'Best saved: {variant} Dice={best_dice:.4f}')
        
        scheduler.step()
    
    writer.close()
    logging.info(f'[{variant}] Training done. Best Dice: {best_dice:.4f}')
    return best_dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True,
                        choices=list(MSHRNetAblation.VARIANTS.keys()))
    parser.add_argument('--train-img', required=True)
    parser.add_argument('--train-mask', required=True)
    parser.add_argument('--val-img', required=True)
    parser.add_argument('--val-mask', required=True)
    parser.add_argument('--in-ch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--loss-weights', nargs='+', default=['combined'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loss_weights = parse_loss_weights(args.loss_weights)
    
    train_ablation(
        variant=args.variant,
        device=device,
        train_img=args.train_img,
        train_mask=args.train_mask,
        val_img=args.val_img,
        val_mask=args.val_mask,
        in_channels=args.in_ch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_weights=loss_weights,
        warmup_epochs=args.warmup_epochs
    )