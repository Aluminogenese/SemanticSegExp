import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.dataset import AdvancedDataset as DatasetClass

dir_checkpoint = 'checkpoints/'

class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss - 优化重叠度"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class BoundaryLoss(nn.Module):
    """边界损失 - 增强建筑物边缘"""
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        
    def forward(self, inputs, targets):
        # Sobel算子提取边界
        laplacian_kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=torch.float32, device=inputs.device
        ).view(1, 1, 3, 3)
        
        targets_boundary = F.conv2d(targets, laplacian_kernel, padding=1)
        targets_boundary = (targets_boundary > 0).float()
        
        inputs_prob = torch.sigmoid(inputs)
        inputs_boundary = F.conv2d(inputs_prob, laplacian_kernel, padding=1)
        
        boundary_loss = F.binary_cross_entropy_with_logits(
            inputs_boundary, targets_boundary, reduction='mean'
        )
        
        return boundary_loss


class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, weights={'bce': 1.0, 'dice': 1.0, 'focal': 0.5, 'boundary': 0.3}):
        super(CombinedLoss, self).__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.boundary = BoundaryLoss()

    def forward(self, inputs, targets):
        loss = 0
        if self.weights.get('bce', 0) > 0:
            loss += self.weights['bce'] * self.bce(inputs, targets)
        if self.weights.get('dice', 0) > 0:
            loss += self.weights['dice'] * self.dice(inputs, targets)
        if self.weights.get('focal', 0) > 0:
            loss += self.weights['focal'] * self.focal(inputs, targets)
        if self.weights.get('boundary', 0) > 0:
            loss += self.weights['boundary'] * self.boundary(inputs, targets)
        return loss


def train_net(net,
              net_name,
              device,
              dataset_name,
              train_img,
              train_mask,
              val_img,
              val_mask,
              epochs=100,
              batch_size=8,
              lr=0.001,
              save_cp=True,
              img_scale=1.0,
              mask_suffix='',
              warmup_epochs=5,
              use_deep_supervision=True):

    # 创建数据集
    train = DatasetClass(train_img, train_mask, img_scale, mask_suffix=mask_suffix, 
                            augment=True, crop_size=512)
    val = DatasetClass(val_img, val_mask, img_scale, mask_suffix=mask_suffix, 
                        augment=False, crop_size=512)
    
    n_train = len(train)
    n_val = len(val)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, 
                            num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, 
                          num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_{net_name}_{dataset_name}_LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Network:         {net_name}
        Dataset:         {dataset_name}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Deep Supervision: {use_deep_supervision}
    ''')

    # 组合损失函数
    criterion = CombinedLoss(weights={
        'bce': 1.0,
        'dice': 1.0,
        'focal': 0.5,
        'boundary': 0.3
    })
    
    # 辅助损失（用于深度监督）
    aux_criterion = nn.BCEWithLogitsLoss()

    # AdamW优化器（比RMSprop更适合HRNet）
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine退火 + 预热
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_dice = 0.0
    
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', disable=False) as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                
                assert imgs.shape[1] == net.n_channels, \
                    f'Network expects {net.n_channels} input channels, got {imgs.shape[1]}'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                # 前向传播（兼容无 aux_pred 的模型）
                outputs = net(imgs)
                aux_pred = None
                if isinstance(outputs, (tuple, list)):
                    masks_pred = outputs[0]
                    if len(outputs) > 1:
                        aux_pred = outputs[1]
                else:
                    masks_pred = outputs

                # 计算损失
                main_loss = criterion(masks_pred, true_masks)
                loss = main_loss
                if use_deep_supervision and (aux_pred is not None):
                    aux_loss = aux_criterion(aux_pred, true_masks)
                    loss = main_loss + 0.4 * aux_loss

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                
                # 定期验证
                if global_step % (n_train // (5 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device)
                    writer.add_scalar('Dice/test', val_score, global_step)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    
                    logging.info(f'Validation Dice: {val_score:.4f}')
                    
                    # 保存最佳模型
                    if val_score > best_dice:
                        best_dice = val_score
                        if save_cp:
                            try:
                                os.makedirs(dir_checkpoint, exist_ok=True)
                                torch.save(net.state_dict(),
                                         dir_checkpoint + f'BEST_{net_name}_{dataset_name}.pth')
                                logging.info(f'Best model saved! Dice: {best_dice:.4f}')
                            except OSError:
                                pass
                    
                    # 可视化
                    vis = imgs[:4]
                    if vis.shape[1] >= 3:
                        vis = vis[:, :3]
                    elif vis.shape[1] == 2:
                        vis = torch.cat([vis, vis[:, :1]], dim=1)
                    else:
                        vis = vis.repeat(1, 3, 1, 1)
                    writer.add_images('images', vis, global_step)
                    writer.add_images('masks/true', true_masks[:4], global_step)
                    writer.add_images('masks/pred', 
                                    torch.sigmoid(masks_pred[:4]) > 0.5, global_step)

        # 每个epoch结束后更新学习率
        scheduler.step()
        
        # 保存checkpoint
        if save_cp and (epoch + 1) % 50 == 0:
            try:
                os.makedirs(dir_checkpoint, exist_ok=True)
                torch.save(net.state_dict(),
                         dir_checkpoint + f'CP_{net_name}_{dataset_name}_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved!')
            except OSError:
                pass

    writer.close()
    logging.info(f'Training finished! Best Dice: {best_dice:.4f}')


def get_args():
    parser = argparse.ArgumentParser(description='Train HRNet-OCR for building segmentation')
    parser.add_argument('--train-img', dest='train_img', type=str, 
                       default='/mnt/U/Dat_Seg/dat_4bands/train/images/',
                       help='Directory of train images')
    parser.add_argument('--train-mask', dest='train_mask', type=str,
                       default='/mnt/U/Dat_Seg/dat_4bands/train/labels/',
                       help='Directory of train masks')
    parser.add_argument('--val-img', dest='val_img', type=str,
                       default='/mnt/U/Dat_Seg/dat_4bands/val/images/',
                       help='Directory of validation images')
    parser.add_argument('--val-mask', dest='val_mask', type=str,
                       default='/mnt/U/Dat_Seg/dat_4bands/val/labels/',
                       help='Directory of validation masks')
    parser.add_argument('-e', '--epochs', type=int, default=400,
                       help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('-f', '--load', type=str, default=False,
                       help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', type=float, default=1.0,
                       help='Downscaling factor of images')
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'unet_plusplus', 'pspnet', 'deeplabv3_plus', 'hrnet_ocr', 'ms_hrnet'],
                       help='Model architecture')
    parser.add_argument('--in-ch', type=int, default=4,
                       help='Number of input channels')
    parser.add_argument('--mask-suffix', type=str, default='',
                       help='Mask filename suffix')
    parser.add_argument('--dataset', type=str, default='4bands',
                       help='Dataset name')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--no-deep-supervision', action='store_true',
                       help='Disable deep supervision')
    
    return parser.parse_args()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 导入模型
    from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNetOCR, MSHRNetOCR
    
    if args.model == 'unet':
        net = UNet(in_channels=args.in_ch, num_classes=1)
    elif args.model == 'unet_plusplus':
        net = UNetPlusPlus(in_channels=args.in_ch, num_classes=1)
    elif args.model == 'pspnet':
        net = PSPNet(in_channels=args.in_ch, num_classes=1)
    elif args.model == 'deeplabv3_plus':
        net = DeepLabV3Plus(in_channels=args.in_ch, num_classes=1)
    elif args.model == 'hrnet_ocr':
        net = HRNetOCR(in_channels=args.in_ch, num_classes=1, base_channels=48)
    elif args.model == 'ms_hrnet':
        net = MSHRNetOCR(in_channels=args.in_ch, num_classes=1, base_channels=48)
    else:
        raise ValueError(f'Unknown model architecture: {args.model}')
    
    logging.info(f'Network: {args.model}')
    logging.info(f'Input channels: {net.n_channels}, Output classes: {net.n_classes}')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in net.parameters())
    logging.info(f'Total parameters: {total_params:,}')

    try:
        train_net(net=net,
                  net_name=args.model,
                  dataset_name=args.dataset,
                  train_img=args.train_img,
                  train_mask=args.train_mask,
                  val_img=args.val_img,
                  val_mask=args.val_mask,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.learning_rate,
                  device=device,
                  img_scale=args.scale,
                  mask_suffix=args.mask_suffix,
                  warmup_epochs=args.warmup_epochs,
                  use_deep_supervision=not args.no_deep_supervision)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)