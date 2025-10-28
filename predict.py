"""
HRNet-OCR 预测脚本
支持：单图/批量推理、TTA、大图分块处理
"""

import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from glob import glob
import json

from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNetOCR, MSHRNetOCR


def read_image_any(path):
    """读取任意格式图像"""
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read()
            if hasattr(arr, 'filled'):
                arr = arr.filled(0)
            if arr.ndim == 3:
                arr = np.transpose(arr, (1, 2, 0))
            return arr
    except:
        pass
    
    try:
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        return arr
    except:
        pass
    
    img = Image.open(path)
    return np.array(img)


def normalize_image(img, stats_file='stats.json'):
    """归一化图像"""
    img = img.astype(np.float32)
    
    if img.ndim == 2:
        img = img[..., None]
    
    # 加载统计信息
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        p_low = np.array(stats['p_low'], dtype=np.float32)
        p_high = np.array(stats['p_high'], dtype=np.float32)
        
        img = np.clip(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-8)
    else:
        # 简单归一化
        if img.max() > 255:
            channel_max = np.array([4553.0, 4166.0, 4489.0, 9142.0], dtype=np.float32)[:img.shape[2]]
            img = img / channel_max
        else:
            img = img / 255.0
    
    return img


def predict_single(net, img_np, device, tta=False):
    """单张图像预测"""
    net.eval()
    
    # 归一化
    img_normalized = normalize_image(img_np)
    
    # HWC -> CHW -> BCHW
    img = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        if tta:
            # 8种TTA变换
            predictions = []
            
            for flip in [False, True]:
                for k in range(4):  # 4个90度旋转
                    img_aug = img.clone()
                    
                    if flip:
                        img_aug = torch.flip(img_aug, dims=[3])
                    
                    if k > 0:
                        img_aug = torch.rot90(img_aug, k, dims=[2, 3])
                    
                    # 预测
                    pred = net(img_aug)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    pred = torch.sigmoid(pred)
                    
                    # 反向变换
                    if k > 0:
                        pred = torch.rot90(pred, 4-k, dims=[2, 3])
                    if flip:
                        pred = torch.flip(pred, dims=[3])
                    
                    predictions.append(pred)
            
            # 平均融合
            pred = torch.stack(predictions).mean(0)
        else:
            pred = net(img)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.sigmoid(pred)
    
    pred = pred.squeeze().cpu().numpy()
    return pred


def predict_large_image(net, img_np, device, tile_size=1024, overlap=128, tta=False):
    """大图分块预测"""
    h, w = img_np.shape[:2]
    
    if h <= tile_size and w <= tile_size:
        return predict_single(net, img_np, device, tta)
    
    stride = tile_size - overlap
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    
    # 创建权重图（边缘羽化）
    tile_weight = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap > 0:
        fade = np.linspace(0, 1, overlap)
        tile_weight[:overlap, :] *= fade[:, None]
        tile_weight[-overlap:, :] *= fade[::-1, None]
        tile_weight[:, :overlap] *= fade[None, :]
        tile_weight[:, -overlap:] *= fade[None, ::-1]
    
    logging.info(f'Processing {h}x{w} image with {tile_size}x{tile_size} tiles...')
    
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            i_end = min(i + tile_size, h)
            j_end = min(j + tile_size, w)
            
            # 提取tile
            tile = img_np[i:i_end, j:j_end]
            
            # Padding到tile_size
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            
            # 预测
            pred_tile = predict_single(net, tile, device, tta=False)
            
            # 裁剪padding
            pred_tile = pred_tile[:i_end-i, :j_end-j]
            w_tile = tile_weight[:i_end-i, :j_end-j]
            
            # 累加
            output[i:i_end, j:j_end] += pred_tile * w_tile
            weight[i:i_end, j:j_end] += w_tile
    
    return output / (weight + 1e-8)


def main():
    parser = argparse.ArgumentParser(description='Predict with HRNet-OCR')
    parser.add_argument('--model', '-m', default='checkpoints/BEST_hrnet_w48_4bands.pth', help='模型权重路径')
    parser.add_argument('--input', '-i', nargs='+', default=['/mnt/U/Dat_Seg/4bands/test/images/*.tif'], help='输入图像路径')
    parser.add_argument('--output', '-o',default='predict_results', help='输出目录或文件路径')
    parser.add_argument('--model-type', default='unet',
                       choices=['unet', 'unet_plusplus', 'pspnet', 'deeplabv3_plus', 'hrnet_ocr_w48', 'ms_hrnet_w48'],
                       help='模型类型')
    parser.add_argument('--in-ch', type=int, default=4, help='输入通道数')
    parser.add_argument('--tta', action='store_true', help='启用TTA（8x变换）')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--tile-size', type=int, default=1024, help='大图分块尺寸')
    parser.add_argument('--overlap', type=int, default=128, help='分块重叠像素')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    if args.model_type == 'unet':
        net = UNet(in_channels=args.in_ch, num_classes=1)
    elif args.model_type == 'unet_plusplus':
        net = UNetPlusPlus(in_channels=args.in_ch, num_classes=1)
    elif args.model_type == 'pspnet':
        net = PSPNet(in_channels=args.in_ch, num_classes=1)
    elif args.model_type == 'deeplabv3_plus':
        net = DeepLabV3Plus(in_channels=args.in_ch, num_classes=1)
    elif args.model_type == 'hrnet_ocr_w48':
        net = HRNetOCR(in_channels=args.in_ch, num_classes=1, base_channels=48)
    elif args.model_type == 'ms_hrnet_w48':
        net = MSHRNetOCR(in_channels=args.in_ch, num_classes=1, base_channels=48)
    else:
        raise ValueError(f'Unknown model architecture: {args.model}')
    
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device=device)
    net.eval()
    
    logging.info(f'Model loaded from {args.model}')
    logging.info(f'TTA: {args.tta}, Tile size: {args.tile_size}, Overlap: {args.overlap}')
    
    # 处理输入
    # 统一规范为列表，兼容默认值为字符串的情况
    input_patterns = args.input if isinstance(args.input, (list, tuple)) else [args.input]

    input_files = []
    for pattern in input_patterns:
        # 展开 ~ 和环境变量
        pattern = os.path.expanduser(os.path.expandvars(pattern))

        # 目录：收集目录下所有文件（按需可改为限定后缀）
        if os.path.isdir(pattern):
            input_files.extend(sorted(glob(os.path.join(pattern, '*'))))
        # 通配：支持 *, ?, [
        elif ('*' in pattern) or ('?' in pattern) or ('[' in pattern):
            input_files.extend(sorted(glob(pattern)))
        else:
            input_files.append(pattern)
    
    if not input_files:
        logging.error('No input files found!')
        return
    
    logging.info(f'Found {len(input_files)} images to process')
    
    # 准备输出
    if args.output:
        output_path = Path(args.output)
        if len(input_files) > 1 or args.batch:
            output_path.mkdir(parents=True, exist_ok=True)
            is_dir = True
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            is_dir = False
    else:
        is_dir = False
    
    # 逐个处理
    for i, img_path in enumerate(input_files):
        logging.info(f'[{i+1}/{len(input_files)}] Processing {img_path}...')
        
        try:
            # 读取图像
            img_np = read_image_any(img_path)
            orig_h, orig_w = img_np.shape[:2]
            logging.info(f'  Image size: {orig_w}x{orig_h}')
            
            # 预测
            if orig_h > args.tile_size or orig_w > args.tile_size:
                pred = predict_large_image(net, img_np, device, 
                                         args.tile_size, args.overlap, args.tta)
            else:
                pred = predict_single(net, img_np, device, args.tta)
            
            # 二值化
            pred_binary = (pred > args.threshold).astype(np.uint8) * 255
            
            # 保存
            if args.output:
                if is_dir:
                    out_file = output_path / f'{Path(img_path).stem}_pred.png'
                else:
                    out_file = output_path
                
                Image.fromarray(pred_binary).save(out_file)
                logging.info(f'  Saved to {out_file}')
            else:
                logging.info(f'  Prediction done (not saved)')
        
        except Exception as e:
            logging.error(f'  Error processing {img_path}: {e}')
            continue
    
    logging.info('Done!')


if __name__ == '__main__':
    main()