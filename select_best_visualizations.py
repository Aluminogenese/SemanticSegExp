"""
论文效果图自动选择与排列工具

功能：
1. 从所有测试图像中自动选出4组最佳/最差/最有代表性的样本
2. 生成：原图 | GT | Model1 | Model2 | Model3 ... 的对比图
3. 支持多种选择策略：最佳、最差、差异最大、典型场景

使用方法：
python select_best_visualizations.py \
    --test-img /path/to/test/images/ \
    --test-mask /path/to/test/labels/ \
    --models-config eval_config.json \
    --output-dir paper_figures \
    --strategy best \
    --num-samples 4

手动指定图像（3种方式）：
1. 命令行直接指定：
   python select_best_visualizations.py \
       --test-img /path/to/test/images/ \
       --test-mask /path/to/test/labels/ \
       --models-config eval_config.json \
       --specify-images image1 image2 image3 image4

2. 从文件读取（推荐，用于论文固定图像）：
   python select_best_visualizations.py \
       --test-img /path/to/test/images/ \
       --test-mask /path/to/test/labels/ \
       --models-config eval_config.json \
       --specify-file selected_images.txt

3. 查看所有可用图像（用于选择）：
   python select_best_visualizations.py \
       --test-img /path/to/test/images/ \
       --test-mask /path/to/test/labels/ \
       --models-config eval_config.json \
       --strategy best \
       --num-samples 10  # 先看前10个最佳的
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json
from typing import List, Dict, Tuple
import seaborn as sns

# 导入现有的模型和工具
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNet, HRNetOCR, MSHRNetOCR
from predict import read_image_any, normalize_image


def calculate_metrics(pred, gt):
    """计算单张图像的评估指标"""
    pred_flat = pred.flatten().astype(bool)
    gt_flat = gt.flatten().astype(bool)
    
    tp = np.sum(pred_flat & gt_flat)
    fp = np.sum(pred_flat & ~gt_flat)
    fn = np.sum(~pred_flat & gt_flat)
    
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {'dice': dice, 'iou': iou}


def load_model(model_type, model_path, in_channels, device):
    """加载模型"""
    if model_type == 'unet':
        net = UNet(in_channels=in_channels, num_classes=1)
    elif model_type == 'unet_plusplus':
        net = UNetPlusPlus(in_channels=in_channels, num_classes=1)
    elif model_type == 'pspnet':
        net = PSPNet(in_channels=in_channels, num_classes=1)
    elif model_type == 'deeplabv3_plus':
        net = DeepLabV3Plus(in_channels=in_channels, num_classes=1)
    elif model_type == 'hrnet':
        net = HRNet(in_channels=in_channels, num_classes=1, base_channels=48)
    elif model_type == 'hrnet_ocr':
        net = HRNetOCR(in_channels=in_channels, num_classes=1, base_channels=48)
    elif model_type == 'ms_hrnet':
        net = MSHRNetOCR(in_channels=in_channels, num_classes=1, base_channels=48)
    elif model_type == 'ms_hrnet_v2':
        from models import MSHRNetV2
        net = MSHRNetV2(in_channels=in_channels, num_classes=1, base_channels=48)
    elif model_type == 'ms_hrnet_v2_min':
        from models import MSHRNetV2
        net = MSHRNetV2(in_channels=in_channels, num_classes=1, 
                        base_channels=48, use_minimal_ssaf=True)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.warning(f'Failed to load {model_path}: {e}')
        logging.warning('Attempting to load with strict=False...')
        net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    net.to(device=device)
    net.eval()
    
    return net


def predict_single(net, img_path, device, threshold=0.5, in_channels=4):
    """预测单张图像，自动适配3/4波段"""
    img_np = read_image_any(img_path)
    
    # 确保图像维度正确
    if img_np.ndim == 2:
        img_np = img_np[..., None]
    
    # 处理通道数不匹配的情况
    current_channels = img_np.shape[2] if img_np.ndim == 3 else 1
    
    if current_channels != in_channels:
        if current_channels > in_channels:
            # 裁剪多余通道（例如4波段裁到3波段）
            logging.warning(f'Image has {current_channels} channels, truncating to {in_channels}')
            img_np = img_np[:, :, :in_channels]
        else:
            # 扩展通道（例如灰度图扩展到3波段）
            logging.warning(f'Image has {current_channels} channels, expanding to {in_channels}')
            if current_channels == 1 and in_channels == 3:
                img_np = np.repeat(img_np, 3, axis=2)
            elif current_channels == 1 and in_channels == 4:
                img_np = np.repeat(img_np, 4, axis=2)
            elif current_channels == 3 and in_channels == 4:
                # RGB扩展到RGB+NIR，用平均值作为NIR
                nir = img_np.mean(axis=2, keepdims=True)
                img_np = np.concatenate([img_np, nir], axis=2)
            else:
                raise ValueError(f'Cannot convert {current_channels} channels to {in_channels}')
    
    img_normalized = normalize_image(img_np)
    
    img = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        pred = net(img)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.uint8)
    
    return img_np, pred_binary


def collect_all_predictions(img_files, gt_dir, models_info, device, threshold=0.5, in_channels=4):
    """收集所有图像的预测结果和指标"""
    all_results = []
    
    logging.info('Collecting predictions from all models...')
    
    for img_path in tqdm(img_files, desc='Processing images'):
        img_name = Path(img_path).stem
        
        # 查找GT
        gt_path = None
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            candidate = Path(gt_dir) / f'{img_name}{ext}'
            if candidate.exists():
                gt_path = candidate
                break
        
        if gt_path is None:
            logging.warning(f'GT not found for {img_name}')
            continue
        
        # 读取GT
        gt_np = read_image_any(str(gt_path))
        if gt_np.ndim == 3:
            gt_np = gt_np[:, :, 0]
        gt = (gt_np > 0).astype(np.uint8)
        
        # 读取原图
        img_np = read_image_any(img_path)
        
        # 预测所有模型
        result = {
            'image_path': img_path,
            'image_name': img_name,
            'image': img_np,
            'gt': gt,
            'predictions': {},
            'metrics': {}
        }
        
        for model_name, (net, _) in models_info.items():
            try:
                _, pred = predict_single(net, img_path, device, threshold, in_channels)
                metrics = calculate_metrics(pred, gt)
                
                result['predictions'][model_name] = pred
                result['metrics'][model_name] = metrics
            except Exception as e:
                logging.error(f'Error predicting {img_name} with {model_name}: {e}')
                continue
        
        # 只有当至少有一个模型成功预测时才添加结果
        if result['predictions']:
            # 计算平均Dice（用于排序）
            avg_dice = np.mean([m['dice'] for m in result['metrics'].values()])
            result['avg_dice'] = avg_dice
            
            # 计算模型间差异（标准差）
            dice_values = [m['dice'] for m in result['metrics'].values()]
            result['dice_std'] = np.std(dice_values)
            
            all_results.append(result)
    
    return all_results


def select_representative_samples(all_results, strategy='best', num_samples=4, 
                                specified_names=None):
    """选择代表性样本
    
    Args:
        all_results: 所有结果列表
        strategy: 选择策略
        num_samples: 样本数量
        specified_names: 指定的图像名称列表（不含扩展名）
    """
    
    # 如果指定了图像名称，直接选择这些图像
    if specified_names is not None:
        selected = []
        specified_set = set(specified_names)
        
        for result in all_results:
            if result['image_name'] in specified_set:
                selected.append(result)
        
        # 检查是否所有指定的图像都找到了
        found_names = {s['image_name'] for s in selected}
        missing = specified_set - found_names
        if missing:
            logging.warning(f'Could not find specified images: {missing}')
        
        if not selected:
            logging.error('None of the specified images were found!')
            logging.info('Available images:')
            for r in all_results[:10]:
                logging.info(f'  - {r["image_name"]}')
            if len(all_results) > 10:
                logging.info(f'  ... and {len(all_results) - 10} more')
            return []
        
        logging.info(f'Selected {len(selected)} specified images: {[s["image_name"] for s in selected]}')
        return selected
    
    # 否则使用自动策略
    if strategy == 'best':
        # 选择平均Dice最高的样本
        sorted_results = sorted(all_results, key=lambda x: x['avg_dice'], reverse=True)
        selected = sorted_results[:num_samples]
        logging.info(f'Selected {num_samples} samples with highest avg Dice')
    
    elif strategy == 'worst':
        # 选择平均Dice最低的样本
        sorted_results = sorted(all_results, key=lambda x: x['avg_dice'])
        selected = sorted_results[:num_samples]
        logging.info(f'Selected {num_samples} samples with lowest avg Dice')
    
    elif strategy == 'diverse':
        # 选择模型间差异最大的样本（最能体现模型差异）
        sorted_results = sorted(all_results, key=lambda x: x['dice_std'], reverse=True)
        selected = sorted_results[:num_samples]
        logging.info(f'Selected {num_samples} samples with highest model diversity')
    
    elif strategy == 'mixed':
        # 混合策略：最好的2个 + 差异最大的2个
        sorted_by_dice = sorted(all_results, key=lambda x: x['avg_dice'], reverse=True)
        sorted_by_std = sorted(all_results, key=lambda x: x['dice_std'], reverse=True)
        
        selected = []
        selected.extend(sorted_by_dice[:2])
        
        for sample in sorted_by_std:
            if sample not in selected and len(selected) < num_samples:
                selected.append(sample)
        
        logging.info(f'Selected {num_samples} samples using mixed strategy')
    
    else:
        # 默认：均匀采样
        step = len(all_results) // num_samples
        selected = [all_results[i * step] for i in range(num_samples)]
        logging.info(f'Selected {num_samples} samples using uniform sampling')
    
    return selected


def create_comparison_figure(samples, model_names, output_path, 
                            show_metrics=True, dpi=300, in_channels=4):
    """创建对比图，自动适配3/4波段"""
    
    num_samples = len(samples)
    num_cols = len(model_names) + 2  # 原图 + GT + 各模型
    
    # 设置图像大小
    fig = plt.figure(figsize=(num_cols * 3, num_samples * 3))
    gs = GridSpec(num_samples, num_cols, hspace=0.1, wspace=0.1)
    
    # 颜色方案
    cmap_pred = 'viridis'
    
    for i, sample in enumerate(samples):
        img = sample['image']
        gt = sample['gt']
        
        # 准备可视化的原图（智能适配通道数）
        if img.ndim == 2:
            # 灰度图
            vis_img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            # 单通道
            vis_img = np.repeat(img, 3, axis=2)
        elif img.shape[2] >= 3:
            # 3通道或4通道：使用前3个波段作为RGB
            vis_img = img[:, :, :3]
        elif img.shape[2] == 2:
            # 双通道：复制第一个通道
            vis_img = np.concatenate([img, img[:, :, 0:1]], axis=2)
        else:
            vis_img = img
        
        # 归一化到0-1用于显示
        vis_img = vis_img.astype(np.float32)
        if vis_img.max() > 1.0:
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-8)
        
        # 增强对比度（可选）
        vis_img = np.clip(vis_img, 0, 1)
        
        # 1. 原图
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(vis_img)
        ax.axis('off')
        if i == 0:
            ax.set_title(f'Input Image', fontsize=12, fontweight='bold')
        
        # 2. Ground Truth
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        
        # 3-N. 各模型预测
        for j, model_name in enumerate(model_names):
            ax = fig.add_subplot(gs[i, j + 2])
            pred = sample['predictions'][model_name]
            ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            if i == 0:
                ax.set_title(model_name, fontsize=12, fontweight='bold')
            
            # 显示指标
            if show_metrics:
                metrics = sample['metrics'][model_name]
                text = f"D:{metrics['dice']:.3f}\nI:{metrics['iou']:.3f}"
                ax.text(0.02, 0.98, text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', 
                                alpha=0.8, edgecolor='none'))
        
        # 在行首显示图像编号
        # fig.text(0.01, 1 - (i + 0.5) / num_samples, f'({chr(65+i)})', 
        #         fontsize=14, fontweight='bold', va='center')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    logging.info(f'Comparison figure saved to {output_path}')


def create_detailed_comparison(samples, model_names, output_dir, dpi=300, in_channels=4):
    """创建详细对比图（每个样本一张独立的图），自动适配3/4波段"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, sample in enumerate(samples):
        img = sample['image']
        gt = sample['gt']
        img_name = sample['image_name']
        
        # 准备原图（智能适配通道数）
        if img.ndim == 2:
            vis_img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            vis_img = np.repeat(img, 3, axis=2)
        elif img.shape[2] >= 3:
            vis_img = img[:, :, :3]
        elif img.shape[2] == 2:
            vis_img = np.concatenate([img, img[:, :, 0:1]], axis=2)
        else:
            vis_img = img
        
        # 归一化
        vis_img = vis_img.astype(np.float32)
        if vis_img.max() > 1.0:
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-8)
        vis_img = np.clip(vis_img, 0, 1)
        
        num_models = len(model_names)
        
        # 创建子图布局：2行，每行显示 原图+GT+部分模型
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 4, hspace=0.3, wspace=0.3)
        
        # 第一行：原图、GT、前两个模型
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(vis_img)
        band_info = f'{in_channels}-band' if in_channels > 1 else 'Grayscale'
        ax1.set_title(f'Input Image\n({band_info})', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        for j in range(min(2, num_models)):
            ax = fig.add_subplot(gs[0, j + 2])
            model_name = model_names[j]
            pred = sample['predictions'][model_name]
            metrics = sample['metrics'][model_name]
            
            ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{model_name}\nDice: {metrics["dice"]:.4f}', 
                        fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # 第二行：差异图 + 剩余模型
        # 差异图：TP=绿, FP=红, FN=蓝
        ax_diff = fig.add_subplot(gs[1, 0])
        
        # 使用最佳模型的预测作为差异图基准
        best_model = max(model_names, key=lambda m: sample['metrics'][m]['dice'])
        best_pred = sample['predictions'][best_model]
        
        diff = np.zeros((*gt.shape, 3))
        tp_mask = (best_pred > 0) & (gt > 0)
        fp_mask = (best_pred > 0) & (gt == 0)
        fn_mask = (best_pred == 0) & (gt > 0)
        
        diff[tp_mask] = [0, 1, 0]   # TP: 绿色
        diff[fp_mask] = [1, 0, 0]   # FP: 红色
        diff[fn_mask] = [0, 0, 1]   # FN: 蓝色
        
        ax_diff.imshow(diff)
        ax_diff.set_title(f'Error Map ({best_model})\nTP:Green, FP:Red, FN:Blue', 
                         fontsize=11, fontweight='bold')
        ax_diff.axis('off')
        
        # 剩余模型
        for j in range(2, min(4, num_models)):
            ax = fig.add_subplot(gs[1, j - 1])
            model_name = model_names[j]
            pred = sample['predictions'][model_name]
            metrics = sample['metrics'][model_name]
            
            ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{model_name}\nDice: {metrics["dice"]:.4f}', 
                        fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # 如果有更多模型，显示指标表格
        if num_models > 4:
            ax_table = fig.add_subplot(gs[1, 3])
            ax_table.axis('off')
            
            table_data = []
            for model_name in model_names[4:]:
                metrics = sample['metrics'][model_name]
                table_data.append([model_name, f"{metrics['dice']:.4f}", 
                                  f"{metrics['iou']:.4f}"])
            
            table = ax_table.table(cellText=table_data,
                                  colLabels=['Model', 'Dice', 'IoU'],
                                  cellLoc='center',
                                  loc='center',
                                  bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
        
        plt.suptitle(f'Sample {chr(65+idx)}: {img_name}\n'
                    f'Avg Dice: {sample["avg_dice"]:.4f}', 
                    fontsize=14, fontweight='bold')
        
        output_path = output_dir / f'detailed_sample_{chr(65+idx)}_{img_name}.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logging.info(f'Detailed figure saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Automatically select and visualize best predictions for paper'
    )
    
    # 数据参数
    parser.add_argument('--test-img', required=True, help='测试图像目录')
    parser.add_argument('--test-mask', required=True, help='测试mask目录')
    parser.add_argument('--models-config', required=True, help='模型配置JSON文件')
    parser.add_argument('--in-ch', type=int, default=4, help='输入通道数')
    
    # 选择策略
    parser.add_argument('--strategy', default='best',
                       choices=['best', 'worst', 'diverse', 'mixed', 'uniform'],
                       help='样本选择策略')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='选择的样本数量')
    parser.add_argument('--specify-images', nargs='+', default=None,
                       help='手动指定要可视化的图像名称（不含扩展名），例如: --specify-images img1 img2 img3 img4')
    parser.add_argument('--specify-file', type=str, default=None,
                       help='从文件读取要可视化的图像名称列表（每行一个，不含扩展名）')
    
    # 输出参数
    parser.add_argument('--output-dir', default='paper_figures',
                       help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='二值化阈值')
    parser.add_argument('--dpi', type=int, default=300,
                       help='输出图像DPI')
    parser.add_argument('--detailed', action='store_true',
                       help='生成详细对比图（每个样本单独一张）')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型配置
    with open(args.models_config, 'r') as f:
        config = json.load(f)
    
    models_config = config['models']
    
    # 加载设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # 加载所有模型
    models_info = {}
    for model_cfg in models_config:
        name = model_cfg['name']
        model_type = model_cfg['type']
        model_path = model_cfg['path']
        
        logging.info(f'Loading model: {name}')
        net = load_model(model_type, model_path, args.in_ch, device)
        models_info[name] = (net, model_type)
    
    model_names = list(models_info.keys())
    
    # 查找测试图像
    img_files = sorted(glob(str(Path(args.test_img) / '*.tif')) + 
                      glob(str(Path(args.test_img) / '*.tiff')) +
                      glob(str(Path(args.test_img) / '*.png')) +
                      glob(str(Path(args.test_img) / '*.jpg')))
    
    if not img_files:
        logging.error(f'No images found in {args.test_img}')
        return
    
    logging.info(f'Found {len(img_files)} test images')
    
    # 收集所有预测结果
    all_results = collect_all_predictions(
        img_files, args.test_mask, models_info, device, args.threshold, args.in_ch
    )
    
    if not all_results:
        logging.error('No valid results collected!')
        return
    
    # 处理手动指定的图像
    specified_names = None
    if args.specify_images:
        specified_names = args.specify_images
        logging.info(f'Using specified images: {specified_names}')
    elif args.specify_file:
        # 从文件读取图像名称
        with open(args.specify_file, 'r') as f:
            specified_names = [line.strip() for line in f if line.strip()]
        logging.info(f'Loaded {len(specified_names)} image names from {args.specify_file}')
        logging.info(f'Images: {specified_names}')
    
    # 选择代表性样本
    selected_samples = select_representative_samples(
        all_results, args.strategy, args.num_samples, specified_names
    )
    
    # 保存选择结果的元数据
    metadata = []
    for i, sample in enumerate(selected_samples):
        meta = {
            'index': i,
            'image_name': sample['image_name'],
            'avg_dice': float(sample['avg_dice']),
            'dice_std': float(sample['dice_std']),
            'metrics': {name: {k: float(v) for k, v in m.items()} 
                       for name, m in sample['metrics'].items()}
        }
        metadata.append(meta)
    
    metadata_path = output_dir / 'selected_samples_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f'Metadata saved to {metadata_path}')
    
    # 生成对比图
    comparison_path = output_dir / f'comparison_{args.strategy}_{args.num_samples}samples-4bands.png'
    create_comparison_figure(
        selected_samples, model_names, comparison_path, 
        show_metrics=False, dpi=args.dpi, in_channels=args.in_ch
    )
    
    # 生成详细对比图（可选）
    if args.detailed:
        detailed_dir = output_dir / 'detailed'
        create_detailed_comparison(
            selected_samples, model_names, detailed_dir, dpi=args.dpi, in_channels=args.in_ch
        )
    
    # 打印选择结果总结
    print('\n' + '='*80)
    print(f'SELECTED SAMPLES SUMMARY ({args.strategy.upper()} STRATEGY)')
    print('='*80)
    for i, sample in enumerate(selected_samples):
        print(f'\nSample {chr(65+i)}: {sample["image_name"]}')
        print(f'  Average Dice: {sample["avg_dice"]:.4f}')
        print(f'  Dice Std Dev: {sample["dice_std"]:.4f}')
        print('  Model Performance:')
        for name in model_names:
            metrics = sample['metrics'][name]
            print(f'    {name:20s}: Dice={metrics["dice"]:.4f}, IoU={metrics["iou"]:.4f}')
    print('='*80)
    
    logging.info('\nVisualization complete!')


if __name__ == '__main__':
    main()