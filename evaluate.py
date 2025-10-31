"""
完整评估工具
计算：Dice, IoU, Precision, Recall, Boundary IoU
生成：混淆矩阵、PR曲线、可视化对比
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
from scipy.ndimage import binary_dilation, binary_erosion
import json

from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNet, MSHRNetOCR
from predict import read_image_any, normalize_image


def calculate_metrics(pred, gt):
    """
    计算全面的评估指标
    
    Args:
        pred: 预测二值图 (0/1)
        gt: 真值二值图 (0/1)
    
    Returns:
        dict: 包含各项指标的字典
    """
    pred_flat = pred.flatten().astype(bool)
    gt_flat = gt.flatten().astype(bool)
    
    # 混淆矩阵
    tp = np.sum(pred_flat & gt_flat)
    fp = np.sum(pred_flat & ~gt_flat)
    fn = np.sum(~pred_flat & gt_flat)
    tn = np.sum(~pred_flat & ~gt_flat)
    
    # 基础指标
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Dice系数
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    # IoU / Jaccard
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def calculate_boundary_metrics(pred, gt, dilation_radius=3):
    """
    计算边界指标
    
    Args:
        pred: 预测二值图
        gt: 真值二值图
        dilation_radius: 边界容忍半径（像素）
    """
    # 提取边界（边缘检测）
    pred_boundary = pred & ~binary_erosion(pred, iterations=1)
    gt_boundary = gt & ~binary_erosion(gt, iterations=1)
    
    if np.sum(gt_boundary) == 0:
        return {'boundary_iou': 0.0, 'boundary_f1': 0.0}
    
    # 膨胀容忍
    pred_dilated = binary_dilation(pred_boundary, iterations=dilation_radius)
    gt_dilated = binary_dilation(gt_boundary, iterations=dilation_radius)
    
    # 边界匹配
    pred_matched = np.sum(gt_dilated & pred_boundary)
    gt_matched = np.sum(pred_dilated & gt_boundary)
    
    boundary_precision = pred_matched / (np.sum(pred_boundary) + 1e-8)
    boundary_recall = gt_matched / (np.sum(gt_boundary) + 1e-8)
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-8)
    
    # 边界IoU
    intersection = np.sum((gt_dilated & pred_boundary) | (pred_dilated & gt_boundary))
    union = np.sum(pred_boundary) + np.sum(gt_boundary)
    boundary_iou = intersection / (union + 1e-8)
    
    return {
        'boundary_iou': float(boundary_iou),
        'boundary_f1': float(boundary_f1),
        'boundary_precision': float(boundary_precision),
        'boundary_recall': float(boundary_recall)
    }


def predict_single(net, img_path, device):
    """预测单张图像"""
    img_np = read_image_any(img_path)
    img_normalized = normalize_image(img_np)
    
    img = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        pred = net(img)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    
    return pred


def visualize_comparison(img, gt, pred, save_path):
    """可视化对比"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图（前3个波段）
    if img.shape[2] >= 3:
        vis_img = img[:, :, :3]
    else:
        vis_img = np.repeat(img[:, :, 0:1], 3, axis=2)
    
    # 归一化到0-1用于显示
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-8)
    
    axes[0, 0].imshow(vis_img)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred, cmap='gray')
    axes[0, 2].set_title('Prediction')
    axes[0, 2].axis('off')
    
    # 叠加显示
    overlay_gt = vis_img.copy()
    overlay_gt[gt > 0] = [0, 1, 0]  # 绿色
    axes[1, 0].imshow(overlay_gt)
    axes[1, 0].set_title('Ground Truth Overlay')
    axes[1, 0].axis('off')
    
    overlay_pred = vis_img.copy()
    overlay_pred[pred > 0] = [1, 0, 0]  # 红色
    axes[1, 1].imshow(overlay_pred)
    axes[1, 1].set_title('Prediction Overlay')
    axes[1, 1].axis('off')
    
    # 差异图：TP绿色，FP红色，FN蓝色
    diff = np.zeros((*gt.shape, 3))
    tp_mask = (pred > 0) & (gt > 0)
    fp_mask = (pred > 0) & (gt == 0)
    fn_mask = (pred == 0) & (gt > 0)
    
    diff[tp_mask] = [0, 1, 0]  # TP: 绿色
    diff[fp_mask] = [1, 0, 0]  # FP: 红色
    diff[fn_mask] = [0, 0, 1]  # FN: 蓝色
    
    axes[1, 2].imshow(diff)
    axes[1, 2].set_title('Difference (TP:Green, FP:Red, FN:Blue)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_dataset(net, img_dir, gt_dir, device, threshold=0.5, 
                     visualize=False, output_dir=None):
    """评估整个数据集"""
    
    # 查找图像文件
    img_files = sorted(glob(str(Path(img_dir) / '*.tif')) + 
                      glob(str(Path(img_dir) / '*.tiff')) +
                      glob(str(Path(img_dir) / '*.png')) +
                      glob(str(Path(img_dir) / '*.jpg')))
    
    if not img_files:
        logging.error(f'No images found in {img_dir}')
        return None
    
    logging.info(f'Found {len(img_files)} images to evaluate')
    
    all_metrics = []
    all_boundary_metrics = []
    
    # 准备输出目录
    if visualize and output_dir:
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 逐个评估
    for img_path in tqdm(img_files, desc='Evaluating'):
        img_name = Path(img_path).stem
        
        # 查找对应的GT
        gt_path = None
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            candidate = Path(gt_dir) / f'{img_name}{ext}'
            if candidate.exists():
                gt_path = candidate
                break
        
        if gt_path is None:
            logging.warning(f'GT not found for {img_name}, skipping')
            continue
        
        try:
            # 读取GT
            gt_np = read_image_any(str(gt_path))
            if gt_np.ndim == 3:
                gt_np = gt_np[:, :, 0]
            gt = (gt_np > 0).astype(np.uint8)
            
            # 预测
            pred_prob = predict_single(net, img_path, device)
            pred = (pred_prob > threshold).astype(np.uint8)
            
            # 计算指标
            metrics = calculate_metrics(pred, gt)
            boundary_metrics = calculate_boundary_metrics(pred, gt, dilation_radius=3)
            
            metrics.update(boundary_metrics)
            metrics['image'] = img_name
            all_metrics.append(metrics)
            all_boundary_metrics.append(boundary_metrics)
            
            # 可视化
            if visualize and output_dir:
                img_np = read_image_any(img_path)
                vis_path = vis_dir / f'{img_name}_comparison.png'
                visualize_comparison(img_np, gt, pred, vis_path)
        
        except Exception as e:
            logging.error(f'Error processing {img_name}: {e}')
            continue
    
    # 汇总统计
    if not all_metrics:
        logging.error('No successful evaluations!')
        return None
    
    summary = {
        'num_images': len(all_metrics),
        'mean_dice': np.mean([m['dice'] for m in all_metrics]),
        'std_dice': np.std([m['dice'] for m in all_metrics]),
        'mean_iou': np.mean([m['iou'] for m in all_metrics]),
        'std_iou': np.std([m['iou'] for m in all_metrics]),
        'mean_precision': np.mean([m['precision'] for m in all_metrics]),
        'mean_recall': np.mean([m['recall'] for m in all_metrics]),
        'mean_f1': np.mean([m['f1'] for m in all_metrics]),
        'mean_boundary_iou': np.mean([m['boundary_iou'] for m in all_boundary_metrics]),
        'mean_boundary_f1': np.mean([m['boundary_f1'] for m in all_boundary_metrics]),
    }
    
    # 计算总体混淆矩阵
    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    total_tn = sum(m['tn'] for m in all_metrics)
    
    summary['total_tp'] = total_tp
    summary['total_fp'] = total_fp
    summary['total_fn'] = total_fn
    summary['total_tn'] = total_tn
    
    # 全局Dice
    global_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
    summary['global_dice'] = global_dice
    
    return summary, all_metrics


def plot_metrics_distribution(all_metrics, output_dir):
    """绘制指标分布"""
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'boundary_iou']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        values = [m[metric] for m in all_metrics]
        axes[i].hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[i].axvline(np.mean(values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(values):.4f}')
        axes[i].set_xlabel(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'metrics_distribution.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate HRNet-OCR on test dataset')
    parser.add_argument('--model', '-m', default='checkpoints/BEST_unet_dat_4bands.pth', help='模型权重路径')
    parser.add_argument('--img-dir', default='/mnt/U/Dat_Seg/dat_4bands/test/images/', help='测试图像目录')
    parser.add_argument('--gt-dir', default='/mnt/U/Dat_Seg/dat_4bands/test/labels/', help='真值mask目录')
    parser.add_argument('--model-type', default='unet',
                       choices=['unet', 'unet_plusplus', 'pspnet', 'deeplabv3_plus', 'hrnet_ocr_w48', 'ms_hrnet_w48'])
    parser.add_argument('--in-ch', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', default='evaluation_results')
    parser.add_argument('--visualize', action='store_true', help='生成可视化对比图')
    parser.add_argument('--save-details', action='store_true', help='保存每张图的详细指标')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        net = HRNet(in_channels=args.in_ch, num_classes=1, base_channels=48)
    elif args.model_type == 'ms_hrnet_w48':
        net = MSHRNetOCR(in_channels=args.in_ch, num_classes=1, base_channels=48)
    else:
        raise ValueError(f'Unknown model architecture: {args.model}')
    
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device=device)
    net.eval()
    
    logging.info(f'Model loaded from {args.model}')
    
    # 评估
    summary, all_metrics = evaluate_dataset(
        net, args.img_dir, args.gt_dir, device,
        threshold=args.threshold,
        visualize=args.visualize,
        output_dir=args.output_dir
    )
    
    if summary is None:
        logging.error('Evaluation failed!')
        return
    
    # 打印结果
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'Number of images: {summary["num_images"]}')
    print(f'\nSegmentation Metrics:')
    print(f'  Mean Dice:      {summary["mean_dice"]:.4f} ± {summary["std_dice"]:.4f}')
    print(f'  Global Dice:    {summary["global_dice"]:.4f}')
    print(f'  Mean IoU:       {summary["mean_iou"]:.4f} ± {summary["std_iou"]:.4f}')
    print(f'  Mean Precision: {summary["mean_precision"]:.4f}')
    print(f'  Mean Recall:    {summary["mean_recall"]:.4f}')
    print(f'  Mean F1:        {summary["mean_f1"]:.4f}')
    print(f'\nBoundary Metrics:')
    print(f'  Boundary IoU:   {summary["mean_boundary_iou"]:.4f}')
    print(f'  Boundary F1:    {summary["mean_boundary_f1"]:.4f}')
    print(f'\nConfusion Matrix (Total):')
    print(f'  TP: {summary["total_tp"]:,}')
    print(f'  FP: {summary["total_fp"]:,}')
    print(f'  FN: {summary["total_fn"]:,}')
    print(f'  TN: {summary["total_tn"]:,}')
    print('='*60 + '\n')
    
    # 保存结果
    summary_path = output_dir / f'{args.model_type}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f'Summary saved to {summary_path}')
    
    if args.save_details:
        details_path = output_dir / 'detailed_metrics.json'
        with open(details_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logging.info(f'Detailed metrics saved to {details_path}')
    
    # 绘制分布图
    plot_metrics_distribution(all_metrics, output_dir)
    logging.info(f'Metrics distribution plot saved to {output_dir}')
    
    logging.info('Evaluation complete!')


if __name__ == '__main__':
    main()