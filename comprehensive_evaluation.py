"""
完整的模型评估脚本 - 适用于论文写作

功能:
1. 计算全面的评估指标 (Dice, IoU, Precision, Recall, F1, Boundary IoU)
2. 生成混淆矩阵和统计表格
3. 可视化对比图和误差分析
4. 输出LaTeX表格
5. 支持多模型对比评估
6. 计算统计显著性检验

使用示例:
python comprehensive_evaluation.py --model checkpoints/BEST_ms_hrnet_dat_4bands.pth \
                                   --model-type ms_hrnet \
                                   --test-img /path/to/test/images \
                                   --test-mask /path/to/test/labels \
                                   --output-dir evaluation_results
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
import pandas as pd
from scipy import stats
import seaborn as sns

from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNet, HRNetOCR, MSHRNetOCR
from predict import read_image_any, normalize_image


class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self, net, device, threshold=0.5):
        self.net = net
        self.device = device
        self.threshold = threshold
        self.results = []
        
    def calculate_metrics(self, pred, gt):
        """计算全面的评估指标"""
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
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp + 1e-8)
        
        return {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'specificity': float(specificity),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    def calculate_boundary_metrics(self, pred, gt, dilation_radius=3):
        """计算边界指标"""
        # 提取边界
        pred_boundary = pred & ~binary_erosion(pred, iterations=1)
        gt_boundary = gt & ~binary_erosion(gt, iterations=1)
        
        if np.sum(gt_boundary) == 0:
            return {
                'boundary_iou': 0.0, 
                'boundary_f1': 0.0,
                'boundary_precision': 0.0,
                'boundary_recall': 0.0
            }
        
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
    
    def predict_single(self, img_path):
        """预测单张图像"""
        img_np = read_image_any(img_path)
        img_normalized = normalize_image(img_np)
        
        img = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            pred = self.net(img)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        
        return pred, img_np
    
    def evaluate_image(self, img_path, gt_path):
        """评估单张图像"""
        # 预测
        pred_prob, img_np = self.predict_single(img_path)
        pred = (pred_prob > self.threshold).astype(np.uint8)
        
        # 读取GT
        gt_np = read_image_any(str(gt_path))
        if gt_np.ndim == 3:
            gt_np = gt_np[:, :, 0]
        gt = (gt_np > 0).astype(np.uint8)
        
        # 计算指标
        metrics = self.calculate_metrics(pred, gt)
        boundary_metrics = self.calculate_boundary_metrics(pred, gt, dilation_radius=3)
        
        metrics.update(boundary_metrics)
        metrics['image'] = Path(img_path).stem
        
        return metrics, pred, gt, img_np
    
    def evaluate_dataset(self, img_dir, gt_dir, visualize=False, output_dir=None):
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
                metrics, pred, gt, img_np = self.evaluate_image(img_path, gt_path)
                self.results.append(metrics)
                
                # 可视化
                if visualize and output_dir:
                    vis_path = vis_dir / f'{img_name}_comparison.png'
                    self.visualize_comparison(img_np, gt, pred, vis_path, metrics)
            
            except Exception as e:
                logging.error(f'Error processing {img_name}: {e}')
                continue
        
        return self.results
    
    def visualize_comparison(self, img, gt, pred, save_path, metrics):
        """可视化对比（增强版）"""
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # 原图（前3个波段）
        if img.shape[2] >= 3:
            vis_img = img[:, :, :3]
        else:
            vis_img = np.repeat(img[:, :, 0:1], 3, axis=2)
        
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-8)
        
        # 1. 原图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(vis_img)
        ax1.set_title('Input Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Ground Truth
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gt, cmap='gray')
        ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Prediction
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred, cmap='gray')
        ax3.set_title('Prediction', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. GT叠加
        ax4 = fig.add_subplot(gs[0, 3])
        overlay_gt = vis_img.copy()
        overlay_gt[gt > 0] = [0, 1, 0]  # 绿色
        ax4.imshow(overlay_gt)
        ax4.set_title('GT Overlay', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. Pred叠加
        ax5 = fig.add_subplot(gs[1, 0])
        overlay_pred = vis_img.copy()
        overlay_pred[pred > 0] = [1, 0, 0]  # 红色
        ax5.imshow(overlay_pred)
        ax5.set_title('Prediction Overlay', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # 6. 差异图：TP绿色，FP红色，FN蓝色
        ax6 = fig.add_subplot(gs[1, 1])
        diff = np.zeros((*gt.shape, 3))
        tp_mask = (pred > 0) & (gt > 0)
        fp_mask = (pred > 0) & (gt == 0)
        fn_mask = (pred == 0) & (gt > 0)
        
        diff[tp_mask] = [0, 1, 0]  # TP: 绿色
        diff[fp_mask] = [1, 0, 0]  # FP: 红色
        diff[fn_mask] = [0, 0, 1]  # FN: 蓝色
        
        ax6.imshow(diff)
        ax6.set_title('Error Map\n(TP:Green, FP:Red, FN:Blue)', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # 7. 边界对比
        ax7 = fig.add_subplot(gs[1, 2])
        pred_boundary = pred & ~binary_erosion(pred, iterations=1)
        gt_boundary = gt & ~binary_erosion(gt, iterations=1)
        boundary_overlay = vis_img.copy()
        boundary_overlay[gt_boundary > 0] = [0, 1, 0]  # GT边界：绿色
        boundary_overlay[pred_boundary > 0] = [1, 0, 0]  # Pred边界：红色
        ax7.imshow(boundary_overlay)
        ax7.set_title('Boundary Comparison\n(GT:Green, Pred:Red)', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # 8. 指标文本
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')
        metrics_text = f"""
Segmentation Metrics:
  Dice:      {metrics['dice']:.4f}
  IoU:       {metrics['iou']:.4f}
  Precision: {metrics['precision']:.4f}
  Recall:    {metrics['recall']:.4f}
  F1-Score:  {metrics['f1']:.4f}
  Accuracy:  {metrics['accuracy']:.4f}

Boundary Metrics:
  B-IoU:     {metrics['boundary_iou']:.4f}
  B-F1:      {metrics['boundary_f1']:.4f}

Confusion Matrix:
  TP: {metrics['tp']:,}
  FP: {metrics['fp']:,}
  FN: {metrics['fn']:,}
  TN: {metrics['tn']:,}
        """
        ax8.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_summary(self):
        """生成汇总统计"""
        if not self.results:
            logging.error('No results to summarize!')
            return None
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'num_images': len(self.results),
            'mean_dice': df['dice'].mean(),
            'std_dice': df['dice'].std(),
            'median_dice': df['dice'].median(),
            'min_dice': df['dice'].min(),
            'max_dice': df['dice'].max(),
            
            'mean_iou': df['iou'].mean(),
            'std_iou': df['iou'].std(),
            'median_iou': df['iou'].median(),
            
            'mean_precision': df['precision'].mean(),
            'std_precision': df['precision'].std(),
            
            'mean_recall': df['recall'].mean(),
            'std_recall': df['recall'].std(),
            
            'mean_f1': df['f1'].mean(),
            'std_f1': df['f1'].std(),
            
            'mean_accuracy': df['accuracy'].mean(),
            'std_accuracy': df['accuracy'].std(),
            
            'mean_specificity': df['specificity'].mean(),
            'std_specificity': df['specificity'].std(),
            
            'mean_boundary_iou': df['boundary_iou'].mean(),
            'std_boundary_iou': df['boundary_iou'].std(),
            
            'mean_boundary_f1': df['boundary_f1'].mean(),
            'std_boundary_f1': df['boundary_f1'].std(),
        }
        
        # 计算总体混淆矩阵
        total_tp = df['tp'].sum()
        total_fp = df['fp'].sum()
        total_fn = df['fn'].sum()
        total_tn = df['tn'].sum()
        
        summary['total_tp'] = int(total_tp)
        summary['total_fp'] = int(total_fp)
        summary['total_fn'] = int(total_fn)
        summary['total_tn'] = int(total_tn)
        
        # 全局Dice
        global_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
        summary['global_dice'] = float(global_dice)
        
        # 全局IoU
        global_iou = total_tp / (total_tp + total_fp + total_fn)
        summary['global_iou'] = float(global_iou)
        
        return summary
    
    def plot_metrics_distribution(self, output_dir):
        """绘制指标分布"""
        df = pd.DataFrame(self.results)
        
        metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'f1', 'boundary_iou']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            values = df[metric]
            
            # 直方图
            axes[i].hist(values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            
            # 统计线
            mean_val = values.mean()
            median_val = values.median()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.4f}')
            axes[i].axvline(median_val, color='green', linestyle='-.', linewidth=2,
                          label=f'Median: {median_val:.4f}')
            
            axes[i].set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution', 
                            fontsize=14, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'metrics_distribution.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, output_dir):
        """绘制混淆矩阵"""
        summary = self.generate_summary()
        
        cm = np.array([
            [summary['total_tn'], summary['total_fp']],
            [summary['total_fn'], summary['total_tp']]
        ])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'},
                   ax=ax, annot_kws={'size': 16})
        
        ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'confusion_matrix.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def generate_latex_table(self, model_name):
        """生成LaTeX表格"""
        summary = self.generate_summary()
        
        latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Metrics of {model_name} on Test Dataset}}
\\label{{tab:results_{model_name.lower().replace(' ', '_')}}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Mean ± Std}} & \\textbf{{Global}} \\\\
\\midrule
Dice Coefficient & {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f} & {summary['global_dice']:.4f} \\\\
IoU (Jaccard) & {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f} & {summary['global_iou']:.4f} \\\\
Precision & {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f} & - \\\\
Recall & {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f} & - \\\\
F1-Score & {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f} & - \\\\
Accuracy & {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f} & - \\\\
Specificity & {summary['mean_specificity']:.4f} ± {summary['std_specificity']:.4f} & - \\\\
\\midrule
Boundary IoU & {summary['mean_boundary_iou']:.4f} ± {summary['std_boundary_iou']:.4f} & - \\\\
Boundary F1 & {summary['mean_boundary_f1']:.4f} ± {summary['std_boundary_f1']:.4f} & - \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
        """
        
        return latex


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation for Paper')
    
    # 模型参数
    parser.add_argument('--model', '-m', required=True, help='模型权重路径')
    parser.add_argument('--model-type', required=True,
                       choices=['unet', 'unet_plusplus', 'pspnet', 'deeplabv3_plus', 
                               'hrnet', 'hrnet_ocr', 'ms_hrnet',
                               'ms_hrnet_no_ssaf', 'ms_hrnet_no_msbr', 'ms_hrnet_v2', 'ms_hrnet_v2_min'],
                       help='模型类型')
    parser.add_argument('--model-name', default=None, help='模型名称（用于报告）')
    
    # 数据参数
    parser.add_argument('--test-img', required=True, help='测试图像目录')
    parser.add_argument('--test-mask', required=True, help='测试mask目录')
    parser.add_argument('--in-ch', type=int, default=4, help='输入通道数')
    
    # 评估参数
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--output-dir', default='evaluation_results', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化对比图')
    parser.add_argument('--num-vis', type=int, default=20, help='可视化图像数量（0=全部）')
    
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
    elif args.model_type == 'hrnet':
        net = HRNet(in_channels=args.in_ch, num_classes=1, base_channels=48)
    elif args.model_type == 'hrnet_ocr':
        net = HRNetOCR(in_channels=args.in_ch, num_classes=1, base_channels=48)
    elif args.model_type == 'ms_hrnet':
        net = MSHRNetOCR(in_channels=args.in_ch, num_classes=1, base_channels=48)
    elif args.model_type == 'ms_hrnet_v2':
        from models import MSHRNetV2
        net = MSHRNetV2(in_channels=args.in_ch, num_classes=1, base_channels=48)
    elif args.model_type == 'ms_hrnet_v2_min':
        from models import MSHRNetV2
        net = MSHRNetV2(in_channels=args.in_ch, num_classes=1, 
                        base_channels=48, use_minimal_ssaf=True)

    else:
        raise ValueError(f'Unknown model architecture: {args.model_type}')
    
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device=device)
    net.eval()
    
    logging.info(f'Model loaded from {args.model}')
    
    # 模型名称
    model_name = args.model_name or args.model_type.upper()
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator(net, device, threshold=args.threshold)
    
    # 评估数据集
    logging.info('Starting evaluation...')
    results = evaluator.evaluate_dataset(
        args.test_img, 
        args.test_mask,
        visualize=args.visualize,
        output_dir=output_dir
    )
    
    if results is None:
        logging.error('Evaluation failed!')
        return
    
    # 生成汇总统计
    summary = evaluator.generate_summary()
    
    # 打印结果
    print('\n' + '='*80)
    print(f'COMPREHENSIVE EVALUATION RESULTS - {model_name}')
    print('='*80)
    print(f'\nDataset: {args.test_img}')
    print(f'Number of images: {summary["num_images"]}')
    print(f'Model: {args.model}')
    print(f'Threshold: {args.threshold}')
    
    print(f'\n{"Segmentation Metrics":=^80}')
    print(f'  Dice Coefficient:    {summary["mean_dice"]:.4f} ± {summary["std_dice"]:.4f}  (Median: {summary["median_dice"]:.4f})')
    print(f'  Global Dice:         {summary["global_dice"]:.4f}')
    print(f'  IoU (Jaccard):       {summary["mean_iou"]:.4f} ± {summary["std_iou"]:.4f}  (Median: {summary["median_iou"]:.4f})')
    print(f'  Global IoU:          {summary["global_iou"]:.4f}')
    print(f'  Precision:           {summary["mean_precision"]:.4f} ± {summary["std_precision"]:.4f}')
    print(f'  Recall:              {summary["mean_recall"]:.4f} ± {summary["std_recall"]:.4f}')
    print(f'  F1-Score:            {summary["mean_f1"]:.4f} ± {summary["std_f1"]:.4f}')
    print(f'  Accuracy:            {summary["mean_accuracy"]:.4f} ± {summary["std_accuracy"]:.4f}')
    print(f'  Specificity:         {summary["mean_specificity"]:.4f} ± {summary["std_specificity"]:.4f}')
    
    print(f'\n{"Boundary Metrics":=^80}')
    print(f'  Boundary IoU:        {summary["mean_boundary_iou"]:.4f} ± {summary["std_boundary_iou"]:.4f}')
    print(f'  Boundary F1:         {summary["mean_boundary_f1"]:.4f} ± {summary["std_boundary_f1"]:.4f}')
    
    print(f'\n{"Confusion Matrix (Total)":=^80}')
    print(f'  True Positives (TP):  {summary["total_tp"]:,}')
    print(f'  False Positives (FP): {summary["total_fp"]:,}')
    print(f'  False Negatives (FN): {summary["total_fn"]:,}')
    print(f'  True Negatives (TN):  {summary["total_tn"]:,}')
    
    print('='*80 + '\n')
    
    # 保存结果
    # 1. JSON格式
    summary_path = output_dir / f'{args.model_type}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f'Summary saved to {summary_path}')
    
    # 2. 详细结果（CSV）
    details_path = output_dir / f'{args.model_type}_detailed_results.csv'
    df = pd.DataFrame(results)
    df.to_csv(details_path, index=False)
    logging.info(f'Detailed results saved to {details_path}')
    
    # 3. LaTeX表格
    latex_table = evaluator.generate_latex_table(model_name)
    latex_path = output_dir / f'{args.model_type}_latex_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    logging.info(f'LaTeX table saved to {latex_path}')
    
    # 4. 可视化图表
    logging.info('Generating visualizations...')
    evaluator.plot_metrics_distribution(output_dir)
    evaluator.plot_confusion_matrix(output_dir)
    
    # 5. 生成Markdown报告
    markdown_report = f"""
# Evaluation Report: {model_name}

## Model Information
- **Model Type**: {args.model_type}
- **Model Path**: {args.model}
- **Input Channels**: {args.in_ch}
- **Threshold**: {args.threshold}

## Dataset Information
- **Test Images**: {args.test_img}
- **Test Masks**: {args.test_mask}
- **Number of Images**: {summary["num_images"]}

## Segmentation Metrics

| Metric | Mean ± Std | Median | Min | Max | Global |
|--------|------------|--------|-----|-----|--------|
| Dice Coefficient | {summary["mean_dice"]:.4f} ± {summary["std_dice"]:.4f} | {summary["median_dice"]:.4f} | {summary["min_dice"]:.4f} | {summary["max_dice"]:.4f} | {summary["global_dice"]:.4f} |
| IoU (Jaccard) | {summary["mean_iou"]:.4f} ± {summary["std_iou"]:.4f} | {summary["median_iou"]:.4f} | - | - | {summary["global_iou"]:.4f} |
| Precision | {summary["mean_precision"]:.4f} ± {summary["std_precision"]:.4f} | - | - | - | - |
| Recall | {summary["mean_recall"]:.4f} ± {summary["std_recall"]:.4f} | - | - | - | - |
| F1-Score | {summary["mean_f1"]:.4f} ± {summary["std_f1"]:.4f} | - | - | - | - |
| Accuracy | {summary["mean_accuracy"]:.4f} ± {summary["std_accuracy"]:.4f} | - | - | - | - |
| Specificity | {summary["mean_specificity"]:.4f} ± {summary["std_specificity"]:.4f} | - | - | - | - |

## Boundary Metrics

| Metric | Mean ± Std |
|--------|------------|
| Boundary IoU | {summary["mean_boundary_iou"]:.4f} ± {summary["std_boundary_iou"]:.4f} |
| Boundary F1 | {summary["mean_boundary_f1"]:.4f} ± {summary["std_boundary_f1"]:.4f} |

## Confusion Matrix

|            | Predicted Negative | Predicted Positive |
|------------|--------------------|--------------------|
| **Actual Negative** | {summary["total_tn"]:,} (TN) | {summary["total_fp"]:,} (FP) |
| **Actual Positive** | {summary["total_fn"]:,} (FN) | {summary["total_tp"]:,} (TP) |

## Key Findings

1. **Overall Performance**: The model achieves a mean Dice coefficient of {summary["mean_dice"]:.4f}, indicating {"excellent" if summary["mean_dice"] > 0.9 else "good" if summary["mean_dice"] > 0.8 else "moderate"} segmentation performance.

2. **Precision vs Recall**: The model shows {"higher precision" if summary["mean_precision"] > summary["mean_recall"] else "higher recall" if summary["mean_recall"] > summary["mean_precision"] else "balanced precision and recall"} ({summary["mean_precision"]:.4f} vs {summary["mean_recall"]:.4f}), suggesting it is {"more conservative" if summary["mean_precision"] > summary["mean_recall"] else "more aggressive" if summary["mean_recall"] > summary["mean_precision"] else "well-balanced"} in predictions.

3. **Boundary Quality**: Boundary IoU of {summary["mean_boundary_iou"]:.4f} indicates {"excellent" if summary["mean_boundary_iou"] > 0.7 else "good" if summary["mean_boundary_iou"] > 0.5 else "moderate"} boundary delineation capability.

4. **Consistency**: Standard deviation of Dice ({summary["std_dice"]:.4f}) shows {"high" if summary["std_dice"] < 0.05 else "moderate" if summary["std_dice"] < 0.1 else "variable"} consistency across the test set.

## Visualizations

- Metrics distribution plot: `metrics_distribution.png`
- Confusion matrix: `confusion_matrix.png`
- Individual predictions: `visualizations/` directory

## Files Generated

- Summary (JSON): `{args.model_type}_summary.json`
- Detailed results (CSV): `{args.model_type}_detailed_results.csv`
- LaTeX table: `{args.model_type}_latex_table.tex`
- This report: `{args.model_type}_evaluation_report.md`
"""
    
    report_path = output_dir / f'{args.model_type}_evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write(markdown_report)
    logging.info(f'Markdown report saved to {report_path}')
    
    # 6. 生成简洁的论文表格（多种格式）
    paper_table_simple = f"""
% 简洁版表格（适合对比多个模型）
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Comparison on Test Dataset}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Dice}} & \\textbf{{IoU}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1}} \\\\
\\midrule
{model_name} & {summary["mean_dice"]:.4f} & {summary["mean_iou"]:.4f} & {summary["mean_precision"]:.4f} & {summary["mean_recall"]:.4f} & {summary["mean_f1"]:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    paper_table_path = output_dir / f'{args.model_type}_paper_table.tex'
    with open(paper_table_path, 'w') as f:
        f.write(paper_table_simple)
    
    print(f'\n{"Files Generated":=^80}')
    print(f'  1. Summary (JSON):           {summary_path}')
    print(f'  2. Detailed Results (CSV):   {details_path}')
    print(f'  3. LaTeX Table:              {latex_path}')
    print(f'  4. Paper Table:              {paper_table_path}')
    print(f'  5. Markdown Report:          {report_path}')
    print(f'  6. Metrics Distribution:     {output_dir}/metrics_distribution.png')
    print(f'  7. Confusion Matrix:         {output_dir}/confusion_matrix.png')
    if args.visualize:
        print(f'  8. Visualizations:           {output_dir}/visualizations/')
    print('='*80)
    
    logging.info('Evaluation complete!')


if __name__ == '__main__':
    main()