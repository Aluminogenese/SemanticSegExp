"""
eval_ssaf_ablation.py
SSAF 子组件消融实验评估脚本

对应训练脚本: train_ablation.py
Checkpoint 命名规则: checkpoints_ssaf_ablation/BEST_ssaf_{variant}.pth

7个变体:
  full          — 完整 MS-HRNet（所有子组件）
  wo_spectral   — 去掉光谱注意力
  wo_channel    — 去掉通道注意力 (SE Block)
  wo_band_inter — 去掉波段交互模块
  wo_ms_spatial — 多尺度空间注意力 → 单尺度
  wo_dyn_gate   — 动态门控 → 固定权重残差
  no_ssaf       — 无 SSAF（等价于原始 HRNet）

使用方法:
  python eval_ssaf_ablation.py \
      --test-img  /path/to/test/images/ \
      --test-mask /path/to/test/labels/ \
      --ckpt-dir  checkpoints_ssaf_ablation \
      --output-dir results_ssaf_ablation \
      --in-ch 4

输出:
  results_ssaf_ablation/
    ssaf_ablation_results.csv      — 完整指标表格
    ssaf_ablation_latex.tex        — 论文用 LaTeX 表格
    ssaf_ablation_summary.json     — JSON 格式汇总
    ssaf_ablation_barplot.png      — 各指标柱状图
    per_image/                     — 每个变体的逐图结果
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation
from tqdm import tqdm

# 导入消融模型和预测工具
sys.path.insert(0, str(Path(__file__).parent))
from models.ms_hrnet_ablation import MSHRNetAblation
from predict import read_image_any, normalize_image


# ─────────────────────────────────────────────────────────────────────────────
# 变体定义（顺序即论文表格行顺序）
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    # (variant_key,  display_name,                          description)
    ("no_ssaf",       "HRNet (w/o SSAF)",                  "Baseline, no SSAF module"),
    ("wo_spectral",   "MS-HRNet w/o Spectral Attn",        "Remove spectral attention"),
    ("wo_channel",    "MS-HRNet w/o Channel Attn",         "Remove SE block"),
    ("wo_band_inter", "MS-HRNet w/o Band Interaction",     "Remove inter-band interaction"),
    ("wo_ms_spatial", "MS-HRNet w/o Multi-Scale Spatial",  "Single-scale spatial attention"),
    ("wo_dyn_gate",   "MS-HRNet w/o Dynamic Gate",         "Fixed-weight residual"),
    ("full",          "MS-HRNet (Full, Ours)",              "Complete model"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_bin, gt_bin):
    """
    计算单张图像的全套指标
    pred_bin, gt_bin: bool ndarray [H, W]
    """
    pred = pred_bin.flatten()
    gt   = gt_bin.flatten()

    tp = int(np.sum( pred &  gt))
    fp = int(np.sum( pred & ~gt))
    fn = int(np.sum(~pred &  gt))
    tn = int(np.sum(~pred & ~gt))

    eps = 1e-8
    iou       = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)

    return dict(iou=iou, precision=precision, recall=recall,
                f1=f1, accuracy=accuracy, tp=tp, fp=fp, fn=fn, tn=tn)


def compute_boundary_iou(pred_bin, gt_bin, tolerance=3):
    """
    Boundary IoU（3像素容忍）
    与论文实验设置一致
    """
    if not gt_bin.any():
        return 0.0

    pred_boundary = pred_bin & ~binary_erosion(pred_bin, iterations=1)
    gt_boundary   = gt_bin   & ~binary_erosion(gt_bin,   iterations=1)

    pred_dilated = binary_dilation(pred_boundary, iterations=tolerance)
    gt_dilated   = binary_dilation(gt_boundary,   iterations=tolerance)

    intersection = np.sum((gt_dilated & pred_boundary) | (pred_dilated & gt_boundary))
    union        = np.sum(pred_boundary) + np.sum(gt_boundary)

    return float(intersection / (union + 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# 单变体评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(variant_key, ckpt_path, test_img_dir, test_mask_dir,
                     in_channels, device, threshold=0.5):
    """
    加载一个消融变体并评估整个测试集
    返回: (per_image_results: list[dict], summary: dict)
    """
    # 加载模型
    net = MSHRNetAblation(variant=variant_key, in_channels=in_channels,
                          num_classes=1)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()

    n_params = sum(p.numel() for p in net.parameters()) / 1e6

    # 查找测试图像
    img_files = sorted(
        glob(str(Path(test_img_dir) / '*.tif')) +
        glob(str(Path(test_img_dir) / '*.tiff')) +
        glob(str(Path(test_img_dir) / '*.png')) +
        glob(str(Path(test_img_dir) / '*.jpg'))
    )
    if not img_files:
        raise FileNotFoundError(f"No images found in {test_img_dir}")

    per_image = []

    for img_path in tqdm(img_files, desc=f"  {variant_key}", leave=False):
        stem = Path(img_path).stem

        # 查找对应 mask
        mask_path = None
        for ext in ['.png', '.tif', '.tiff', '.jpg']:
            cand = Path(test_mask_dir) / f'{stem}{ext}'
            if cand.exists():
                mask_path = cand
                break
        if mask_path is None:
            logging.warning(f"Mask not found for {stem}, skipping")
            continue

        # 读取并预处理图像
        img_np = read_image_any(str(img_path))
        img_norm = normalize_image(img_np)
        img_t = torch.from_numpy(
            img_norm.transpose(2, 0, 1)
        ).unsqueeze(0).to(device, dtype=torch.float32)

        # 推理
        with torch.no_grad():
            out = net(img_t)
            # 消融变体训练时可能返回 tuple（带 attention_maps），取第一个
            if isinstance(out, tuple):
                out = out[0]
            prob = torch.sigmoid(out).squeeze().cpu().numpy()

        pred_bin = prob > threshold

        # 读取 GT
        gt_np = read_image_any(str(mask_path))
        if gt_np.ndim == 3:
            gt_np = gt_np[:, :, 0]
        gt_bin = gt_np > 0

        # 计算指标
        m = compute_metrics(pred_bin, gt_bin)
        m['boundary_iou'] = compute_boundary_iou(pred_bin, gt_bin)
        m['image'] = stem
        per_image.append(m)

    # 聚合
    df = pd.DataFrame(per_image)
    metric_cols = ['iou', 'precision', 'recall', 'f1', 'boundary_iou', 'accuracy']

    summary = {
        'variant':  variant_key,
        'n_params': round(n_params, 2),
        'n_images': len(per_image),
    }
    for col in metric_cols:
        summary[f'mean_{col}'] = float(df[col].mean())
        summary[f'std_{col}']  = float(df[col].std())

    # 全局 IoU（用汇总混淆矩阵计算，更稳定）
    total_tp = int(df['tp'].sum())
    total_fp = int(df['fp'].sum())
    total_fn = int(df['fn'].sum())
    summary['global_iou'] = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    return per_image, summary


# ─────────────────────────────────────────────────────────────────────────────
# 结果输出
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(all_summaries, variant_display_names, output_path):
    rows = []
    for key, name, _ in VARIANTS:
        if key not in all_summaries:
            continue
        s = all_summaries[key]
        rows.append({
            'Variant':      key,
            'Model':        name,
            'Params (M)':   s['n_params'],
            'IoU (%)':      round(s['mean_iou'] * 100, 2),
            'Precision (%)':round(s['mean_precision'] * 100, 2),
            'Recall (%)':   round(s['mean_recall'] * 100, 2),
            'F1 (%)':       round(s['mean_f1'] * 100, 2),
            'B-IoU (%)':    round(s['mean_boundary_iou'] * 100, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logging.info(f"CSV saved: {output_path}")
    return df


def save_latex(all_summaries, output_path):
    """
    生成论文格式的 LaTeX 表格
    对应审稿人要求的 SSAF 子组件消融表（替换原 Table 2 或新建 Table）
    最佳值加粗
    """
    # 找各指标最优值
    keys_in_order = [v[0] for v in VARIANTS if v[0] in all_summaries]
    best = {}
    for metric in ['mean_iou', 'mean_precision', 'mean_recall',
                   'mean_f1', 'mean_boundary_iou']:
        best[metric] = max(all_summaries[k][metric] for k in keys_in_order)

    def fmt(val, best_val, pct=True):
        s = f"{val * 100:.2f}" if pct else f"{val:.2f}"
        return f"\\textbf{{{s}}}" if abs(val - best_val) < 1e-6 else s

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study of each component in the SSAF module. "
        r"The best results are highlighted in \textbf{bold}. "
        r"``w/o'' denotes the removal of the corresponding component.}",
        r"\label{tab:ssaf_ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params (M)} & \textbf{IoU (\%)} "
        r"& \textbf{Precision (\%)} & \textbf{Recall (\%)} "
        r"& \textbf{F1 (\%)} & \textbf{B-IoU (\%)} \\",
        r"\midrule",
    ]

    for key, display_name, _ in VARIANTS:
        if key not in all_summaries:
            continue
        s = all_summaries[key]

        # 最后一行（full）前加分隔线
        if key == 'full':
            lines.append(r"\midrule")

        iou_s   = fmt(s['mean_iou'],          best['mean_iou'])
        prec_s  = fmt(s['mean_precision'],     best['mean_precision'])
        rec_s   = fmt(s['mean_recall'],        best['mean_recall'])
        f1_s    = fmt(s['mean_f1'],            best['mean_f1'])
        biou_s  = fmt(s['mean_boundary_iou'],  best['mean_boundary_iou'])
        param_s = f"{s['n_params']:.2f}"

        lines.append(
            f"{display_name} & {param_s} & {iou_s} & {prec_s} "
            f"& {rec_s} & {f1_s} & {biou_s} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    logging.info(f"LaTeX saved: {output_path}")


def save_barplot(all_summaries, output_path):
    """
    生成各指标柱状图，直观展示各组件的贡献
    """
    keys_ordered = [v[0] for v in VARIANTS if v[0] in all_summaries]
    short_names  = {
        'no_ssaf':       'No SSAF\n(Baseline)',
        'wo_spectral':   'w/o\nSpectral',
        'wo_channel':    'w/o\nChannel',
        'wo_band_inter': 'w/o\nBand Inter',
        'wo_ms_spatial': 'w/o\nMS-Spatial',
        'wo_dyn_gate':   'w/o\nDyn. Gate',
        'full':          'Full\n(Ours)',
    }

    metrics = [
        ('mean_iou',          'IoU (%)',          '#4C72B0'),
        ('mean_f1',           'F1 (%)',           '#DD8452'),
        ('mean_precision',    'Precision (%)',    '#55A868'),
        ('mean_recall',       'Recall (%)',       '#C44E52'),
        ('mean_boundary_iou', 'B-IoU (%)',        '#8172B2'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    fig.suptitle('SSAF Component Ablation Study', fontsize=14, fontweight='bold')

    x      = np.arange(len(keys_ordered))
    labels = [short_names[k] for k in keys_ordered]

    for ax, (metric, ylabel, color) in zip(axes, metrics):
        vals = [all_summaries[k][metric] * 100 for k in keys_ordered]

        bars = ax.bar(x, vals, color=color, alpha=0.75, edgecolor='black',
                      linewidth=0.8)

        # 高亮最佳（full）
        best_idx = keys_ordered.index('full') if 'full' in keys_ordered else np.argmax(vals)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.5)

        # 数值标注
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)

        # 设置 y 轴范围（留出标注空间，且不从0开始避免差异不明显）
        y_min = max(0, min(vals) - 2)
        y_max = max(vals) + 1.5
        ax.set_ylim(y_min, y_max)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"Bar plot saved: {output_path}")


def print_table(all_summaries):
    """终端打印对齐表格"""
    header = (f"{'Model':<38} {'Params':>8} {'IoU':>7} {'Prec':>7} "
              f"{'Recall':>7} {'F1':>7} {'B-IoU':>7}")
    sep    = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for key, display_name, _ in VARIANTS:
        if key not in all_summaries:
            continue
        s = all_summaries[key]
        mark = ' ←best' if key == 'full' else ''
        print(
            f"{display_name:<38} "
            f"{s['n_params']:>7.2f}M "
            f"{s['mean_iou']*100:>6.2f}% "
            f"{s['mean_precision']*100:>6.2f}% "
            f"{s['mean_recall']*100:>6.2f}% "
            f"{s['mean_f1']*100:>6.2f}% "
            f"{s['mean_boundary_iou']*100:>6.2f}%"
            f"{mark}"
        )
    print(sep + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SSAF ablation variants on test set'
    )
    parser.add_argument('--test-img',   required=True,
                        help='Test image directory')
    parser.add_argument('--test-mask',  required=True,
                        help='Test mask directory')
    parser.add_argument('--ckpt-dir',   default='checkpoints_ssaf_ablation',
                        help='Directory containing BEST_ssaf_*.pth checkpoints '
                             '(default: checkpoints_ssaf_ablation)')
    parser.add_argument('--output-dir', default='results_ssaf_ablation',
                        help='Output directory (default: results_ssaf_ablation)')
    parser.add_argument('--in-ch',      type=int, default=4,
                        help='Input channels (default: 4)')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Binarization threshold (default: 0.5)')
    parser.add_argument('--variants',   nargs='+', default=None,
                        help='Evaluate only these variants (default: all). '
                             'Example: --variants full no_ssaf wo_spectral')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir   = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'per_image').mkdir(exist_ok=True)

    logging.info(f"Device:     {device}")
    logging.info(f"Checkpoint: {ckpt_dir}")
    logging.info(f"Output:     {output_dir}")

    # 决定要评估哪些变体
    target_variants = args.variants if args.variants else [v[0] for v in VARIANTS]

    all_summaries = {}
    all_per_image = {}

    print(f"\n{'='*60}")
    print(f"  SSAF Ablation Evaluation  ({len(target_variants)} variants)")
    print(f"{'='*60}\n")

    for variant_key, display_name, desc in VARIANTS:
        if variant_key not in target_variants:
            continue

        ckpt_path = ckpt_dir / f'BEST_ssaf_{variant_key}.pth'
        if not ckpt_path.exists():
            logging.warning(f"Checkpoint not found: {ckpt_path}  →  skipping")
            continue

        logging.info(f"Evaluating: {display_name}")
        logging.info(f"  Checkpoint: {ckpt_path}")

        try:
            per_image, summary = evaluate_variant(
                variant_key=variant_key,
                ckpt_path=str(ckpt_path),
                test_img_dir=args.test_img,
                test_mask_dir=args.test_mask,
                in_channels=args.in_ch,
                device=device,
                threshold=args.threshold,
            )
            all_summaries[variant_key] = summary
            all_per_image[variant_key] = per_image

            # 保存每个变体的逐图结果
            per_img_path = output_dir / 'per_image' / f'{variant_key}.csv'
            pd.DataFrame(per_image).to_csv(per_img_path, index=False)

            logging.info(
                f"  → IoU={summary['mean_iou']*100:.2f}%  "
                f"F1={summary['mean_f1']*100:.2f}%  "
                f"B-IoU={summary['mean_boundary_iou']*100:.2f}%  "
                f"({summary['n_images']} images)"
            )

        except Exception as e:
            logging.error(f"Failed to evaluate {variant_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_summaries:
        logging.error("No variants evaluated. Check checkpoint paths.")
        return

    # ── 终端打印 ──
    print_table(all_summaries)

    # ── 保存 CSV ──
    csv_path = output_dir / 'ssaf_ablation_results.csv'
    df = save_csv(all_summaries,
                  {v[0]: v[1] for v in VARIANTS},
                  csv_path)
    print(df.to_string(index=False))

    # ── 保存 LaTeX ──
    latex_path = output_dir / 'ssaf_ablation_latex.tex'
    save_latex(all_summaries, latex_path)

    # ── 保存 JSON ──
    json_path = output_dir / 'ssaf_ablation_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    logging.info(f"JSON saved: {json_path}")

    # ── 保存柱状图 ──
    barplot_path = output_dir / 'ssaf_ablation_barplot.png'
    save_barplot(all_summaries, barplot_path)

    # ── 打印增量分析 ──
    if 'no_ssaf' in all_summaries and 'full' in all_summaries:
        base = all_summaries['no_ssaf']
        full = all_summaries['full']
        print("Improvement over baseline (no_ssaf → full):")
        for m in ['mean_iou', 'mean_precision', 'mean_recall',
                  'mean_f1', 'mean_boundary_iou']:
            delta = (full[m] - base[m]) * 100
            label = m.replace('mean_', '').replace('_', ' ').title()
            print(f"  {label:20s}: {base[m]*100:.2f}% → {full[m]*100:.2f}%  "
                  f"(+{delta:.2f}pp)")
        print()

    print(f"All results saved to: {output_dir}/")
    print(f"  {csv_path.name}")
    print(f"  {latex_path.name}")
    print(f"  {json_path.name}")
    print(f"  {barplot_path.name}")
    print(f"  per_image/  (per-image CSV for each variant)")


if __name__ == '__main__':
    main()