"""
gen_paper_figures.py
论文图生成脚本 — IASUNet 风格（完整可运行版）
 
三步工作流：
  Step 1  scan    — 扫描测试集，按场景类型打印候选图片名
  Step 2  compare — 指定图片名，生成 IASUNet 风格多行对比图
  Step 3  failure — 自动找失败案例并生成分析图
 
使用示例：
  # Step1：扫描候选（先跑这步，记下想要的图片名）
  python gen_paper_figures.py --mode scan \
      --dataset custom \
      --test-img /home/lucianlu/data/dat_4bands/images/ --test-mask /home/lucianlu/data/dat_4bands/labels/
 
  # Step2：生成对比图（不带红框）
  python gen_paper_figures.py --mode compare \
      --dataset custom \
      --test-img /home/lucianlu/data/dat_4bands/images/ --test-mask /home/lucianlu/data/dat_4bands/labels/ \
      --images 000000193 000000047 000000215
 
  # Step2b：带红框版本
  python gen_paper_figures.py --mode compare \
      --dataset custom \
      --test-img /path/to/images/ --test-mask /path/to/labels/ \
      --images 000000193 000000047 000000215 \
      --draw-box
 
  # Step3：失败案例分析
  python gen_paper_figures.py --mode failure \
      --dataset custom \
      --test-img /path/to/images/ --test-mask /path/to/labels/
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from glob import glob
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from scipy.ndimage import (binary_erosion, binary_dilation,
                           uniform_filter, label as sp_label)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNet, MSHRNet, UNetFormer
from predict import read_image_any, normalize_image



# =============================================================================
# ① 配置区 — 按实际路径修改
# =============================================================================

CUSTOM_CHECKPOINTS = {
    'UNet':            'checkpoints_all/BEST_unet_combined_dat_4bands_seed42.pth',
    'UNet++':          'checkpoints_all/BEST_unet_plusplus_combined_dat_4bands_seed42.pth',
    'PSPNet':          'checkpoints_all/BEST_pspnet_dat_4bands.pth',
    'DeepLabV3+':      'checkpoints_all/BEST_deeplabv3_plus_combined_dat_4bands_seed456.pth',
    'UNetFormer':      'checkpoints_all/BEST_unetformer_combined_dat_4bands.pth',
    'HRNet':           'checkpoints_all/BEST_hrnet_combined_dat_4bands_seed123.pth',
    'MS-HRNet (Ours)': 'checkpoints_all/BEST_ms_hrnet_combined_dat_4bands_seed2025.pth',
}

POTSDAM_CHECKPOINTS = {
    'UNet':            'checkpoints_potsdam/BEST_unet_combined_potsdam.pth',
    'UNet++':          'checkpoints_potsdam/BEST_unet_plusplus_combined_potsdam.pth',
    'PSPNet':          'checkpoints_potsdam/BEST_pspnet_combined_potsdam.pth',
    'DeepLabV3+':      'checkpoints_potsdam/BEST_deeplabv3_plus_combined_potsdam.pth',
    'UNetFormer':      'checkpoints_potsdam/BEST_unetformer_combined_potsdam_4.pth',
    'HRNet':           'checkpoints_potsdam/BEST_hrnet_combined_potsdam.pth',
    'MS-HRNet (Ours)': 'checkpoints_potsdam/BEST_ms_hrnet_combined_potsdam_8.pth',
}

# 列顺序：和 IASUNet 论文一样，最后一列是我们的方法
MODEL_ORDER = ['UNet', 'UNetFormer', 'UNet++', 'PSPNet', 'DeepLabV3+',
                'HRNet', 'MS-HRNet (Ours)']


# =============================================================================
# ② 模型工厂
# =============================================================================

def build_model(name, in_ch):
    tbl = {
        'UNet':            lambda: UNet(in_channels=in_ch, num_classes=1),
        'UNetFormer':      lambda: UNetFormer(in_channels=in_ch, num_classes=1),
        'UNet++':          lambda: UNetPlusPlus(in_channels=in_ch, num_classes=1),
        'PSPNet':          lambda: PSPNet(in_channels=in_ch, num_classes=1),
        'DeepLabV3+':      lambda: DeepLabV3Plus(in_channels=in_ch, num_classes=1),
        'HRNet':           lambda: HRNet(in_channels=in_ch, num_classes=1,
                                         base_channels=48),
        'MS-HRNet (Ours)': lambda: MSHRNet(in_channels=in_ch, num_classes=1,
                                            base_channels=48),
    }
    if name not in tbl:
        raise ValueError(f'Unknown model: {name}')
    return tbl[name]()
 
 
def load_models(ckpts, in_ch, device):
    nets = {}
    for name in MODEL_ORDER:
        path = ckpts.get(name, '')
        if not Path(path).exists():
            print(f'  [skip] {name}')
            continue
        try:
            net = build_model(name, in_ch)
            net.load_state_dict(torch.load(path, map_location=device))
            net.to(device).eval()
            nets[name] = net
            print(f'  [ok]   {name}')
        except Exception as e:
            print(f'  [err]  {name}: {e}')
    return nets
 
 
# =============================================================================
# 推理 & 指标
# =============================================================================
 
def infer(net, img_np, device, thr=0.5):
    norm = normalize_image(img_np)
    t = (torch.from_numpy(norm.transpose(2, 0, 1))
         .unsqueeze(0).to(device, dtype=torch.float32))
    with torch.no_grad():
        out = net(t)
        if isinstance(out, tuple):
            out = out[0]
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
    return prob > thr
 
 
def calc_metrics(p, g):
    tp  = int(np.sum(p & g))
    fp  = int(np.sum(p & ~g))
    fn  = int(np.sum(~p & g))
    eps = 1e-8
    iou  = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    if g.any():
        pb   = p & ~binary_erosion(p, iterations=1)
        gb   = g & ~binary_erosion(g, iterations=1)
        nt   = int(np.sum(
            (binary_dilation(gb, iterations=3) & pb) |
            (binary_dilation(pb, iterations=3) & gb)))
        biou = nt / (int(np.sum(pb)) + int(np.sum(gb)) + eps)
    else:
        biou = 0.0
    return dict(iou=iou, precision=prec, recall=rec,
                f1=f1, biou=float(biou))
 
 
# =============================================================================
# 工具函数
# =============================================================================
 
def to_rgb(img_np):
    arr = (img_np[:, :, :3] if img_np.ndim == 3
           else np.stack([img_np]*3, axis=-1)).astype(np.float32)
    for c in range(3):
        lo, hi = np.percentile(arr[:, :, c], [2, 98])
        arr[:, :, c] = np.clip(
            (arr[:, :, c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return arr
 
 
def classify_scene(gt, preds_dict):
    cov = float(gt.sum()) / gt.size
    _, n_comp = sp_label(gt)
    iou_vals = []
    for p in preds_dict.values():
        tp = np.sum(p & gt); fp = np.sum(p & ~gt); fn = np.sum(~p & gt)
        iou_vals.append(float(tp / (tp + fp + fn + 1e-8)))
    iou_std = float(np.std(iou_vals)) if iou_vals else 0.0
    if cov < 0.08:
        return 'single'
    elif cov > 0.35 or (n_comp > 15 and cov > 0.15):
        return 'dense'
    elif iou_std > 0.06:
        return 'complex'
    else:
        return 'boundary'
 
 
def find_mask_path(stem, mask_dir):
    for ext in ['.png', '.tif', '.tiff', '.jpg']:
        p = Path(mask_dir) / f'{stem}{ext}'
        if p.exists():
            return p
    return None
 
 
def build_sample(img_path, mask_dir, nets, device):
    stem = Path(img_path).stem
    mp   = find_mask_path(stem, mask_dir)
    if mp is None:
        return None
    img    = read_image_any(str(img_path))
    gt_raw = read_image_any(str(mp))
    gt     = (gt_raw[:, :, 0] if gt_raw.ndim == 3 else gt_raw) > 0
    preds  = {name: infer(net, img, device) for name, net in nets.items()}
    mets   = {name: calc_metrics(p, gt) for name, p in preds.items()}
    cov    = float(gt.sum()) / gt.size
    scene  = classify_scene(gt, preds)
    our    = mets.get('MS-HRNet (Ours)', {})
    base   = mets.get('HRNet', {})
    score  = ((our.get('iou', 0) - base.get('iou', 0)) *
               our.get('biou', 0) * min(cov / 0.3, 1.0))
    return dict(name=stem, img=img, gt=gt, preds=preds,
                metrics=mets, coverage=cov, scene=scene, score=score)
 
 
def load_samples(img_dir, mask_dir, nets, device,
                 specify=None, max_scan=120):
    files = sorted(
        glob(f'{img_dir}/*.tif')  + glob(f'{img_dir}/*.tiff') +
        glob(f'{img_dir}/*.png')  + glob(f'{img_dir}/*.jpg'))
    print(f'  Total images: {len(files)}')
    if specify:
        fmap  = {Path(f).stem: f for f in files}
        files = [fmap[s] for s in specify if s in fmap]
        miss  = [s for s in specify if s not in fmap]
        if miss:
            print(f'  [WARN] not found: {miss}')
        print(f'  Loading {len(files)} specified images...')
    else:
        random.seed(42)
        files = random.sample(files, min(max_scan, len(files)))
        print(f'  Scanning {len(files)} images...')
    samples = []
    for f in tqdm(files, desc='  Processing', leave=False):
        s = build_sample(f, mask_dir, nets, device)
        if s is not None:
            samples.append(s)
    return samples
 
 
# =============================================================================
# 模式 1：scan
# =============================================================================
 
def mode_scan(samples, out_dir):
    scene_desc = {
        'single':   'Single/Sparse Building',
        'dense':    'Dense Small Buildings',
        'complex':  'Complex Scene',
        'boundary': 'Boundary Challenge',
    }
    groups = {}
    for s in samples:
        groups.setdefault(s['scene'], []).append(s)
 
    print('\n' + '='*68)
    print('  Candidate Images by Scene Type')
    print('  Copy image names to --images when running --mode compare')
    print('='*68)
 
    result = {}
    for key in ['single', 'dense', 'complex', 'boundary']:
        grp = groups.get(key, [])
        if not grp:
            continue
        top = sorted(grp,
                     key=lambda s: s['metrics'].get(
                         'MS-HRNet (Ours)', {}).get('iou', 0),
                     reverse=True)[:6]
        print(f'\n  [{scene_desc.get(key, key)}]  ({len(grp)} found)')
        print(f'  {"Image Name":<36} {"Coverage":>9} {"Ours IoU":>9}'
              f' {"HRNet IoU":>10} {"Diff":>7}')
        print(f'  {"-"*72}')
        names = []
        for s in top:
            oi = s['metrics'].get('MS-HRNet (Ours)', {}).get('iou', 0)
            bi = s['metrics'].get('HRNet', {}).get('iou', 0)
            print(f'  {s["name"]:<36} {s["coverage"]:>8.1%}'
                  f' {oi:>9.4f} {bi:>10.4f} {oi-bi:>+7.4f}')
            names.append(s['name'])
        result[key] = names
 
    out_path = Path(out_dir) / 'scan_candidates.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n  Candidates saved -> {out_path}')
    print('='*68)
 
 
# =============================================================================
# 模式 2：compare
# =============================================================================
 
def best_diff_box(pred_ours, pred_base, gt, box_size):
    h, w    = gt.shape
    improve = ((pred_ours == gt).astype(float) -
               (pred_base == gt).astype(float))
    density = uniform_filter(improve, size=box_size)
    half    = box_size // 2
    tmp     = density.copy()
    tmp[:half, :] = tmp[-half:, :] = -1
    tmp[:, :half] = tmp[:, -half:] = -1
    cy, cx  = np.unravel_index(np.argmax(tmp), tmp.shape)
    if tmp[cy, cx] < 0.02:
        return None
    y1 = max(0, cy - half); y2 = min(h, y1 + box_size)
    x1 = max(0, cx - half); x2 = min(w, x1 + box_size)
    if y2 - y1 < box_size: y1 = max(0, y2 - box_size)
    if x2 - x1 < box_size: x1 = max(0, x2 - box_size)
    return (y1, x1, y2, x2)
 
 
def add_red_box(ax, roi, lw=2.0):
    y1, x1, y2, x2 = roi
    ax.add_patch(mpatches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        lw=lw, edgecolor='red', facecolor='none', zorder=10))
 
 
def mode_compare(samples, model_names, output_path,
                 draw_box=False, box_size=100, dpi=300):
    """
    IASUNet 风格对比图。
    每行一个场景，每列一个模型，无任何文字标注。
    列标题格式：(a)Images (b)Labels (c)UNet ... (h)MS-HRNet(Ours)
    """
    n_rows = len(samples)
 
    col_labels = ['(a)\nImages', '(b)\nLabels']
    for i, name in enumerate(model_names):
        letter = chr(ord('c') + i)
        short  = name.replace(' (Ours)', '')
        suffix = '\n(Ours)' if 'Ours' in name else ''
        col_labels.append(f'({letter})\n{short}{suffix}')
    n_cols = len(col_labels)
 
    cell = 1.85
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell * n_cols, cell * n_rows),
        gridspec_kw=dict(hspace=0.04, wspace=0.03,
                         left=0.01, right=0.99,
                         top=0.92,  bottom=0.01))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
 
    for ci, title in enumerate(col_labels):
        kw = dict(fontsize=8, fontweight='bold', pad=3)
        if 'Ours' in title:
            kw['color'] = '#CC0000'
        axes[0, ci].set_title(title, **kw)
 
    for row, s in enumerate(samples):
        vis   = to_rgb(s['img'])
        gt    = s['gt']
        preds = s['preds']
 
        col_data = [vis, gt.astype(float)]
        col_rgb  = [True, False]
        for mn in model_names:
            p = preds.get(mn, np.zeros_like(gt, dtype=float)).astype(float)
            col_data.append(p)
            col_rgb.append(False)
 
        box = None
        if draw_box and 'MS-HRNet (Ours)' in preds and 'HRNet' in preds:
            box = best_diff_box(preds['MS-HRNet (Ours)'],
                                preds['HRNet'], gt, box_size)
 
        for ci, (data, is_rgb) in enumerate(zip(col_data, col_rgb)):
            ax = axes[row, ci]
            if is_rgb:
                ax.imshow(data, interpolation='bilinear')
            else:
                ax.imshow(data, cmap='gray', vmin=0, vmax=1,
                          interpolation='bilinear')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            # 红框只画在 Input 和 GT 列上
            if box is not None and ci <= 1:
                add_red_box(ax, box)
 
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', pad_inches=0.05)
    plt.close(fig)
    print(f'  Saved: {output_path}')
 
 
# =============================================================================
# 模式 3：failure
# =============================================================================
 
def classify_failure(s):
    m    = s['metrics'].get('MS-HRNet (Ours)', {})
    cov  = s['coverage']
    prec = m.get('precision', 0)
    rec  = m.get('recall', 0)
    _, n = sp_label(s['gt'])
    if rec < 0.5 and cov < 0.1:
        return 'Small/Isolated\nBuildings'
    elif prec < 0.6 and rec > 0.7:
        return 'False Positives\n(Background)'
    elif rec < 0.5 and n > 20:
        return 'Dense Small\nBuildings'
    elif m.get('biou', 0) < 0.3 and m.get('iou', 0) > 0.5:
        return 'Boundary\nImprecision'
    else:
        return 'Complex\nBackground'
 
 
def error_overlay(pred, gt):
    h, w = gt.shape
    err  = np.zeros((h, w, 3), dtype=np.float32)
    err[pred & gt]  = [0.15, 0.80, 0.15]   # TP 绿
    err[pred & ~gt] = [0.90, 0.15, 0.15]   # FP 红
    err[~pred & gt] = [0.15, 0.35, 0.90]   # FN 蓝
    return err
 
 
def failure_box(pred, gt, box_size=120):
    h, w    = gt.shape
    err     = (pred != gt).astype(float)
    density = uniform_filter(err, size=box_size)
    half    = box_size // 2
    tmp     = density.copy()
    tmp[:half, :] = tmp[-half:, :] = -1
    tmp[:, :half] = tmp[:, -half:] = -1
    cy, cx  = np.unravel_index(np.argmax(tmp), tmp.shape)
    if tmp[cy, cx] < 0.05:
        return None
    y1 = max(0, cy - half); y2 = min(h, y1 + box_size)
    x1 = max(0, cx - half); x2 = min(w, x1 + box_size)
    if y2 - y1 < box_size: y1 = max(0, y2 - box_size)
    if x2 - x1 < box_size: x1 = max(0, x2 - box_size)
    return (y1, x1, y2, x2)
 
 
# def mode_failure(samples, model_names, out_dir, n_cases=4, dpi=300):
#     """
#     失败案例分析图。
#     布局：(a)Images | (b)Labels | (c)Error Map | (d)HRNet | (e)MS-HRNet(Ours)
#     红框标出误差集中区域，左侧行标注失败原因类型。
#     """
#     valid = [s for s in samples
#              if s['coverage'] > 0.05
#              and 'MS-HRNet (Ours)' in s['metrics']]
#     if not valid:
#         print('  No valid samples for failure analysis.')
#         return
 
#     worst = sorted(valid,
#                    key=lambda s: s['metrics']['MS-HRNet (Ours)']['iou'])[:n_cases]
 
#     print(f'\n  Failure cases (lowest MS-HRNet IoU):')
#     print(f'  {"Image":<36} {"IoU":>7} {"Prec":>7}'
#           f' {"Recall":>8} {"Scene":>10}')
#     print(f'  {"-"*70}')
 
#     cases = []
#     for s in worst:
#         m      = s['metrics']['MS-HRNet (Ours)']
#         reason = classify_failure(s)
#         print(f'  {s["name"]:<36} {m["iou"]:>7.4f}'
#               f' {m["precision"]:>7.4f} {m["recall"]:>8.4f}'
#               f' {s["scene"]:>10}')
#         cases.append((s, reason))
 
#     col_labels = ['(a)\nImages', '(b)\nLabels',
#                   '(c)\nError Map', '(d)\nHRNet', '(e)\nMS-HRNet\n(Ours)']
#     n_cols = len(col_labels)
#     n_rows = len(cases)
#     cell   = 1.85
 
#     fig, axes = plt.subplots(
#         n_rows, n_cols,
#         figsize=(cell * n_cols, cell * n_rows),
#         gridspec_kw=dict(hspace=0.06, wspace=0.03,
#                          left=0.11, right=0.99,
#                          top=0.92,  bottom=0.07))
#     if n_rows == 1:
#         axes = axes[np.newaxis, :]
 
#     for ci, title in enumerate(col_labels):
#         kw = dict(fontsize=8, fontweight='bold', pad=3)
#         if 'Ours' in title:
#             kw['color'] = '#CC0000'
#         axes[0, ci].set_title(title, **kw)
 
#     for row, (s, reason) in enumerate(cases):
#         vis    = to_rgb(s['img'])
#         gt     = s['gt']
#         our_p  = s['preds'].get('MS-HRNet (Ours)', np.zeros_like(gt))
#         base_p = s['preds'].get('HRNet', np.zeros_like(gt))
#         err    = error_overlay(our_p, gt)
#         fbox   = failure_box(our_p, gt)
 
#         col_data = [vis, gt.astype(float), err,
#                     base_p.astype(float), our_p.astype(float)]
#         col_rgb  = [True, False, True, False, False]
 
#         for ci, (data, is_rgb) in enumerate(zip(col_data, col_rgb)):
#             ax = axes[row, ci]
#             if is_rgb:
#                 ax.imshow(data, interpolation='bilinear')
#             else:
#                 ax.imshow(data, cmap='gray', vmin=0, vmax=1,
#                           interpolation='bilinear')
#             ax.set_xticks([]); ax.set_yticks([])
#             for sp in ax.spines.values():
#                 sp.set_visible(False)
#             if fbox is not None and ci <= 2:
#                 ax.add_patch(mpatches.Rectangle(
#                     (fbox[1], fbox[0]),
#                     fbox[3] - fbox[1], fbox[2] - fbox[0],
#                     lw=2.0, edgecolor='red',
#                     facecolor='none', zorder=10))
 
#         axes[row, 0].set_ylabel(
#             reason, fontsize=7.5, rotation=0,
#             ha='right', va='center', labelpad=52,
#             fontweight='bold')
 
#     # 图例
#     legend_els = [
#         mpatches.Patch(facecolor='#26CC26', label='TP (Correct)'),
#         mpatches.Patch(facecolor='#E62626', label='FP (False alarm)'),
#         mpatches.Patch(facecolor='#2659E6', label='FN (Missed)'),
#     ]
#     fig.legend(handles=legend_els, loc='lower center', ncol=3,
#                fontsize=7.5, frameon=True,
#                bbox_to_anchor=(0.5, -0.02),
#                title='Error Map Color Code',
#                title_fontsize=7.5)
 
#     out_path = Path(out_dir) / 'failure_cases.png'
#     plt.savefig(out_path, dpi=dpi, bbox_inches='tight',
#                 facecolor='white', pad_inches=0.05)
#     plt.close(fig)
#     print(f'  Saved: {out_path}')
 
#     # 保存指标 CSV
#     csv_path = Path(out_dir) / 'failure_metrics.csv'
#     with open(csv_path, 'w', newline='') as f:
#         w = csv.writer(f)
#         w.writerow(['Image', 'Scene', 'Failure Reason',
#                     'IoU', 'Precision', 'Recall', 'F1', 'B-IoU', 'Coverage'])
#         for s, reason in cases:
#             m = s['metrics'].get('MS-HRNet (Ours)', {})
#             w.writerow([s['name'], s['scene'],
#                         reason.replace('\n', ' '),
#                         f"{m.get('iou',0):.4f}",
#                         f"{m.get('precision',0):.4f}",
#                         f"{m.get('recall',0):.4f}",
#                         f"{m.get('f1',0):.4f}",
#                         f"{m.get('biou',0):.4f}",
#                         f"{s['coverage']:.3f}"])
#     print(f'  Metrics CSV: {csv_path}')

def overlay_error_on_image(img, pred, gt, alpha=0.5):
    vis = to_rgb(img).copy()
    err = np.zeros_like(vis)

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    err[tp] = [0.2, 0.8, 0.2]
    err[fp] = [0.9, 0.2, 0.2]
    err[fn] = [0.2, 0.4, 0.9]

    return vis * (1 - alpha) + err * alpha


def find_worst_patch(pred, gt, size=120):
    err = (pred != gt).astype(float)

    # 强调 FN（论文更关注漏检）
    fn = (~pred & gt).astype(float)
    score = err + 2 * fn

    density = uniform_filter(score, size=size)

    h, w = gt.shape
    half = size // 2

    density[:half, :] = -1
    density[-half:, :] = -1
    density[:, :half] = -1
    density[:, -half:] = -1

    cy, cx = np.unravel_index(np.argmax(density), density.shape)

    y1 = cy - half
    x1 = cx - half
    y2 = y1 + size
    x2 = x1 + size

    return (y1, x1, y2, x2)


def crop(img, box):
    y1, x1, y2, x2 = box
    return img[y1:y2, x1:x2]


def mode_failure(samples, model_names, out_dir, n_cases=4, dpi=300):

    valid = [s for s in samples if 'MS-HRNet (Ours)' in s['metrics']]

    # ⭐ 更合理排序（不是单纯 IoU）
    def failure_score(s):
        m = s['metrics']['MS-HRNet (Ours)']
        return (1 - m['iou']) + (1 - m['recall']) * 0.5

    worst = sorted(valid, key=failure_score, reverse=True)[:n_cases]

    n_rows = len(worst) * 2   # ⭐ 每个case两行
    n_cols = 5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows),
        gridspec_kw=dict(hspace=0.05, wspace=0.03)
    )

    col_titles = ['Image', 'GT', 'Error', 'HRNet', 'Ours']

    for i in range(n_cols):
        axes[0, i].set_title(col_titles[i], fontsize=9)

    for idx, s in enumerate(worst):

        r = idx * 2

        img = s['img']
        gt = s['gt']
        ours = s['preds']['MS-HRNet (Ours)']
        base = s['preds']['HRNet']

        overlay = overlay_error_on_image(img, ours, gt)

        box = find_worst_patch(ours, gt)

        # ===== 第一行（全图）=====
        data_row1 = [
            to_rgb(img),
            gt,
            overlay,
            base,
            ours
        ]

        for c in range(n_cols):
            ax = axes[r, c]
            d = data_row1[c]

            if c == 0 or c == 2:
                ax.imshow(d)
            else:
                ax.imshow(d, cmap='gray')

            ax.axis('off')

            # 画框
            y1, x1, y2, x2 = box
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                edgecolor='red', fill=False, lw=2
            ))

        # ===== 第二行（zoom）=====
        zoom_data = [
            crop(to_rgb(img), box),
            crop(gt, box),
            crop(overlay, box),
            crop(base, box),
            crop(ours, box)
        ]

        for c in range(n_cols):
            ax = axes[r+1, c]
            d = zoom_data[c]

            if c == 0 or c == 2:
                ax.imshow(d)
            else:
                ax.imshow(d, cmap='gray')

            ax.axis('off')

    out_path = Path(out_dir) / 'failure_cases_advanced.png'
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f'Saved advanced failure figure -> {out_path}')
 
 
# =============================================================================
# 主函数
# =============================================================================
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='scan',
                    choices=['scan', 'compare', 'failure'])
    ap.add_argument('--dataset', default='custom',
                    choices=['custom', 'potsdam'])
    ap.add_argument('--test-img',  required=True)
    ap.add_argument('--test-mask', required=True)
    ap.add_argument('--in-ch',     type=int, default=4)
    ap.add_argument('--output-dir',default='fig/paper_figures')
    ap.add_argument('--images',    nargs='+', default=None,
                    help='compare 模式：手动指定图像名（不含扩展名）')
    ap.add_argument('--draw-box',  action='store_true',
                    help='compare 模式：用红框标出关键差异区域')
    ap.add_argument('--box-size',  type=int, default=100,
                    help='红框边长像素（默认100）')
    ap.add_argument('--n-scan',    type=int, default=120,
                    help='scan 模式扫描图片数（默认120）')
    ap.add_argument('--n-failure', type=int, default=4,
                    help='失败案例数量（默认4）')
    ap.add_argument('--dpi',       type=int, default=300)
    args = ap.parse_args()
 
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
 
    ckpts = (CUSTOM_CHECKPOINTS if args.dataset == 'custom'
             else POTSDAM_CHECKPOINTS)
    tag   = 'Custom' if args.dataset == 'custom' else 'Potsdam'
 
    print('\nLoading models...')
    nets  = load_models(ckpts, args.in_ch, device)
    avail = [m for m in MODEL_ORDER if m in nets]
    print(f'Models ready: {avail}\n')
    if not nets:
        print('ERROR: No models loaded. Check checkpoint paths.')
        return
 
    specify = args.images if args.mode == 'compare' and args.images else None
 
    print('Loading samples...')
    samples = load_samples(
        args.test_img, args.test_mask, nets, device,
        specify=specify,
        max_scan=args.n_scan,
    )
    print(f'  Loaded {len(samples)} samples\n')
    if not samples:
        print('ERROR: No samples loaded.')
        return
 
    if args.mode == 'scan':
        mode_scan(samples, out)
 
    elif args.mode == 'compare':
        if not args.images:
            print('ERROR: --mode compare requires --images <name1> <name2> ...')
            print('  Run --mode scan first to get candidate names.')
            return
        suffix   = '_box' if args.draw_box else ''
        out_path = out / f'figure5_{tag}{suffix}.png'
        mode_compare(samples, avail, str(out_path),
                     draw_box=args.draw_box,
                     box_size=args.box_size,
                     dpi=args.dpi)
 
    elif args.mode == 'failure':
        mode_failure(samples, avail, out,
                     n_cases=args.n_failure,
                     dpi=args.dpi)
 
    print(f'\nDone. Output: {out}/')
 
 
if __name__ == '__main__':
    main()