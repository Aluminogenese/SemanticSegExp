"""
gen_grid_figures.py
论文图生成脚本 — 3x3 网格高清多行版 (带自动寻找差异红框)
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
        'HRNet':           lambda: HRNet(in_channels=in_ch, num_classes=1, base_channels=48),
        'MS-HRNet (Ours)': lambda: MSHRNet(in_channels=in_ch, num_classes=1, base_channels=48),
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
# 推理 & 指标 (保持原样)
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
        nt   = int(np.sum((binary_dilation(gb, iterations=3) & pb) | (binary_dilation(pb, iterations=3) & gb)))
        biou = nt / (int(np.sum(pb)) + int(np.sum(gb)) + eps)
    else:
        biou = 0.0
    return dict(iou=iou, precision=prec, recall=rec, f1=f1, biou=float(biou))


# =============================================================================
# 工具函数 (保持原样)
# =============================================================================
 
def to_rgb(img_np):
    arr = (img_np[:, :, :3] if img_np.ndim == 3 else np.stack([img_np]*3, axis=-1)).astype(np.float32)
    for c in range(3):
        lo, hi = np.percentile(arr[:, :, c], [2, 98])
        arr[:, :, c] = np.clip((arr[:, :, c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return arr
 
def classify_scene(gt, preds_dict):
    cov = float(gt.sum()) / gt.size
    _, n_comp = sp_label(gt)
    iou_vals = []
    for p in preds_dict.values():
        tp = np.sum(p & gt); fp = np.sum(p & ~gt); fn = np.sum(~p & gt)
        iou_vals.append(float(tp / (tp + fp + fn + 1e-8)))
    iou_std = float(np.std(iou_vals)) if iou_vals else 0.0
    if cov < 0.08: return 'single'
    elif cov > 0.35 or (n_comp > 15 and cov > 0.15): return 'dense'
    elif iou_std > 0.06: return 'complex'
    else: return 'boundary'
 
def find_mask_path(stem, mask_dir):
    for ext in ['.png', '.tif', '.tiff', '.jpg']:
        p = Path(mask_dir) / f'{stem}{ext}'
        if p.exists(): return p
    return None
 
def build_sample(img_path, mask_dir, nets, device):
    stem = Path(img_path).stem
    mp   = find_mask_path(stem, mask_dir)
    if mp is None: return None
    img    = read_image_any(str(img_path))
    gt_raw = read_image_any(str(mp))
    gt     = (gt_raw[:, :, 0] if gt_raw.ndim == 3 else gt_raw) > 0
    preds  = {name: infer(net, img, device) for name, net in nets.items()}
    mets   = {name: calc_metrics(p, gt) for name, p in preds.items()}
    cov    = float(gt.sum()) / gt.size
    scene  = classify_scene(gt, preds)
    our    = mets.get('MS-HRNet (Ours)', {})
    base   = mets.get('HRNet', {})
    score  = ((our.get('iou', 0) - base.get('iou', 0)) * our.get('biou', 0) * min(cov / 0.3, 1.0))
    return dict(name=stem, img=img, gt=gt, preds=preds, metrics=mets, coverage=cov, scene=scene, score=score)
 
def load_samples(img_dir, mask_dir, nets, device, specify=None, max_scan=120):
    files = sorted(glob(f'{img_dir}/*.tif') + glob(f'{img_dir}/*.tiff') + glob(f'{img_dir}/*.png') + glob(f'{img_dir}/*.jpg'))
    print(f'  Total images: {len(files)}')
    if specify:
        fmap  = {Path(f).stem: f for f in files}
        files = [fmap[s] for s in specify if s in fmap]
        miss  = [s for s in specify if s not in fmap]
        if miss: print(f'  [WARN] not found: {miss}')
        print(f'  Loading {len(files)} specified images...')
    else:
        random.seed(42)
        files = random.sample(files, min(max_scan, len(files)))
        print(f'  Scanning {len(files)} images...')
    samples = []
    for f in tqdm(files, desc='  Processing', leave=False):
        s = build_sample(f, mask_dir, nets, device)
        if s is not None: samples.append(s)
    return samples
 

# =============================================================================
# 模式 2：compare (加入自动寻找差异最大区域并画框的逻辑)
# =============================================================================
 
def best_diff_box(pred_ours, pred_base, gt, box_size):
    """自动寻找 MS-HRNet 正确且 HRNet 错误最集中的区域"""
    h, w    = gt.shape
    # improve: Ours是对的(+1)，Base是错的(-0.something) 综合判断
    improve = ((pred_ours == gt).astype(float) - (pred_base == gt).astype(float))
    density = uniform_filter(improve, size=box_size)
    half    = box_size // 2
    tmp     = density.copy()
    # 避免框跑到图像边缘外
    tmp[:half, :] = tmp[-half:, :] = -1
    tmp[:, :half] = tmp[:, -half:] = -1
    cy, cx  = np.unravel_index(np.argmax(tmp), tmp.shape)
    
    # 如果最大差异也很小，就不画框了
    if tmp[cy, cx] < 0.02: 
        return None
        
    y1 = max(0, cy - half); y2 = min(h, y1 + box_size)
    x1 = max(0, cx - half); x2 = min(w, x1 + box_size)
    if y2 - y1 < box_size: y1 = max(0, y2 - box_size)
    if x2 - x1 < box_size: x1 = max(0, x2 - box_size)
    return (y1, x1, y2, x2)
 
def add_red_box(ax, roi, lw=2.5):
    y1, x1, y2, x2 = roi
    # 调整了线宽和颜色，使其在论文中更加醒目
    ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, lw=lw, edgecolor='#FF0000', facecolor='none', zorder=10))
 
def mode_compare(samples, model_names, output_dir, draw_box=False, box_size=120, dpi=300):
    out_path_base = Path(output_dir)

    # 生成底层标签，记录需要画红框的子图索引
    col_labels = ['(a) Images', '(b) Labels']
    draw_box_indices = [0, 1]  # 默认在原图和GT上画框
    
    for i, name in enumerate(model_names):
        letter = chr(ord('c') + i)
        short  = name.replace(' (Ours)', '')
        suffix = ' (Ours)' if 'Ours' in name else ''
        col_labels.append(f'({letter}) {short}{suffix}')
        
        # 记录 Baseline 和 Ours 的索引，以便同步画红框对比
        if name == 'HRNet':
            draw_box_indices.append(i + 2)
        elif name == 'MS-HRNet (Ours)':
            draw_box_indices.append(i + 2)

    n_rows, n_cols = 3, 3
    cell = 3.5  

    for s in samples:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(cell * n_cols, cell * n_rows),
            gridspec_kw=dict(hspace=0.25, wspace=0.05,
                             left=0.02, right=0.98,
                             top=0.95,  bottom=0.05)
        )
        axes = axes.flatten()

        vis   = to_rgb(s['img'])
        gt    = s['gt']
        preds = s['preds']

        col_data = [vis, gt.astype(float)]
        col_rgb  = [True, False]
        for mn in model_names:
            p = preds.get(mn, np.zeros_like(gt, dtype=float)).astype(float)
            col_data.append(p)
            col_rgb.append(False)

        # 核心：计算出最优红框位置
        box = None
        if draw_box and 'MS-HRNet (Ours)' in preds and 'HRNet' in preds:
            box = best_diff_box(preds['MS-HRNet (Ours)'], preds['HRNet'], gt, box_size)

        for ci in range(9):
            ax = axes[ci]
            if ci < len(col_data):
                data = col_data[ci]
                is_rgb = col_rgb[ci]
                if is_rgb:
                    ax.imshow(data, interpolation='bilinear')
                else:
                    ax.imshow(data, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
                
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color('black')
                    sp.set_linewidth(1.0)

                ax.set_xlabel(col_labels[ci], fontsize=15, labelpad=8)

                # 在指定的子图上画出对比红框
                if box is not None and ci in draw_box_indices:
                    add_red_box(ax, box)
            else:
                ax.axis('off')

        suffix = '_box' if draw_box else ''
        out_file = out_path_base / f"figure_3x3_{s['name']}{suffix}.png"
        
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close(fig)
        print(f'  [Success] Saved grid for {s["name"]} -> {out_file}')


# =============================================================================
# 主函数
# =============================================================================
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='compare', choices=['compare'])
    ap.add_argument('--dataset', default='custom', choices=['custom', 'potsdam'])
    ap.add_argument('--test-img',  required=True)
    ap.add_argument('--test-mask', required=True)
    ap.add_argument('--in-ch',     type=int, default=4)
    ap.add_argument('--output-dir',default='fig/paper_figures')
    ap.add_argument('--images',    nargs='+', required=True, help='需指定的图片名 (如 000000532 000000281)')
    ap.add_argument('--draw-box',  action='store_true', help='开启自动寻找差异并画红框功能')
    ap.add_argument('--box-size',  type=int, default=120, help='红框大小 (默认120像素)')
    ap.add_argument('--dpi',       type=int, default=300)
    args = ap.parse_args()
 
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
 
    ckpts = CUSTOM_CHECKPOINTS if args.dataset == 'custom' else POTSDAM_CHECKPOINTS
    
    print('\nLoading models...')
    nets  = load_models(ckpts, args.in_ch, device)
    avail = [m for m in MODEL_ORDER if m in nets]
    print(f'Models ready: {avail}\n')
    if not nets:
        print('ERROR: No models loaded.')
        return
 
    print('Loading samples...')
    samples = load_samples(args.test_img, args.test_mask, nets, device, specify=args.images)
    if not samples:
        print('ERROR: No samples loaded.')
        return
 
    # 传入 box_size 和 draw_box 参数
    mode_compare(samples, avail, out, draw_box=args.draw_box, box_size=args.box_size, dpi=args.dpi)
    print(f'\nDone. Output: {out}/')
 
if __name__ == '__main__':
    main()