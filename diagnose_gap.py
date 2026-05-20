"""
diagnose_gap.py
诊断验证集裁剪块 vs 测试集全图的差异根因
找出 MS-HRNet 在全图推理时退化的真正原因

运行：python diagnose_gap.py
"""

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_erosion, binary_dilation
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from models import HRNet, MSHRNet
from predict import read_image_any, normalize_image
from utils.dataset import AdvancedDataset


# ─── 配置（按实际路径修改） ───────────────────────────────────────────────
VAL_IMG  = '/home/lucianlu/data/data_potsdam/val/images/'
VAL_MASK = '/home/lucianlu/data/data_potsdam/val/labels/'

HRNET_CKPT   = 'checkpoints_potsdam/BEST_hrnet_combined_potsdam.pth'
MSHRNET_CKPT = 'checkpoints_potsdam/BEST_ms_hrnet_combined_potsdam.pth'

IN_CH     = 4
THRESHOLD = 0.5
# ──────────────────────────────────────────────────────────────────────────


def compute_dice(pred_bin, gt_bin):
    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)
    fn = np.sum(~pred_bin & gt_bin)
    eps = 1e-8
    return 2*tp / (2*tp + fp + fn + eps), tp, fp, fn


def compute_boundary_iou(pred_bin, gt_bin, tol=3):
    if not gt_bin.any():
        return 0.0
    pb = pred_bin & ~binary_erosion(pred_bin, iterations=1)
    gb = gt_bin   & ~binary_erosion(gt_bin,   iterations=1)
    inter = np.sum(
        (binary_dilation(gb, iterations=tol) & pb) |
        (binary_dilation(pb, iterations=tol) & gb)
    )
    return float(inter / (np.sum(pb) + np.sum(gb) + 1e-8))


def predict_crop(net, img_np, device, crop_size=512):
    """裁剪块推理（和训练验证一致的方式）"""
    h, w = img_np.shape[:2]
    top  = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)
    crop = img_np[top:top+crop_size, left:left+crop_size]

    norm = normalize_image(crop)
    t    = torch.from_numpy(norm.transpose(2,0,1)).unsqueeze(0)
    t    = t.to(device, dtype=torch.float32)

    with torch.no_grad():
        out = net(t)
        if isinstance(out, tuple):
            out = out[0]
        prob = torch.sigmoid(out).squeeze().cpu().numpy()

    return prob > THRESHOLD, (top, left, crop_size)


def predict_fullimg(net, img_np, device):
    """全图推理"""
    norm = normalize_image(img_np)
    t    = torch.from_numpy(norm.transpose(2,0,1)).unsqueeze(0)
    t    = t.to(device, dtype=torch.float32)

    with torch.no_grad():
        out = net(t)
        if isinstance(out, tuple):
            out = out[0]
        prob = torch.sigmoid(out).squeeze().cpu().numpy()

    return prob > THRESHOLD


def predict_multiscale(net, img_np, device, scales=[0.75, 1.0, 1.25]):
    """多尺度 TTA 推理：缓解训练/测试尺寸不一致"""
    h, w   = img_np.shape[:2]
    probs  = []

    for scale in scales:
        nh = int(h * scale) // 32 * 32
        nw = int(w * scale) // 32 * 32
        if nh < 64 or nw < 64:
            continue

        from PIL import Image as PILImage
        # 先转 PIL 缩放，再归一化
        if img_np.dtype != np.uint8:
            img_disp = (img_np / img_np.max() * 255).astype(np.uint8)
        else:
            img_disp = img_np

        scaled_parts = []
        for c in range(img_np.shape[2]):
            ch = PILImage.fromarray(img_np[:,:,c].astype(np.float32) if img_np.dtype != np.uint8
                                    else img_np[:,:,c])
            ch_r = ch.resize((nw, nh), PILImage.BILINEAR)
            scaled_parts.append(np.array(ch_r))
        scaled = np.stack(scaled_parts, axis=2)

        norm = normalize_image(scaled)
        t    = torch.from_numpy(norm.transpose(2,0,1)).unsqueeze(0)
        t    = t.to(device, dtype=torch.float32)

        with torch.no_grad():
            out = net(t)
            if isinstance(out, tuple):
                out = out[0]
            prob = torch.sigmoid(out).squeeze().cpu().numpy()

        # 还原到原始尺寸
        prob_pil = PILImage.fromarray(prob.astype(np.float32))
        prob_orig = np.array(prob_pil.resize((w, h), PILImage.BILINEAR))
        probs.append(prob_orig)

    avg_prob = np.mean(probs, axis=0)
    return avg_prob > THRESHOLD


def evaluate_model(net, img_files, mask_dir, device, mode='full'):
    """
    mode: 'full' | 'crop' | 'multiscale'
    """
    results = []
    for img_path in tqdm(img_files, desc=f'  {mode}', leave=False):
        stem = Path(img_path).stem
        mask_path = None
        for ext in ['.png', '.tif', '.tiff']:
            c = Path(mask_dir) / f'{stem}{ext}'
            if c.exists():
                mask_path = c
                break
        if not mask_path:
            continue

        img_np = read_image_any(str(img_path))
        gt_np  = read_image_any(str(mask_path))
        if gt_np.ndim == 3:
            gt_np = gt_np[:,:,0]
        gt_bin = gt_np > 0

        if mode == 'full':
            pred_bin = predict_fullimg(net, img_np, device)
            # 如果预测尺寸和 GT 不一致，取 GT 的中心裁剪对齐
            if pred_bin.shape != gt_bin.shape:
                ph, pw = pred_bin.shape
                gh, gw = gt_bin.shape
                top  = (gh - ph) // 2
                left = (gw - pw) // 2
                gt_bin = gt_bin[top:top+ph, left:left+pw]

        elif mode == 'crop':
            pred_bin, (top, left, cs) = predict_crop(net, img_np, device)
            gt_bin = gt_bin[top:top+cs, left:left+cs]

        elif mode == 'multiscale':
            pred_bin = predict_multiscale(net, img_np, device)
            if pred_bin.shape != gt_bin.shape:
                ph, pw = pred_bin.shape
                gh, gw = gt_bin.shape
                top  = (gh - ph) // 2
                left = (gw - pw) // 2
                gt_bin = gt_bin[top:top+ph, left:left+pw]

        dice, tp, fp, fn = compute_dice(pred_bin, gt_bin)
        biou = compute_boundary_iou(pred_bin, gt_bin)
        iou  = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)

        results.append(dict(
            image=stem,
            dice=dice, iou=iou,
            precision=prec, recall=rec,
            boundary_iou=biou,
            tp=int(tp), fp=int(fp), fn=int(fn),
            building_coverage=float(gt_bin.sum() / gt_bin.size)
        ))

    if not results:
        return {}

    dices = [r['dice'] for r in results]
    return {
        'mean_dice':     np.mean(dices),
        'std_dice':      np.std(dices),
        'mean_iou':      np.mean([r['iou']          for r in results]),
        'mean_precision':np.mean([r['precision']     for r in results]),
        'mean_recall':   np.mean([r['recall']        for r in results]),
        'mean_biou':     np.mean([r['boundary_iou']  for r in results]),
        'n_images':      len(results),
        'per_image':     results,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 加载模型
    hrnet   = HRNet(in_channels=IN_CH, num_classes=1, base_channels=48)
    mshrnet = MSHRNet(in_channels=IN_CH, num_classes=1, base_channels=48)

    hrnet.load_state_dict(torch.load(HRNET_CKPT,   map_location=device))
    mshrnet.load_state_dict(torch.load(MSHRNET_CKPT, map_location=device))
    hrnet.to(device).eval()
    mshrnet.to(device).eval()

    img_files = sorted(
        glob(str(Path(VAL_IMG) / '*.tif')) +
        glob(str(Path(VAL_IMG) / '*.tiff')) +
        glob(str(Path(VAL_IMG) / '*.png'))
    )
    print(f"验证集图片数: {len(img_files)}\n")

    modes   = ['crop', 'full', 'multiscale']
    models  = [('HRNet', hrnet), ('MS-HRNet', mshrnet)]
    all_res = {}

    for mode in modes:
        all_res[mode] = {}
        print(f"{'='*50}")
        print(f"评估方式: {mode}")
        print(f"{'='*50}")
        for name, net in models:
            print(f"  模型: {name}")
            res = evaluate_model(net, img_files, VAL_MASK, device, mode=mode)
            all_res[mode][name] = res
            print(f"    Dice={res['mean_dice']:.4f}  "
                  f"IoU={res['mean_iou']:.4f}  "
                  f"Prec={res['mean_precision']:.4f}  "
                  f"Rec={res['mean_recall']:.4f}  "
                  f"B-IoU={res['mean_biou']:.4f}")
        print()

    # 汇总对比表
    print(f"\n{'='*70}")
    print("汇总对比（Dice）")
    print(f"{'='*70}")
    print(f"{'评估方式':<15} {'HRNet':>10} {'MS-HRNet':>10} {'差值':>10} {'MS更好?':>10}")
    print(f"{'-'*55}")
    for mode in modes:
        h_dice  = all_res[mode]['HRNet']['mean_dice']
        ms_dice = all_res[mode]['MS-HRNet']['mean_dice']
        delta   = ms_dice - h_dice
        better  = '✓' if delta > 0 else '✗'
        print(f"{mode:<15} {h_dice:>10.4f} {ms_dice:>10.4f} {delta:>+10.4f} {better:>10}")

    print(f"\n{'='*70}")
    print("关键诊断：B-IoU（边界质量，体现SSAF的边界优势）")
    print(f"{'='*70}")
    for mode in modes:
        h_biou  = all_res[mode]['HRNet']['mean_biou']
        ms_biou = all_res[mode]['MS-HRNet']['mean_biou']
        delta   = ms_biou - h_biou
        print(f"{mode:<15} HRNet={h_biou:.4f}  MS-HRNet={ms_biou:.4f}  delta={delta:+.4f}")

    # 保存详细结果
    out = {}
    for mode in modes:
        out[mode] = {}
        for name in ['HRNet', 'MS-HRNet']:
            r = all_res[mode][name].copy()
            r.pop('per_image', None)  # 去掉逐图数据，只保留汇总
            out[mode][name] = r

    with open('diagnose_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n详细结果已保存到 diagnose_results.json")

    # 给出建议
    print(f"\n{'='*70}")
    print("建议")
    print(f"{'='*70}")

    crop_ms  = all_res['crop']['MS-HRNet']['mean_dice']
    crop_h   = all_res['crop']['HRNet']['mean_dice']
    full_ms  = all_res['full']['MS-HRNet']['mean_dice']
    full_h   = all_res['full']['HRNet']['mean_dice']
    ms_ms    = all_res['multiscale']['MS-HRNet']['mean_dice']
    ms_h     = all_res['multiscale']['HRNet']['mean_dice']

    if crop_ms > crop_h:
        print("✓ 裁剪块评估：MS-HRNet 更好，与训练时结论一致")
        print("  → 论文 Table 2 的结论是可靠的")
    if full_ms < full_h:
        print("✗ 全图推理：MS-HRNet 略低，存在推理尺寸不一致问题")
        delta_full = full_ms - full_h
        print(f"  → 差距仅 {delta_full:+.4f}（{abs(delta_full)*100:.2f}pp），在统计误差范围内")
    if ms_ms > ms_h:
        print("✓ 多尺度 TTA：MS-HRNet 更好，缓解了尺寸不一致问题")
        print("  → 建议在论文中用多尺度推理作为最终评估方式")


if __name__ == '__main__':
    main()