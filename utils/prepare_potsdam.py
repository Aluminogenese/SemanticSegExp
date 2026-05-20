"""
ISPRS Potsdam 数据集预处理脚本
根据实际目录结构修正版

实际目录结构：
  src/
    4_Ortho_RGBIR/
      top_potsdam_2_10_RGBIR.tif   ← 直接就是4波段RGBIR（IR,R,G,B顺序）
      top_potsdam_2_10_RGBIR.tfw
      ...
    5_Labels_all/
      top_potsdam_2_10_label.tif   ← 完整标注（含所有tile）
      ...
    5_Labels_for_participants/
      top_potsdam_2_10_label.tif   ← 部分tile的标注（无遮挡区域）
      ...

颜色标注体系（RGB）：
  [255, 255, 255] = Impervious surfaces (不透水面)
  [0,   0,   255] = Building           ← 我们需要提取的
  [0,   255, 255] = Low vegetation
  [0,   255,   0] = Tree
  [255, 255,   0] = Car
  [255,   0,   0] = Clutter/Background

波段顺序（4_Ortho_RGBIR）：
  RGBIR格式，但Potsdam实际存储是 IR-R-G-B 或 R-G-B-IR，需要检测
  本脚本会自动检测并输出为 R-G-B-IR 顺序（与你的模型一致）

使用方法:
  python prepare_potsdam.py \
      --src /path/to/ISPRS_Potsdam/src \
      --dst ./data_potsdam \
      --crop-size 1024 \
      --overlap 128 \
      --min-building-ratio 0.02 \
      --use-all-labels   # 使用5_Labels_all（含所有tile），否则用5_Labels_for_participants
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm


# ============================================================
# 颜色常量
# ============================================================
BUILDING_COLOR = np.array([0, 0, 255], dtype=np.uint8)     # 建筑物（蓝色）


# ============================================================
# 工具函数
# ============================================================

def read_rgbir_image(tif_path):
    """
    读取Potsdam的RGBIR图像
    
    Potsdam的4_Ortho_RGBIR文件波段顺序是 IR-R-G-B（4波段，uint16）
    我们重新排列为 R-G-B-IR 以与模型输入约定一致
    
    返回：np.ndarray [H, W, 4] uint16，顺序为RGBIR
    """
    img = tifffile.imread(str(tif_path))  # 可能是 [4,H,W] 或 [H,W,4]
    
    if img.ndim == 3 and img.shape[0] == 4:
        # [4, H, W] → [H, W, 4]
        img = img.transpose(1, 2, 0)
    elif img.ndim == 3 and img.shape[2] == 4:
        pass  # 已经是 [H, W, 4]
    else:
        raise ValueError(f"Unexpected image shape: {img.shape} in {tif_path}")
    
    # Potsdam RGBIR文件实际波段顺序是 IR, R, G, B
    # 重排为 R, G, B, IR（与你的模型约定一致）
    ir = img[:, :, 0:1]
    r  = img[:, :, 1:2]
    g  = img[:, :, 2:3]
    b  = img[:, :, 3:4]
    rgbir = np.concatenate([r, g, b, ir], axis=2)  # [H, W, 4] RGBIR
    
    return rgbir


def read_label_image(label_path):
    """
    读取标注图像，提取建筑物二值mask
    
    标注是RGB彩色图像，建筑物为蓝色 [0, 0, 255]
    返回：np.ndarray [H, W] uint8，255=建筑物，0=非建筑物
    """
    label = tifffile.imread(str(label_path))
    
    if label.ndim == 3 and label.shape[0] == 3:
        label = label.transpose(1, 2, 0)  # [3,H,W] → [H,W,3]
    
    if label.ndim == 3 and label.shape[2] > 3:
        label = label[:, :, :3]  # 只取RGB
    
    # 提取建筑物：蓝色 [0, 0, 255]
    building_mask = (
        (label[:, :, 0] == 0) &
        (label[:, :, 1] == 0) &
        (label[:, :, 2] == 255)
    ).astype(np.uint8) * 255
    
    return building_mask


def crop_tiles(image, mask, crop_size=1024, overlap=128, min_building_ratio=0.02):
    """
    将大图（通常6000×6000）裁剪为小块
    
    参数:
        image: [H, W, C] numpy array
        mask:  [H, W]    numpy array (0/255)
        crop_size: 裁剪尺寸
        overlap: 相邻块的重叠像素（减少边界效应）
        min_building_ratio: 至少包含这么多比例的建筑物像素才保留
    
    返回:
        list of (img_crop, mask_crop, row, col)
    """
    h, w = image.shape[:2]
    stride = crop_size - overlap
    crops = []
    
    rows = list(range(0, h - crop_size + 1, stride))
    cols = list(range(0, w - crop_size + 1, stride))
    
    # 确保能覆盖右边缘和下边缘
    if rows and rows[-1] + crop_size < h:
        rows.append(h - crop_size)
    if cols and cols[-1] + crop_size < w:
        cols.append(w - crop_size)
    
    for r in rows:
        for c in cols:
            img_crop  = image[r:r+crop_size, c:c+crop_size]
            mask_crop = mask[r:r+crop_size, c:c+crop_size]
            
            # 过滤建筑物比例过低的块
            building_ratio = mask_crop.sum() / (crop_size * crop_size * 255.0)
            if building_ratio >= min_building_ratio:
                crops.append((img_crop, mask_crop, r, c))
    
    return crops


def find_matching_pairs(rgbir_dir, label_dir):
    """
    匹配RGBIR影像和标注文件
    
    RGBIR文件名: top_potsdam_X_XX_RGBIR.tif
    Label文件名: top_potsdam_X_XX_label.tif
    
    返回: list of (rgbir_path, label_path, tile_id)
    """
    rgbir_dir = Path(rgbir_dir)
    label_dir = Path(label_dir)
    
    pairs = []
    
    # 找所有RGBIR文件
    rgbir_files = sorted(rgbir_dir.glob('top_potsdam_*_RGBIR.tif'))
    
    for rgbir_path in rgbir_files:
        # 提取tile_id: top_potsdam_2_10_RGBIR → top_potsdam_2_10
        stem = rgbir_path.stem  # top_potsdam_2_10_RGBIR
        tile_id = stem.replace('_RGBIR', '')  # top_potsdam_2_10
        
        # 寻找对应标注
        label_path = label_dir / f'{tile_id}_label.tif'
        
        if label_path.exists():
            pairs.append((rgbir_path, label_path, tile_id))
        else:
            print(f"  [WARN] No label found for {tile_id}, skipping")
    
    return pairs


def compute_and_save_stats(img_dir, output_path='stats_potsdam.json', num_samples=50):
    """
    计算数据集统计信息（2%和98%百分位数）
    用于归一化（替换原有的stats.json）
    """
    img_files = sorted(Path(img_dir).glob('*.tif'))
    random.shuffle(img_files)
    img_files = img_files[:num_samples]
    
    print(f"\nComputing dataset statistics from {len(img_files)} samples...")
    
    all_pixels = []
    for f in tqdm(img_files, desc="Sampling pixels"):
        img = tifffile.imread(str(f))
        if img.ndim == 3 and img.shape[0] == 4:
            img = img.transpose(1, 2, 0)
        # 随机采样像素，避免内存溢出
        h, w = img.shape[:2]
        n_sample = min(10000, h * w)
        idx = np.random.choice(h * w, n_sample, replace=False)
        pixels = img.reshape(-1, img.shape[-1])[idx]
        all_pixels.append(pixels.astype(np.float32))
    
    all_pixels = np.concatenate(all_pixels, axis=0)
    
    p_low  = np.percentile(all_pixels, 2,  axis=0).tolist()
    p_high = np.percentile(all_pixels, 98, axis=0).tolist()
    mean   = all_pixels.mean(axis=0).tolist()
    std    = all_pixels.std(axis=0).tolist()
    
    stats = {
        'p_low':  p_low,
        'p_high': p_high,
        'mean':   mean,
        'std':    std,
        'note':   'Potsdam dataset, band order: R-G-B-IR'
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Stats saved to {output_path}")
    print(f"  p_low  (2%):  {[f'{v:.1f}' for v in p_low]}")
    print(f"  p_high (98%): {[f'{v:.1f}' for v in p_high]}")
    print(f"  mean:         {[f'{v:.1f}' for v in mean]}")
    return stats


# ============================================================
# 主函数
# ============================================================

def prepare_potsdam_dataset(
    src_root,
    dst_root,
    crop_size=1024,
    overlap=128,
    min_building_ratio=0.02,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    use_all_labels=True,  # True=用5_Labels_all，False=用5_Labels_for_participants
    seed=42
):
    """
    完整的Potsdam数据集准备流程
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    np.random.seed(seed)
    
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    # 确定标注目录
    rgbir_dir  = src_root / '4_Ortho_RGBIR'
    label_dir  = src_root / ('5_Labels_all' if use_all_labels
                              else '5_Labels_for_participants')
    
    print(f"=" * 60)
    print(f"ISPRS Potsdam Dataset Preparation")
    print(f"=" * 60)
    print(f"  RGBIR dir:  {rgbir_dir}")
    print(f"  Label dir:  {label_dir}")
    print(f"  Output dir: {dst_root}")
    print(f"  Crop size:  {crop_size} × {crop_size}")
    print(f"  Overlap:    {overlap}")
    print(f"  Min building ratio: {min_building_ratio:.1%}")
    print()
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (dst_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 匹配文件对
    pairs = find_matching_pairs(rgbir_dir, label_dir)
    print(f"Found {len(pairs)} matched tile pairs")
    
    if len(pairs) == 0:
        print("ERROR: No matched pairs found! Check your directory structure.")
        return
    
    # 打印找到的tiles
    for _, _, tile_id in pairs:
        print(f"  ✓ {tile_id}")
    print()
    
    # 处理所有tiles，收集裁剪块
    all_crops = []
    
    for rgbir_path, label_path, tile_id in tqdm(pairs, desc="Processing tiles"):
        try:
            print(f"  Reading {tile_id}...")
            
            # 读取影像（重排为RGBIR顺序）
            image = read_rgbir_image(rgbir_path)
            
            # 读取标注（提取建筑物mask）
            mask = read_label_image(label_path)
            
            print(f"    Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"    Mask shape:  {mask.shape}")
            building_pct = mask.sum() / (mask.size * 255) * 100
            print(f"    Building coverage: {building_pct:.1f}%")
            
            # 裁剪
            crops = crop_tiles(
                image, mask,
                crop_size=crop_size,
                overlap=overlap,
                min_building_ratio=min_building_ratio
            )
            
            print(f"    Generated {len(crops)} valid crops")
            
            for img_crop, mask_crop, r, c in crops:
                all_crops.append({
                    'image':   img_crop,
                    'mask':    mask_crop,
                    'tile_id': tile_id,
                    'pos':     f'r{r}_c{c}'
                })
        
        except Exception as e:
            print(f"  ERROR processing {tile_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTotal valid crops: {len(all_crops)}")
    
    if len(all_crops) == 0:
        print("ERROR: No crops generated! Check min_building_ratio or tile content.")
        return
    
    # 按tile划分（避免同一tile的裁剪块出现在不同集合中，防止数据泄露）
    unique_tiles = list(set(c['tile_id'] for c in all_crops))
    random.shuffle(unique_tiles)
    
    n = len(unique_tiles)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))
    
    train_tiles = set(unique_tiles[:n_train])
    val_tiles   = set(unique_tiles[n_train:n_train + n_val])
    test_tiles  = set(unique_tiles[n_train + n_val:])
    
    print(f"\nTile split (by tile, not by crop to avoid data leakage):")
    print(f"  Train tiles ({len(train_tiles)}): {sorted(train_tiles)}")
    print(f"  Val   tiles ({len(val_tiles)}):   {sorted(val_tiles)}")
    print(f"  Test  tiles ({len(test_tiles)}):  {sorted(test_tiles)}")
    
    # 按tile划分crops
    split_crops = {'train': [], 'val': [], 'test': []}
    for crop in all_crops:
        tid = crop['tile_id']
        if tid in train_tiles:
            split_crops['train'].append(crop)
        elif tid in val_tiles:
            split_crops['val'].append(crop)
        else:
            split_crops['test'].append(crop)
    
    for split, crops in split_crops.items():
        print(f"  {split}: {len(crops)} crops")
    
    # 保存crops
    print("\nSaving crops...")
    for split_name, crops in split_crops.items():
        print(f"  Saving {split_name}...")
        for idx, crop_data in enumerate(tqdm(crops, desc=f"  {split_name}")):
            filename = f"{crop_data['tile_id']}_{crop_data['pos']}_{idx:05d}"
            
            # 保存图像（tiff格式，CHW顺序，保持uint16）
            img_out = crop_data['image'].transpose(2, 0, 1)  # [H,W,4] → [4,H,W]
            img_path = dst_root / split_name / 'images' / f'{filename}.tif'
            tifffile.imwrite(str(img_path), img_out)
            
            # 保存mask（PNG，二值）
            mask_path = dst_root / split_name / 'labels' / f'{filename}.png'
            Image.fromarray(crop_data['mask']).save(str(mask_path))
    
    # 计算统计信息（用于归一化）
    print("\nComputing normalization statistics...")
    compute_and_save_stats(
        dst_root / 'train' / 'images',
        output_path=str(dst_root / 'stats_potsdam.json')
    )
    
    # 保存数据集信息
    info = {
        'dataset':            'ISPRS_Potsdam',
        'band_order':         'R-G-B-IR',
        'image_dtype':        'uint16',
        'crop_size':          crop_size,
        'overlap':            overlap,
        'min_building_ratio': min_building_ratio,
        'total_crops':        len(all_crops),
        'splits': {
            s: {
                'n_crops': len(split_crops[s]),
                'tiles':   list({c['tile_id'] for c in split_crops[s]})
            }
            for s in ['train', 'val', 'test']
        }
    }
    with open(dst_root / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"  Output: {dst_root}")
    print(f"  Train:  {len(split_crops['train'])} crops")
    print(f"  Val:    {len(split_crops['val'])} crops")
    print(f"  Test:   {len(split_crops['test'])} crops")
    print(f"  Stats:  {dst_root}/stats_potsdam.json")
    print(f"  → Copy stats_potsdam.json to your project root as stats.json")
    print(f"    cp {dst_root}/stats_potsdam.json ./stats.json")
    print(f"{'='*60}")


def verify_dataset(dst_root):
    """验证生成的数据集"""
    dst_root = Path(dst_root)
    print("\nVerifying dataset...")
    
    for split in ['train', 'val', 'test']:
        img_dir  = dst_root / split / 'images'
        mask_dir = dst_root / split / 'labels'
        
        img_files  = sorted(img_dir.glob('*.tif'))
        mask_files = sorted(mask_dir.glob('*.png'))
        
        print(f"\n  {split}:")
        print(f"    Images: {len(img_files)}")
        print(f"    Masks:  {len(mask_files)}")
        
        # 检查配对
        img_stems  = {f.stem for f in img_files}
        mask_stems = {f.stem for f in mask_files}
        missing_masks  = img_stems - mask_stems
        missing_images = mask_stems - img_stems
        
        if missing_masks:
            print(f"    ⚠ Missing masks: {len(missing_masks)}")
        if missing_images:
            print(f"    ⚠ Missing images: {len(missing_images)}")
        if not missing_masks and not missing_images:
            print(f"    ✓ All paired correctly")
        
        # 检查第一个文件
        if img_files:
            img = tifffile.imread(str(img_files[0]))
            print(f"    Sample image shape: {img.shape}, dtype: {img.dtype}")
        if mask_files:
            mask = np.array(Image.open(str(mask_files[0])))
            building_pct = mask.sum() / (mask.size * 255) * 100
            print(f"    Sample mask shape:  {mask.shape}, "
                  f"building coverage: {building_pct:.1f}%")


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare ISPRS Potsdam dataset for MS-HRNet training'
    )
    parser.add_argument('--src', default='src',
                        help='Potsdam dataset root (contains 4_Ortho_RGBIR/, 5_Labels_all/)')
    parser.add_argument('--dst', default='data_potsdam',
                        help='Output directory (default: data_potsdam)')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='Crop size in pixels (default: 1024)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap between adjacent crops (default: 128)')
    parser.add_argument('--min-building-ratio', type=float, default=0.02,
                        help='Min building pixel ratio to keep a crop (default: 0.02)')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio',   type=float, default=0.15)
    parser.add_argument('--test-ratio',  type=float, default=0.15)
    parser.add_argument('--use-all-labels', action='store_true', default=True,
                        help='Use 5_Labels_all (more tiles) vs 5_Labels_for_participants')
    parser.add_argument('--use-participant-labels', action='store_true',
                        help='Use 5_Labels_for_participants instead of 5_Labels_all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing dataset, skip preparation')
    
    args = parser.parse_args()
    
    use_all = not args.use_participant_labels
    
    if args.verify_only:
        verify_dataset(args.dst)
    else:
        prepare_potsdam_dataset(
            src_root=args.src,
            dst_root=args.dst,
            crop_size=args.crop_size,
            overlap=args.overlap,
            min_building_ratio=args.min_building_ratio,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            use_all_labels=use_all,
            seed=args.seed
        )
        verify_dataset(args.dst)
