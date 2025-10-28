"""
数据集划分工具
将 /mnt/U/Dat_Seg/4bands/ 中的数据划分为 train/val/test

使用方法:
python data_split.py
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm
import json


def split_dataset(
    src_img_dir='/mnt/U/Dat_Seg/4bands/images/',
    src_mask_dir='/mnt/U/Dat_Seg/4bands/labels/',
    dst_root='/mnt/U/Dat_Seg/4bands_split/',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    copy_files=True  # False则创建软链接，节省空间
):
    """
    划分数据集为 train/val/test
    
    参数:
        src_img_dir: 源图像目录
        src_mask_dir: 源mask目录
        dst_root: 目标根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        copy_files: True复制文件，False创建软链接（节省空间）
    """
    
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建目录结构
    dst_root = Path(dst_root)
    splits = ['train', 'val', 'test']
    
    for split in splits:
        (dst_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    src_img_dir = Path(src_img_dir)
    img_files = list(src_img_dir.glob('*.tif')) + list(src_img_dir.glob('*.tiff'))
    
    if not img_files:
        img_files = list(src_img_dir.glob('*.png')) + list(src_img_dir.glob('*.jpg'))
    
    print(f"找到 {len(img_files)} 张图像")
    
    # 提取文件名（不含扩展名）
    file_ids = [f.stem for f in img_files]
    
    # 随机打乱
    random.shuffle(file_ids)
    
    # 计算划分点
    n_total = len(file_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # 划分
    train_ids = file_ids[:n_train]
    val_ids = file_ids[n_train:n_train+n_val]
    test_ids = file_ids[n_train+n_val:]
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_ids)} ({len(train_ids)/n_total*100:.1f}%)")
    print(f"  验证集: {len(val_ids)} ({len(val_ids)/n_total*100:.1f}%)")
    print(f"  测试集: {len(test_ids)} ({len(test_ids)/n_total*100:.1f}%)")
    
    # 复制或链接文件
    splits_data = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    src_mask_dir = Path(src_mask_dir)
    
    for split, ids in splits_data.items():
        print(f"\n处理 {split} 集...")
        
        for file_id in tqdm(ids):
            # 查找图像文件
            img_src = None
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                candidate = src_img_dir / f"{file_id}{ext}"
                if candidate.exists():
                    img_src = candidate
                    break
            
            if img_src is None:
                print(f"警告: 找不到图像 {file_id}")
                continue
            
            # 查找mask文件
            mask_src = None
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                candidate = src_mask_dir / f"{file_id}{ext}"
                if candidate.exists():
                    mask_src = candidate
                    break
            
            if mask_src is None:
                print(f"警告: 找不到mask {file_id}")
                continue
            
            # 目标路径
            img_dst = dst_root / split / 'images' / img_src.name
            mask_dst = dst_root / split / 'labels' / mask_src.name
            
            # 复制或链接
            if copy_files:
                shutil.copy2(img_src, img_dst)
                shutil.copy2(mask_src, mask_dst)
            else:
                # 创建软链接（节省空间）
                if not img_dst.exists():
                    os.symlink(img_src.absolute(), img_dst)
                if not mask_dst.exists():
                    os.symlink(mask_src.absolute(), mask_dst)
    
    # 保存划分信息
    split_info = {
        'total': n_total,
        'train': len(train_ids),
        'val': len(val_ids),
        'test': len(test_ids),
        'seed': seed,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids
    }
    
    with open(dst_root / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ 数据划分完成!")
    print(f"   输出目录: {dst_root}")
    print(f"   划分信息已保存到: {dst_root / 'split_info.json'}")
    
    return split_info


def verify_split(dst_root='/mnt/U/Dat_Seg/4bands_split/'):
    """验证数据划分的完整性"""
    dst_root = Path(dst_root)
    
    print("\n验证数据划分...")
    
    for split in ['train', 'val', 'test']:
        img_dir = dst_root / split / 'images'
        mask_dir = dst_root / split / 'labels'
        
        img_files = list(img_dir.glob('*.tif')) + list(img_dir.glob('*.tiff')) + \
                   list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
        mask_files = list(mask_dir.glob('*.tif')) + list(mask_dir.glob('*.tiff')) + \
                    list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.jpg'))
        
        print(f"\n{split}:")
        print(f"  图像: {len(img_files)}")
        print(f"  Mask: {len(mask_files)}")
        
        # 检查是否配对
        img_ids = {f.stem for f in img_files}
        mask_ids = {f.stem for f in mask_files}
        
        missing_masks = img_ids - mask_ids
        missing_images = mask_ids - img_ids
        
        if missing_masks:
            print(f"  ⚠️  缺少mask: {len(missing_masks)}")
        if missing_images:
            print(f"  ⚠️  缺少图像: {len(missing_images)}")
        
        if not missing_masks and not missing_images:
            print(f"  ✅ 数据完整")


def create_file_lists(dst_root='/mnt/U/Dat_Seg/4bands_split/'):
    """创建文件列表（方便某些训练框架使用）"""
    dst_root = Path(dst_root)
    
    for split in ['train', 'val', 'test']:
        img_dir = dst_root / split / 'images'
        
        img_files = sorted(list(img_dir.glob('*.tif')) + list(img_dir.glob('*.tiff')) + 
                          list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
        
        # 保存文件列表
        list_file = dst_root / f'{split}_list.txt'
        with open(list_file, 'w') as f:
            for img_file in img_files:
                f.write(f"{img_file.name}\n")
        
        print(f"创建文件列表: {list_file}")


def show_split_statistics(dst_root='/mnt/U/Dat_Seg/4bands_split/'):
    """显示数据集统计信息"""
    import numpy as np
    from PIL import Image
    
    dst_root = Path(dst_root)
    
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        print(f"\n【{split.upper()}】")
        
        img_dir = dst_root / split / 'images'
        mask_dir = dst_root / split / 'labels'
        
        img_files = list(img_dir.glob('*.tif')) + list(img_dir.glob('*.tiff')) + \
                   list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
        
        if not img_files:
            continue
        
        # 图像尺寸统计
        sizes = []
        for img_file in img_files[:20]:  # 只统计前20张
            try:
                img = Image.open(img_file)
                sizes.append(img.size)
            except:
                pass
        
        if sizes:
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            print(f"  图像数量: {len(img_files)}")
            print(f"  平均尺寸: {np.mean(widths):.0f} × {np.mean(heights):.0f}")
            print(f"  尺寸范围: {min(widths)}~{max(widths)} × {min(heights)}~{max(heights)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集划分工具')
    parser.add_argument('--src-img', default='/mnt/U/Dat_Seg/4bands/images/',
                       help='源图像目录')
    parser.add_argument('--src-mask', default='/mnt/U/Dat_Seg/4bands/labels/',
                       help='源mask目录')
    parser.add_argument('--dst-root', default='/mnt/U/Dat_Seg/4bands_split/',
                       help='目标根目录')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例 (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='验证集比例 (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='测试集比例 (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--symlink', action='store_true',
                       help='使用软链接而非复制（节省空间）')
    parser.add_argument('--verify-only', action='store_true',
                       help='仅验证已有的划分')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_split(args.dst_root)
        show_split_statistics(args.dst_root)
    else:
        # 执行划分
        split_info = split_dataset(
            src_img_dir=args.src_img,
            src_mask_dir=args.src_mask,
            dst_root=args.dst_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            copy_files=not args.symlink
        )
        
        # 验证
        verify_split(args.dst_root)
        
        # 创建文件列表
        create_file_lists(args.dst_root)
        
        # 显示统计
        show_split_statistics(args.dst_root)