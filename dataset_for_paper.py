import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
from glob import glob

# 引用项目中的读取工具
from predict import read_image_any

def create_horizontal_gallery(img_dir, mask_dir, output_path, num_samples=4, seed=42):
    """
    将数据集样本横向排列：第一行原图，第二行标签
    """
    random.seed(seed)
    
    # 1. 获取并筛选图像文件
    img_files = sorted(glob(os.path.join(img_dir, "*.tif")) + glob(os.path.join(img_dir, "*.tiff")))
    if not img_files:
        img_files = sorted(glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpg")))
        
    if len(img_files) < num_samples:
        num_samples = len(img_files)

    # 随机采样
    selected_indices = random.sample(range(len(img_files)), num_samples)
    
    # 2. 设置绘图布局 (2行 n列)
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    plt.rcParams['font.family'] = 'serif'
    
    # 如果只有一个样本，确保 axes 是二维的
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(selected_indices):
        img_path = img_files[idx]
        img_name = Path(img_path).stem
        
        # 查找对应 Mask
        mask_path = None
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            candidate = Path(mask_dir) / f"{img_name}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if not mask_path:
            continue

        # 读取数据 (使用项目内置 read_image_any)
        img_np = read_image_any(str(img_path))
        mask_np = read_image_any(str(mask_path))

        # --- 图像预处理 (针对遥感4波段) ---
        # 提取 RGB 并在通道上进行 2%-98% 归一化以增强对比度
        if img_np.ndim == 3 and img_np.shape[2] >= 3:
            vis_img = img_np[:, :, :3].astype(np.float32)
        else:
            vis_img = np.stack([img_np]*3, axis=-1).astype(np.float32)
            
        for j in range(vis_img.shape[2]):
            p_low, p_high = np.percentile(vis_img[:,:,j], (2, 98))
            vis_img[:,:,j] = np.clip((vis_img[:,:,j] - p_low) / (p_high - p_low + 1e-8), 0, 1)

        # --- Mask 预处理 ---
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        vis_mask = (mask_np > 0).astype(np.uint8) * 255

        # --- 绘图 ---
        # 第一行：输入图像
        axes[0, i].imshow(vis_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("Input Image", fontsize=14, fontweight='bold', labelpad=20)
            # 在侧边显示行标题
            axes[0, i].text(-0.1, 0.5, "Input Image", transform=axes[0, i].transAxes, 
                           rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
        
        # 第二行：Ground Truth
        axes[1, i].imshow(vis_mask, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].text(-0.1, 0.5, "Ground Truth", transform=axes[1, i].transAxes, 
                           rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 横向排列示例图已保存至: {output_path}")

if __name__ == "__main__":
    # 路径根据你的 actual 环境修改
    TRAIN_IMG = "/home/lucianlu/data/data_potsdam/val/images/"
    TRAIN_MASK = "/home/lucianlu/data/data_potsdam/val/labels/"
    OUTPUT = "fig/potsdam_samples.png"
    
    create_horizontal_gallery(TRAIN_IMG, TRAIN_MASK, OUTPUT, num_samples=5)