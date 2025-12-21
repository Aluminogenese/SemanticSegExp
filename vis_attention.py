"""
SSAF注意力图可视化工具

功能：
1. 加载训练好的MS-HRNet模型
2. 对指定图像提取SSAF模块的注意力图
3. 可视化：光谱权重、通道权重、空间注意力、门控权重

使用示例：
python visualize_attention.py \
    --model checkpoints/BEST_ms_hrnet_v2_dat_4bands.pth \
    --image /path/to/test/image.tif \
    --output attention_vis
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image

from models import MSHRNet
from predict import read_image_any, normalize_image


def compute_pca_rgb(feat_map):
    """
    使用 PCA 将多通道特征图降维到 3 通道 RGB 用于可视化
    保留最大方差信息，避免直接截断通道导致的信息丢失
    """
    # feat_map: [C, H, W]
    C, H, W = feat_map.shape
    
    # 展平: [N, C] where N = H*W
    flat_feat = feat_map.reshape(C, -1).T
    
    # 标准化
    mean = flat_feat.mean(axis=0)
    std = flat_feat.std(axis=0) + 1e-8
    flat_feat_norm = (flat_feat - mean) / std
    
    try:
        # 使用 SVD 进行 PCA
        # U: [N, N], S: [K], Vh: [K, C]
        # 我们只需要前3个主成分
        u, s, vh = np.linalg.svd(flat_feat_norm, full_matrices=False)
        
        # 投影到前3个主成分: [N, 3]
        # 注意：u * s 包含了投影后的坐标信息
        pca_feat = u[:, :3] @ np.diag(s[:3])
        
        # 归一化到 [0, 1] 用于显示
        pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min() + 1e-8)
        
        # 重塑回 [H, W, 3]
        return pca_feat.reshape(H, W, 3)
        
    except Exception as e:
        logging.warning(f"PCA visualization failed: {e}. Falling back to mean.")
        # 降级方案：平均值
        mean_map = np.mean(feat_map, axis=0)
        mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)
        return np.stack([mean_map]*3, axis=-1)


def load_model(model_path, in_channels=4, device='cuda'):
    """加载训练好的模型"""
    net = MSHRNet(in_channels=in_channels, num_classes=1, base_channels=48)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device=device)
    net.eval()
    logging.info(f'Model loaded from {model_path}')
    return net


def extract_attention_maps(net, img_path, device):
    """提取SSAF注意力图"""
    # 读取并预处理图像
    img_np = read_image_any(img_path)
    img_normalized = normalize_image(img_np)
    
    # 转换为tensor
    img = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    # 前向传播并提取注意力
    net.eval()
    with torch.no_grad():
        output, attention_maps = net(img)
    
    # 提取注意力图
    spectral_weights = attention_maps['spectral_weights'].cpu().numpy()  # [B, C, 1, 1]
    channel_weights = attention_maps['channel_weights'].cpu().numpy()    # [B, C, 1, 1]
    spatial_weights = attention_maps['spatial_weights'].cpu().numpy()    # [B, 1, H, W]
    gate_weights = attention_maps['gate_weights'].cpu().numpy()          # [B, C, 1, 1]
    temperature = attention_maps['temperature']
    
    # 获取 SSAF 输出特征图
    if 'ssaf_output' in attention_maps:
        ssaf_output = attention_maps['ssaf_output'].cpu().numpy()        # [B, C, H, W]
    else:
        ssaf_output = np.zeros_like(img.cpu().numpy())
    
    # 预测结果
    pred = torch.sigmoid(output).squeeze().cpu().numpy()
    
    return {
        'image': img_np,
        'prediction': pred,
        'spectral_weights': spectral_weights.squeeze(),      # [C]
        'channel_weights': channel_weights.squeeze(),        # [C]
        'spatial_weights': spatial_weights.squeeze(),        # [H, W]
        'gate_weights': gate_weights.squeeze(),              # [C]
        'temperature': temperature,
        'ssaf_output': ssaf_output.squeeze()                 # [C, H, W]
    }


def visualize_attention_comprehensive(attention_data, output_path, band_names=None):
    """全面可视化注意力图"""
    
    if band_names is None:
        band_names = ['Red', 'Green', 'Blue', 'NIR']
    
    img = attention_data['image']
    pred = attention_data['prediction']
    spectral_w = attention_data['spectral_weights']
    channel_w = attention_data['channel_weights']
    spatial_w = attention_data['spatial_weights']
    gate_w = attention_data['gate_weights']
    temp = attention_data['temperature']
    ssaf_out = attention_data.get('ssaf_output')
    
    # 归一化空间注意力图用于可视化（解决显示过暗问题）
    spatial_w_min = spatial_w.min()
    spatial_w_max = spatial_w.max()
    spatial_w_norm = (spatial_w - spatial_w_min) / (spatial_w_max - spatial_w_min + 1e-8)
    
    # 准备可视化的RGB图像
    if img.shape[2] >= 3:
        vis_img = img[:, :, :3]
    else:
        vis_img = np.repeat(img[:, :, 0:1], 3, axis=2)
    
    vis_img = vis_img.astype(np.float32)
    if vis_img.max() > 1.0:
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-8)
        
    # 准备 SSAF 输出的可视化 (使用 PCA)
    if ssaf_out is not None:
        # 计算 PCA 特征图
        vis_ssaf = compute_pca_rgb(ssaf_out)
        
        # 计算差值图 (Difference Map)
        # 比较 SSAF 输出和原始输入的差异
        # 注意：需要确保维度一致。如果 ssaf_out 和 img 维度一样
        if ssaf_out.shape == img.transpose(2, 0, 1).shape:
            img_tensor = img.transpose(2, 0, 1)
            # 计算平均绝对误差 (MAE) 作为差异强度
            diff_map = np.mean(np.abs(ssaf_out - img_tensor), axis=0)
            # 归一化差异图
            diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
        else:
            diff_map = np.zeros_like(spatial_w)
    else:
        vis_ssaf = np.zeros_like(vis_img)
        diff_map = np.zeros_like(spatial_w)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, hspace=0.35, wspace=0.35)
    
    # ========== 第一行：输入 -> SSAF输出(PCA) -> 差值图 -> 预测 ==========
    # 1. 原图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(vis_img)
    ax1.set_title('Input Image (RGB)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. SSAF 输出特征图 (PCA)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(vis_ssaf)
    ax2.set_title('SSAF Output (PCA Vis)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. 差值图 (Difference Map)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(diff_map, cmap='magma', vmin=0, vmax=1)
    ax3.set_title('Enhancement Intensity\n|Output - Input|', fontsize=14, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('Diff Intensity', fontsize=10)
    
    # 4. 预测结果
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(pred, cmap='viridis', vmin=0, vmax=1)
    ax4.set_title('Prediction', fontsize=14, fontweight='bold')
    ax4.axis('off')
    cbar4 = plt.colorbar(ax4.images[0], ax=ax4, fraction=0.046)
    cbar4.set_label('Probability', fontsize=10)
    
    # ========== 第二行：光谱与通道注意力 ==========
    # 5. 光谱注意力权重（柱状图）
    ax5 = fig.add_subplot(gs[1, 0])
    colors_spectral = plt.cm.Set3(np.linspace(0, 1, len(band_names)))
    bars5 = ax5.bar(band_names, spectral_w, color=colors_spectral, 
                    edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax5.set_title(f'Spectral Attention\n(Temperature: {temp:.2f})', 
                 fontsize=14, fontweight='bold')
    ax5.set_ylim([0, max(spectral_w) * 1.2])
    ax5.grid(axis='y', alpha=0.3)
    
    # 在柱状图上标注数值
    for i, (bar, val) in enumerate(zip(bars5, spectral_w)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # 6. 通道注意力权重（柱状图）
    ax6 = fig.add_subplot(gs[1, 1])
    colors_channel = plt.cm.Set2(np.linspace(0, 1, len(band_names)))
    bars6 = ax6.bar(band_names, channel_w, color=colors_channel,
                    edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax6.set_title('Channel Attention (SE Block)', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, max(channel_w) * 1.2])
    ax6.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars6, channel_w)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # 7. 门控权重（柱状图）
    ax7 = fig.add_subplot(gs[1, 2])
    colors_gate = plt.cm.Pastel1(np.linspace(0, 1, len(band_names)))
    bars7 = ax7.bar(band_names, gate_w, color=colors_gate,
                    edgecolor='black', linewidth=1.5)
    ax7.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax7.set_title('Dynamic Gating Weights', fontsize=14, fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars7, gate_w)):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # 8. 权重对比（雷达图）
    ax8 = fig.add_subplot(gs[1, 3], projection='polar')
    
    # 归一化权重
    spectral_norm = spectral_w / spectral_w.sum()
    channel_norm = channel_w / channel_w.max()
    gate_norm = gate_w
    
    angles = np.linspace(0, 2 * np.pi, len(band_names), endpoint=False).tolist()
    angles += angles[:1]
    
    spectral_plot = spectral_norm.tolist() + [spectral_norm[0]]
    channel_plot = channel_norm.tolist() + [channel_norm[0]]
    gate_plot = gate_norm.tolist() + [gate_norm[0]]
    
    ax8.plot(angles, spectral_plot, 'o-', linewidth=2, label='Spectral', color='#E74C3C')
    ax8.fill(angles, spectral_plot, alpha=0.15, color='#E74C3C')
    
    ax8.plot(angles, channel_plot, 's-', linewidth=2, label='Channel', color='#3498DB')
    ax8.fill(angles, channel_plot, alpha=0.15, color='#3498DB')
    
    ax8.plot(angles, gate_plot, '^-', linewidth=2, label='Gating', color='#2ECC71')
    ax8.fill(angles, gate_plot, alpha=0.15, color='#2ECC71')
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(band_names, fontsize=11)
    ax8.set_ylim([0, 1])
    ax8.set_title('Attention Weights Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax8.grid(True)
    
    # ========== 第三行：各波段可视化与空间注意力叠加 ==========
    for i in range(4):
        ax = fig.add_subplot(gs[2, i])
        
        if i < img.shape[2]:
            band_img = img[:, :, i]
            
            # 归一化到0-1
            band_img = band_img.astype(np.float32)
            if band_img.max() > 1.0:
                band_img = (band_img - band_img.min()) / (band_img.max() - band_img.min() + 1e-8)
            
            # 叠加空间注意力
            overlay_band = np.stack([band_img] * 3, axis=-1)
            
            # 使用热图显示注意力区域 (使用归一化后的权重)
            attention_colored = plt.cm.hot(spatial_w_norm)[:, :, :3]
            overlay_band = overlay_band * 0.6 + attention_colored * 0.4
            
            ax.imshow(overlay_band)
            ax.set_title(f'{band_names[i]} Band + Spatial Attention\n'
                        f'Spectral W: {spectral_w[i]:.3f}, Channel W: {channel_w[i]:.3f}',
                        fontsize=11, fontweight='bold')
        else:
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 总标题
    plt.suptitle('SSAF Module Attention Visualization', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f'Comprehensive attention visualization saved to {output_path}')


def visualize_attention_simple(attention_data, output_path, band_names=None):
    """简化版可视化（适合论文）"""
    
    if band_names is None:
        band_names = ['Red', 'Green', 'Blue', 'NIR']
    
    img = attention_data['image']
    spatial_w = attention_data['spatial_weights']
    spectral_w = attention_data['spectral_weights']
    ssaf_out = attention_data.get('ssaf_output')
    
    # 归一化空间注意力图
    spatial_w_max = spatial_w.max()
    spatial_w_norm = (spatial_w - spatial_w.min()) / (spatial_w_max - spatial_w.min() + 1e-8)
    
    # 准备RGB图像
    if img.shape[2] >= 3:
        vis_img = img[:, :, :3]
    else:
        vis_img = np.repeat(img[:, :, 0:1], 3, axis=2)
    
    vis_img = vis_img.astype(np.float32)
    if vis_img.max() > 1.0:
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-8)
        
    # 准备 SSAF 输出 (PCA) 和 差值图
    if ssaf_out is not None:
        vis_ssaf = compute_pca_rgb(ssaf_out)
        
        if ssaf_out.shape == img.transpose(2, 0, 1).shape:
            img_tensor = img.transpose(2, 0, 1)
            diff_map = np.mean(np.abs(ssaf_out - img_tensor), axis=0)
            diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
        else:
            diff_map = np.zeros_like(spatial_w)
    else:
        vis_ssaf = np.zeros_like(vis_img)
        diff_map = np.zeros_like(spatial_w)
    
    # 创建2x2布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # (a) SSAF 输出 (PCA)
    axes[0, 0].imshow(vis_ssaf)
    axes[0, 0].set_title('(a) SSAF Output Features (PCA)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # (b) 增强强度 (差值图)
    im_diff = axes[0, 1].imshow(diff_map, cmap='magma')
    axes[0, 1].set_title('(b) Enhancement Intensity', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    cbar_diff = plt.colorbar(im_diff, ax=axes[0, 1], fraction=0.046)
    cbar_diff.set_label('Intensity', fontsize=11)
    
    # (c) 空间注意力图
    im = axes[1, 0].imshow(spatial_w_norm, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'(c) Spatial Attention Map', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    cbar.set_label('Norm. Weight', fontsize=11)
    
    # (d) 预测结果叠加
    # 准备预测叠加图
    pred = attention_data['prediction']
    overlay = vis_img.copy()
    mask = pred > 0.5
    overlay[mask] = overlay[mask] * 0.5 + np.array([1, 0, 0]) * 0.5
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('(d) Prediction Overlay', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f'Simple attention visualization saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize SSAF attention maps from trained MS-HRNet'
    )
    
    parser.add_argument('--model', '-m', default='checkpoints/BEST_ms_hrnet_v2_dat_4bands.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', '-i', default='/home/lucianlu/data/dat_4bands/val/images/000000193.tif',
                       help='Path to input image')
    parser.add_argument('--output', '-o', default='attention_vis',
                       help='Output directory')
    parser.add_argument('--in-ch', type=int, default=4,
                       help='Number of input channels')
    parser.add_argument('--band-names', nargs='+', 
                       default=['Red', 'Green', 'Blue', 'NIR'],
                       help='Names of input bands')
    parser.add_argument('--simple', action='store_true',
                       help='Generate simple visualization for paper')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 准备输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net = load_model(args.model, args.in_ch, device)
    
    # 提取注意力图
    logging.info(f'Processing image: {args.image}')
    attention_data = extract_attention_maps(net, args.image, device)
    
    # 输出注意力权重信息
    print('\n' + '='*60)
    print('SSAF Attention Weights')
    print('='*60)
    print(f'Temperature: {attention_data["temperature"]:.4f}')
    print(f'\nSpectral Attention Weights:')
    for name, weight in zip(args.band_names, attention_data['spectral_weights']):
        print(f'  {name:8s}: {weight:.4f}')
    print(f'\nChannel Attention Weights:')
    for name, weight in zip(args.band_names, attention_data['channel_weights']):
        print(f'  {name:8s}: {weight:.4f}')
    print(f'\nGating Weights:')
    for name, weight in zip(args.band_names, attention_data['gate_weights']):
        print(f'  {name:8s}: {weight:.4f}')
    print('='*60 + '\n')
    
    # 可视化
    img_name = Path(args.image).stem
    
    if args.simple:
        # 简化版（适合论文）
        output_path = output_dir / f'{img_name}_attention_simple.png'
        visualize_attention_simple(attention_data, output_path, args.band_names)
    else:
        # 完整版
        output_path = output_dir / f'{img_name}_attention_full.png'
        visualize_attention_comprehensive(attention_data, output_path, args.band_names)
    
    logging.info('Visualization complete!')


if __name__ == '__main__':
    main()