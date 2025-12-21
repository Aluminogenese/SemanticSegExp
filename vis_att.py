import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import MSHRNet
from predict import read_image_any, normalize_image

# ==========================================
# 论文绘图风格设置
# ==========================================
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
except:
    pass
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

def load_model(model_path, in_channels=4, device='cuda'):
    """加载模型"""
    net = MSHRNet(in_channels=in_channels, num_classes=1, base_channels=48)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device=device)
    net.eval()
    return net

def process_single_image(net, img_path, device):
    """处理单张图片，返回可视化用的原图、差值热力图和光谱权重"""
    # 1. 读取和预处理
    img_np = read_image_any(img_path)
    img_normalized = normalize_image(img_np)
    
    # 2. 转换为 Tensor
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    
    # 3. 推理
    with torch.no_grad():
        _, attention_maps = net(img_tensor)
    
    # 提取光谱权重
    spectral_weights = attention_maps['spectral_weights'].cpu().numpy().squeeze()
    
    # 4. 获取 SSAF 输出
    if 'ssaf_output' in attention_maps:
        ssaf_out = attention_maps['ssaf_output'].cpu().numpy().squeeze() # [C, H, W]
    else:
        logging.warning("Model does not return 'ssaf_output'. Using zero map.")
        ssaf_out = np.zeros_like(img_normalized.transpose(2, 0, 1))

    # 5. 准备 RGB 原图用于显示
    if img_np.shape[2] >= 3:
        vis_img = img_np[:, :, :3]
    else:
        vis_img = np.repeat(img_np[:, :, 0:1], 3, axis=2)
    
    # 简单的 Min-Max 归一化用于显示
    vis_img = vis_img.astype(np.float32)
    if vis_img.max() > 255:
        vis_img = vis_img / vis_img.max()
    elif vis_img.max() > 1.0:
        vis_img = vis_img / 255.0
    
    # 6. 计算增强差值热力图 (Enhancement Difference Map)
    # Formula: Mean(|SSAF_Output - Input|)
    # 确保维度一致
    input_tensor_np = img_normalized.transpose(2, 0, 1)
    
    if ssaf_out.shape == input_tensor_np.shape:
        # 计算绝对差值
        diff = np.abs(ssaf_out - input_tensor_np)
        # 在通道维度求平均，得到 [H, W] 的强度图
        diff_map = np.mean(diff, axis=0)
        
        # 归一化到 0-1 以便可视化
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    else:
        diff_map = np.zeros((img_np.shape[0], img_np.shape[1]))

    return vis_img, diff_map, spectral_weights

def create_paper_figure(image_paths, net, device, output_file, band_names=None):
    """生成论文风格的对比图"""
    if band_names is None:
        band_names = ['Red', 'Green', 'Blue', 'NIR']

    num_images = len(image_paths)
    
    # 创建画布：每行一张图，三列（原图，热力图，光谱权重）
    # 调整 figsize 以保证图片比例合适
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 3.5 * num_images), constrained_layout=True)
    
    # 确保 axes 是二维数组
    if num_images == 1:
        axes = np.array([axes])
    
    logging.info(f"Generating visualization for {num_images} images...")
    
    for i, img_path in enumerate(image_paths):
        logging.info(f"Processing: {img_path}")
        vis_img, diff_map, spectral_w = process_single_image(net, img_path, device)
        
        # --- 左侧：原图 ---
        ax_img = axes[i, 0]
        ax_img.imshow(vis_img)
        ax_img.axis('off')
        
        # 仅在第一行显示标题
        if i == 0:
            ax_img.set_title('(a) Input Image', fontweight='bold', pad=10)
            
        # --- 中间：光谱权重 ---
        ax_spec = axes[i, 1]
        
        # 颜色设置
        colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12'] # R, G, B, NIR
        if len(spectral_w) > 4:
             colors = plt.cm.Set3(np.linspace(0, 1, len(spectral_w)))
        
        # 绘制柱状图
        bars = ax_spec.bar(band_names[:len(spectral_w)], spectral_w, color=colors[:len(spectral_w)], 
                          edgecolor='black', linewidth=1.5, width=0.6)
        
        # 设置样式
        ax_spec.set_ylim(0, max(spectral_w) * 1.3) # 留出顶部空间写数字
        ax_spec.grid(axis='y', linestyle='--', alpha=0.3)
        ax_spec.spines['top'].set_visible(False)
        ax_spec.spines['right'].set_visible(False)
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax_spec.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        if i == 0:
            ax_spec.set_title('(b) Spectral Attention Weights', fontweight='bold', pad=10)

        # --- 右侧：热力图 ---
        ax_diff = axes[i, 2]
        # 使用 'magma' 或 'inferno' 色图，这些在论文中很常用且黑白打印可读性好
        im = ax_diff.imshow(diff_map, cmap='magma', vmin=0, vmax=1)
        ax_diff.axis('off')
        
        if i == 0:
            ax_diff.set_title('(c) SSAF Difference HeatMap', fontweight='bold', pad=10)
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        # cbar.set_label('Intensity', fontsize=8)

    # 保存
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logging.info(f"Saved result to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate paper-ready SSAF visualization')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--images', '-i', nargs='+', required=True, help='List of input image paths')
    parser.add_argument('--output', '-o', default='paper_vis_heatmap.png', help='Output filename')
    parser.add_argument('--in-ch', type=int, default=4, help='Number of input channels')
    parser.add_argument('--band-names', nargs='+', default=['Red', 'Green', 'Blue', 'NIR'], help='Names of input bands')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    net = load_model(args.model, args.in_ch, device)
    
    # 生成图表
    create_paper_figure(args.images, net, device, args.output, args.band_names)

if __name__ == '__main__':
    main()