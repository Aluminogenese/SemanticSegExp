"""
简化版模型复杂度分析 - 仅需PyTorch
计算参数量、理论FLOPs(手动计算)、推理速度

使用方法:
python simple_model_analysis.py

输出:
1. 参数量对比表格
2. 推理速度对比
3. LaTeX表格
"""

import torch
import time
import numpy as np
from pathlib import Path

from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNet, MSHRNet


def count_parameters(model):
    """计算参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'total_m': total / 1e6
    }


def estimate_flops_conv2d(in_channels, out_channels, kernel_size, 
                          input_h, input_w, stride=1, padding=0, groups=1):
    """估算2D卷积的FLOPs"""
    output_h = (input_h + 2 * padding - kernel_size) // stride + 1
    output_w = (input_w + 2 * padding - kernel_size) // stride + 1
    
    kernel_flops = kernel_size * kernel_size * (in_channels // groups)
    output_size = output_h * output_w * out_channels
    
    flops = kernel_flops * output_size
    return flops, (output_h, output_w)


def measure_inference_time(model, input_size, device, iterations=100, warmup=10):
    """测量推理时间"""
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # 同步
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    times = np.array(times) * 1000  # 转换为ms
    
    return {
        'mean_ms': times.mean(),
        'std_ms': times.std(),
        'min_ms': times.min(),
        'max_ms': times.max(),
        'fps': 1000 / times.mean()
    }


def analyze_model(name, model_class, in_channels, device, input_size=512):
    """分析单个模型"""
    print(f'\nAnalyzing {name}...')
    
    # 创建模型
    model = model_class(in_channels=in_channels, num_classes=1)
    
    # 参数量
    params = count_parameters(model)
    
    # 推理速度
    speed = measure_inference_time(
        model, (in_channels, input_size, input_size), device
    )
    
    # 显存
    memory_mb = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        dummy_input = torch.randn(1, in_channels, input_size, input_size).to(device)
        with torch.no_grad():
            _ = model.to(device)(dummy_input)
        
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return {
        'name': name,
        'params_m': params['total_m'],
        'params_total': params['total'],
        'time_ms': speed['mean_ms'],
        'time_std_ms': speed['std_ms'],
        'fps': speed['fps'],
        'memory_mb': memory_mb
    }


def generate_comparison_table(results):
    """生成对比表格"""
    
    # 排序
    results_sorted = sorted(results, key=lambda x: x['params_m'])
    
    print('\n' + '='*100)
    print('MODEL COMPLEXITY COMPARISON')
    print('='*100)
    print(f"{'Model':<20} {'Params (M)':<15} {'Time (ms)':<15} {'FPS':<10} {'Memory (MB)':<15}")
    print('-'*100)
    
    for r in results_sorted:
        mem_str = f"{r['memory_mb']:.1f}" if r['memory_mb'] else 'N/A'
        print(f"{r['name']:<20} {r['params_m']:<15.2f} "
              f"{r['time_ms']:<15.2f} {r['fps']:<10.2f} {mem_str:<15}")
    
    print('='*100)
    
    # LaTeX表格
    latex = generate_latex_table(results_sorted)
    
    # 保存
    output_dir = Path('complexity_analysis')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'model_complexity.tex', 'w') as f:
        f.write(latex)
    
    print(f'\nLaTeX table saved to {output_dir}/model_complexity.tex')
    
    # CSV
    with open(output_dir / 'model_complexity.csv', 'w') as f:
        f.write('Model,Params(M),Time(ms),FPS,Memory(MB)\n')
        for r in results_sorted:
            mem_str = f"{r['memory_mb']:.1f}" if r['memory_mb'] else 'N/A'
            f.write(f"{r['name']},{r['params_m']:.2f},{r['time_ms']:.2f},"
                   f"{r['fps']:.2f},{mem_str}\n")
    
    print(f'CSV table saved to {output_dir}/model_complexity.csv')


def generate_latex_table(results):
    """生成LaTeX表格"""
    
    # 找最优值
    min_params_idx = min(range(len(results)), key=lambda i: results[i]['params_m'])
    min_time_idx = min(range(len(results)), key=lambda i: results[i]['time_ms'])
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Model Complexity and Efficiency Comparison}
\label{tab:model_complexity}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Params (M)} & \textbf{Time (ms)} & \textbf{FPS} & \textbf{Memory (MB)} \\
\midrule
"""
    
    for i, r in enumerate(results):
        name = r['name']
        
        # 参数量
        params_str = f"{r['params_m']:.2f}"
        if i == min_params_idx:
            params_str = f"\\textbf{{{params_str}}}"
        
        # 时间
        time_str = f"{r['time_ms']:.2f}"
        if i == min_time_idx:
            time_str = f"\\textbf{{{time_str}}}"
        
        fps_str = f"{r['fps']:.2f}"
        
        mem_str = f"{r['memory_mb']:.1f}" if r['memory_mb'] else 'N/A'
        
        latex += f"{name} & {params_str} & {time_str} & {fps_str} & {mem_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def main():
    print('Model Complexity Analysis')
    print('='*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    in_channels = 4
    input_size = 512
    
    print(f'Input: {in_channels} channels, {input_size}×{input_size}')
    
    # 定义要分析的模型
    models = [
        ('UNet', UNet),
        ('UNet++', UNetPlusPlus),
        ('PSPNet', PSPNet),
        ('DeepLabV3+', DeepLabV3Plus),
        ('HRNet', HRNet),
        ('MS-HRNet', MSHRNet),
    ]
    
    # 分析所有模型
    results = []
    for name, model_class in models:
        try:
            result = analyze_model(name, model_class, in_channels, device, input_size)
            results.append(result)
            
            # 清理
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f'Error analyzing {name}: {e}')
            continue
    
    # 生成对比表格
    if results:
        generate_comparison_table(results)
    
    print('\nAnalysis complete!')


if __name__ == '__main__':
    main()