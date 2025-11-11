"""
批量评估多个模型并生成对比表格

使用示例:
python batch_evaluation.py --config eval_config.json \
                           --test-img /path/to/test/images \
                           --test-mask /path/to/test/labels \
                           --output-dir comparison_results
"""

import argparse
import logging
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# 导入综合评估器
import sys
sys.path.insert(0, str(Path(__file__).parent))
from comprehensive_evaluation import ComprehensiveEvaluator

import torch
from models import UNet, UNetPlusPlus, PSPNet, DeepLabV3Plus, HRNet, HRNetOCR, MSHRNetOCR, MSHRNetAblation


def load_model(model_type, model_path, in_channels, device):
    """加载模型"""
    if model_type == 'unet':
        net = UNet(in_channels=in_channels, num_classes=1)
    elif model_type == 'unet_plusplus':
        net = UNetPlusPlus(in_channels=in_channels, num_classes=1)
    elif model_type == 'pspnet':
        net = PSPNet(in_channels=in_channels, num_classes=1)
    elif model_type == 'deeplabv3_plus':
        net = DeepLabV3Plus(in_channels=in_channels, num_classes=1)
    elif model_type == 'hrnet':
        net = HRNet(in_channels=in_channels, num_classes=1, base_channels=48)
    elif model_type == 'hrnet_ocr':
        net = HRNetOCR(in_channels=in_channels, num_classes=1, base_channels=48)
    elif model_type == 'ms_hrnet':
        net = MSHRNetOCR(in_channels=in_channels, num_classes=1, base_channels=48)
    elif model_type == 'ms_hrnet_no_ssaf':
        net = MSHRNetAblation(in_channels=in_channels, num_classes=1, base_channels=48,
                             use_ssaf=False, use_msbr=True)
    elif model_type == 'ms_hrnet_no_msbr':
        net = MSHRNetAblation(in_channels=in_channels, num_classes=1, base_channels=48,
                             use_ssaf=True, use_msbr=False)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device=device)
    net.eval()
    
    return net


def statistical_test(results1, results2, metric='dice'):
    """统计显著性检验（配对t检验）"""
    values1 = [r[metric] for r in results1]
    values2 = [r[metric] for r in results2]
    
    # 配对t检验
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    # Wilcoxon符号秩检验（非参数）
    w_stat, w_p_value = stats.wilcoxon(values1, values2)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'wilcoxon_statistic': float(w_stat),
        'wilcoxon_p_value': float(w_p_value),
        'significant': bool(p_value < 0.05)  # 显式转换为Python bool
    }


def generate_comparison_table(all_summaries, model_names):
    """生成对比表格"""
    
    # 创建DataFrame
    data = []
    for name in model_names:
        summary = all_summaries[name]
        data.append({
            'Model': name,
            'Dice': f"{summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}",
            'IoU': f"{summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}",
            'Precision': f"{summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}",
            'Recall': f"{summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}",
            'F1': f"{summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}",
            'B-IoU': f"{summary['mean_boundary_iou']:.4f} ± {summary['std_boundary_iou']:.4f}",
            'Dice_mean': summary['mean_dice'],  # 用于排序
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Dice_mean', ascending=False)
    df = df.drop('Dice_mean', axis=1)
    
    return df


def generate_latex_comparison_table(all_summaries, model_names):
    """生成LaTeX对比表格"""
    
    # 按Dice排序
    sorted_names = sorted(model_names, 
                         key=lambda x: all_summaries[x]['mean_dice'], 
                         reverse=True)
    
    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Different Models on Test Dataset}
\\label{tab:model_comparison}
\\begin{tabular}{lccccccc}
\\toprule
\\textbf{Model} & \\textbf{Dice} & \\textbf{IoU} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Accuracy} & \\textbf{B-IoU} \\\\
\\midrule
"""
    
    for name in sorted_names:
        s = all_summaries[name]
        
        # 找出最佳值（用于加粗）
        is_best_dice = s['mean_dice'] == max(all_summaries[n]['mean_dice'] for n in model_names)
        is_best_iou = s['mean_iou'] == max(all_summaries[n]['mean_iou'] for n in model_names)
        
        dice_str = f"\\textbf{{{s['mean_dice']:.4f}}}" if is_best_dice else f"{s['mean_dice']:.4f}"
        iou_str = f"\\textbf{{{s['mean_iou']:.4f}}}" if is_best_iou else f"{s['mean_iou']:.4f}"
        
        latex += f"{name} & {dice_str} & {iou_str} & "
        latex += f"{s['mean_precision']:.4f} & {s['mean_recall']:.4f} & "
        latex += f"{s['mean_f1']:.4f} & {s['mean_accuracy']:.4f} & "
        latex += f"{s['mean_boundary_iou']:.4f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex


def plot_model_comparison(all_summaries, model_names, output_dir):
    """绘制模型对比图"""
    
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'boundary_iou']
    metric_labels = ['Dice', 'IoU', 'Precision', 'Recall', 'F1', 'Boundary IoU']
    
    # 按Dice排序模型
    sorted_names = sorted(model_names, 
                         key=lambda x: all_summaries[x]['mean_dice'], 
                         reverse=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = [all_summaries[name][f'mean_{metric}'] for name in sorted_names]
        stds = [all_summaries[name][f'std_{metric}'] for name in sorted_names]
        
        x = np.arange(len(sorted_names))
        bars = axes[i].bar(x, means, yerr=stds, alpha=0.7, capsize=5, 
                          color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names))))
        
        axes[i].set_xlabel('Model', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(label, fontsize=12, fontweight='bold')
        axes[i].set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(sorted_names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # 标注最佳值
        best_idx = np.argmax(means)
        axes[i].text(best_idx, means[best_idx] + stds[best_idx], 
                    f'{means[best_idx]:.4f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_radar_chart(all_summaries, model_names, output_dir):
    """绘制雷达图对比"""
    
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'boundary_iou']
    metric_labels = ['Dice', 'IoU', 'Precision', 'Recall', 'F1', 'B-IoU']
    
    # 归一化到0-1（使用所有模型的最大值）
    max_values = {m: max(all_summaries[n][f'mean_{m}'] for n in model_names) 
                  for m in metrics}
    
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for idx, name in enumerate(model_names):
        values = [all_summaries[name][f'mean_{m}'] / max_values[m] for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.title('Model Performance Radar Chart\n(Normalized)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'radar_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Batch Evaluation and Comparison')
    
    # 配置文件
    parser.add_argument('--config', required=True, help='配置JSON文件路径')
    
    # 数据参数
    parser.add_argument('--test-img', required=True, help='测试图像目录')
    parser.add_argument('--test-mask', required=True, help='测试mask目录')
    parser.add_argument('--in-ch', type=int, default=4, help='输入通道数')
    
    # 评估参数
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--output-dir', default='comparison_results', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化对比图')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    models_config = config['models']
    
    # 加载设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # 评估所有模型
    all_summaries = {}
    all_results = {}
    model_names = []
    
    for model_cfg in models_config:
        name = model_cfg['name']
        model_type = model_cfg['type']
        model_path = model_cfg['path']
        
        logging.info(f'\n{"="*80}')
        logging.info(f'Evaluating: {name}')
        logging.info(f'{"="*80}')
        
        try:
            # 加载模型
            net = load_model(model_type, model_path, args.in_ch, device)
            
            # 创建评估器
            evaluator = ComprehensiveEvaluator(net, device, threshold=args.threshold)
            
            # 评估
            results = evaluator.evaluate_dataset(
                args.test_img,
                args.test_mask,
                visualize=args.visualize,
                output_dir=output_dir / name
            )
            
            if results:
                summary = evaluator.generate_summary()
                all_summaries[name] = summary
                all_results[name] = results
                model_names.append(name)
                
                logging.info(f'{name} - Dice: {summary["mean_dice"]:.4f}, IoU: {summary["mean_iou"]:.4f}')
            
        except Exception as e:
            logging.error(f'Error evaluating {name}: {e}')
            continue
    
    if not all_summaries:
        logging.error('No models evaluated successfully!')
        return
    
    # 生成对比表格
    logging.info('\n' + '='*80)
    logging.info('Generating comparison tables...')
    logging.info('='*80)
    
    # 1. CSV表格
    df_comparison = generate_comparison_table(all_summaries, model_names)
    csv_path = output_dir / 'comparison_table.csv'
    df_comparison.to_csv(csv_path, index=False)
    logging.info(f'CSV table saved to {csv_path}')
    
    # 2. LaTeX表格
    latex_table = generate_latex_comparison_table(all_summaries, model_names)
    latex_path = output_dir / 'comparison_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    logging.info(f'LaTeX table saved to {latex_path}')
    
    # 3. 可视化对比
    plot_model_comparison(all_summaries, model_names, output_dir)
    logging.info(f'Comparison plot saved to {output_dir}/model_comparison.png')
    
    plot_radar_chart(all_summaries, model_names, output_dir)
    logging.info(f'Radar chart saved to {output_dir}/radar_comparison.png')
    
    # 4. 统计显著性检验（如果有多个模型）
    if len(model_names) >= 2:
        logging.info('\nPerforming statistical significance tests...')
        
        # 找出最佳模型
        best_model = max(model_names, key=lambda x: all_summaries[x]['mean_dice'])
        
        sig_tests = {}
        for name in model_names:
            if name != best_model:
                test_result = statistical_test(
                    all_results[best_model], 
                    all_results[name], 
                    metric='dice'
                )
                sig_tests[f'{best_model}_vs_{name}'] = test_result
                
                sig_symbol = '***' if test_result['p_value'] < 0.001 else \
                            '**' if test_result['p_value'] < 0.01 else \
                            '*' if test_result['p_value'] < 0.05 else 'ns'
                
                logging.info(f'  {best_model} vs {name}: p={test_result["p_value"]:.4f} {sig_symbol}')
        
        # 保存统计检验结果
        sig_path = output_dir / 'statistical_tests.json'
        with open(sig_path, 'w') as f:
            json.dump(sig_tests, f, indent=2)
    
    # 5. 生成Markdown对比报告
    markdown_report = f"""
# Model Comparison Report

## Dataset Information
- **Test Images**: {args.test_img}
- **Test Masks**: {args.test_mask}
- **Number of Models**: {len(model_names)}

## Performance Ranking (by Dice Coefficient)

{df_comparison.to_markdown(index=False)}

## Best Performing Model

**{max(model_names, key=lambda x: all_summaries[x]['mean_dice'])}** achieves the highest Dice coefficient of **{max(all_summaries[x]['mean_dice'] for x in model_names):.4f}**.

## Key Observations

"""
    
    # 添加关键观察
    best_model = max(model_names, key=lambda x: all_summaries[x]['mean_dice'])
    best_dice = all_summaries[best_model]['mean_dice']
    
    for name in model_names:
        if name != best_model:
            diff = best_dice - all_summaries[name]['mean_dice']
            markdown_report += f"- **{name}**: Dice is {diff:.4f} lower than {best_model}\n"
    
    markdown_report += f"""

## Visualizations

- Bar chart comparison: `model_comparison.png`
- Radar chart: `radar_comparison.png`
- Individual model results: `<model_name>/` directories

## Files Generated

- Comparison table (CSV): `comparison_table.csv`
- Comparison table (LaTeX): `comparison_table.tex`
- Statistical tests: `statistical_tests.json`
- This report: `comparison_report.md`
"""
    
    report_path = output_dir / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.write(markdown_report)
    logging.info(f'Comparison report saved to {report_path}')
    
    # 打印最终结果
    print('\n' + '='*80)
    print('COMPARISON RESULTS')
    print('='*80)
    print(df_comparison.to_string(index=False))
    print('='*80)
    
    logging.info('\nBatch evaluation complete!')


if __name__ == '__main__':
    main()