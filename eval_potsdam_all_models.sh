#!/usr/bin/env bash
# =============================================================================
# eval_potsdam_all_models.sh
# 在Potsdam测试集上评估所有模型，生成对比表格
#
# 使用方法:
#   chmod +x eval_potsdam_all_models.sh
#   ./eval_potsdam_all_models.sh
# =============================================================================

set -euo pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

TEST_IMG="/home/lucianlu/data/data_potsdam/val/images/"
TEST_MASK="/home/lucianlu/data/data_potsdam/val/labels/"
OUTPUT_DIR="results_potsdam_val"
IN_CH=4


echo "======================================"
echo " Evaluating all models on Potsdam"
echo "======================================"

# 生成eval_potsdam.json配置文件
cat > eval_potsdam.json << 'EOF'
{
  "models": [
    {
      "name": "UNet",
      "type": "unet",
      "path": "checkpoints_potsdam/BEST_unet_combined_potsdam.pth"
    },
    {
      "name": "UNet++",
      "type": "unet_plusplus",
      "path": "checkpoints_potsdam/BEST_unet_plusplus_combined_potsdam.pth"
    },
    {
      "name": "PSPNet",
      "type": "pspnet",
      "path": "checkpoints_potsdam/BEST_pspnet_combined_potsdam.pth"
    },
    {
      "name": "DeepLabV3+",
      "type": "deeplabv3_plus",
      "path": "checkpoints_potsdam/BEST_deeplabv3_plus_combined_potsdam.pth"
    },
    {
      "name": "HRNet",
      "type": "hrnet",
      "path": "checkpoints_potsdam/BEST_hrnet_combined_potsdam.pth"
    },
    {
      "name": "UNetFormer",
      "type": "unetformer",
      "path": "checkpoints_potsdam/BEST_unetformer_combined_potsdam_4.pth"
    },
    {
      "name": "MS-HRNet (Ours)",
      "type": "ms_hrnet",
      "path": "checkpoints_potsdam/BEST_ms_hrnet_combined_potsdam_8.pth"
    }
  ]
}
EOF

echo "Generated eval_potsdam.json"

python batch_evaluation.py \
    --config      eval_potsdam.json  \
    --test-img    "$TEST_IMG"         \
    --test-mask   "$TEST_MASK"        \
    --in-ch       "$IN_CH"            \
    --output-dir  "$OUTPUT_DIR"       \
    --visualize

echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo "  - comparison_table.csv"
echo "  - comparison_table.tex"
echo "  - comparison_report.md"