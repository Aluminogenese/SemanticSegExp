#!/usr/bin/env bash
set -euo pipefail

# GPU 选择（逗号分隔多卡，如 0,1）
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# 数据集路径（可通过环境变量覆盖）
DATASET_NAME="${DATASET_NAME:-dat_3bands}"
TRAIN_IMG="${TRAIN_IMG:-/mnt/U/Dat_Seg/dat_3bands/train/images/}"
TRAIN_MASK="${TRAIN_MASK:-/mnt/U/Dat_Seg/dat_3bands/train/labels/}"
VAL_IMG="${VAL_IMG:-/mnt/U/Dat_Seg/dat_3bands/val/images/}"
VAL_MASK="${VAL_MASK:-/mnt/U/Dat_Seg/dat_3bands/val/labels/}"

# 训练超参（可通过环境变量覆盖）
IN_CH="${IN_CH:-3}"
EPOCHS="${EPOCHS:-400}"
BS="${BS:-4}"
LR="${LR:-1e-3}"
SCALE="${SCALE:-1.0}"
WARMUP="${WARMUP:-5}"
DEEP_SUP="${DEEP_SUP:-1}"   # 1 启用深度监督，0 关闭

MODELS_STR="${MODELS:-unet unet_plusplus pspnet deeplabv3_plus hrnet_ocr ms_hrnet}"
IFS=' ' read -r -a MODELS <<< "$MODELS_STR"

# 日志目录
STAMP=$(date +%b%d_%H-%M-%S)
LOG_DIR="logs/${STAMP}_${DATASET_NAME}"
mkdir -p "$LOG_DIR"

# 打印配置
echo "GPU: $GPU"
echo "DATASET: $DATASET_NAME"
echo "TRAIN_IMG: $TRAIN_IMG"
echo "TRAIN_MASK: $TRAIN_MASK"
echo "VAL_IMG: $VAL_IMG"
echo "VAL_MASK: $VAL_MASK"
echo "MODELS: ${MODELS[*]}"

echo "Starting experiments..."

for MODEL in "${MODELS[@]}"; do
  LOG_FILE="${LOG_DIR}/${MODEL}.log"
  echo "\n==== Running: ${MODEL} (BS=${BS}) ====" | tee -a "$LOG_FILE"

  CMD=(python -u train.py \
    --train-img "$TRAIN_IMG" \
    --train-mask "$TRAIN_MASK" \
    --val-img "$VAL_IMG" \
    --val-mask "$VAL_MASK" \
    -e "$EPOCHS" \
    -b "$BS" \
    -l "$LR" \
    -s "$SCALE" \
    --model "$MODEL" \
    --in-ch "$IN_CH" \
    --dataset "$DATASET_NAME" \
    --warmup-epochs "$WARMUP"
  )

  if [[ "$DEEP_SUP" == "0" ]]; then
    CMD+=(--no-deep-supervision)
  fi

  echo "Command: ${CMD[*]}" | tee -a "$LOG_FILE"
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

done

echo "All experiments finished. Logs saved to $LOG_DIR"
