#!/usr/bin/env bash
set -euo pipefail

# GPU 选择（逗号分隔多卡，如 0,1）
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# 数据集路径（可通过环境变量覆盖）
DATASET_NAME="${DATASET_NAME:-4bands}"
TRAIN_IMG="${TRAIN_IMG:-/mnt/U/Dat_Seg/dat_4bands/train/images/}"
TRAIN_MASK="${TRAIN_MASK:-/mnt/U/Dat_Seg/dat_4bands/train/labels/}"
VAL_IMG="${VAL_IMG:-/mnt/U/Dat_Seg/dat_4bands/val/images/}"
VAL_MASK="${VAL_MASK:-/mnt/U/Dat_Seg/dat_4bands/val/labels/}"

# 训练超参（可通过环境变量覆盖）
IN_CH="${IN_CH:-4}"
EPOCHS="${EPOCHS:-200}"
BS_DEFAULT="${BS:-4}"
LR="${LR:-1e-3}"
SCALE="${SCALE:-1.0}"
WARMUP="${WARMUP:-5}"
DEEP_SUP="${DEEP_SUP:-1}"   # 1 启用深度监督，0 关闭

# 可按模型覆写 batch size（可选）
BS_UNET="${BS_UNET:-}"          # 例如 2
BS_UNETPP="${BS_UNETPP:-}"      # 例如 2
BS_HRNET="${BS_HRNET:-}"        # 例如 8
BS_DEEPLAB="${BS_DEEPLAB:-}"    # 例如 4
BS_PSPNET="${BS_PSPNET:-}"      # 例如 4

# 需要跑的模型列表（空格分隔，可通过环境变量 MODELS 覆盖）
MODELS_STR="${MODELS:-unet unet_plusplus pspnet deeplabv3_plus hrnet_ocr_w48 ms_hrnet_w48}"
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
  # 模型特定 batch size
  CUR_BS="$BS_DEFAULT"
  case "$MODEL" in
    unet)
      CUR_BS="${BS_UNET:-$BS_DEFAULT}"
      ;;
    unet_plusplus)
      CUR_BS="${BS_UNETPP:-$BS_DEFAULT}"
      ;;
    hrnet_ocr_w48|ms_hrnet_w48)
      CUR_BS="${BS_HRNET:-$BS_DEFAULT}"
      ;;
    deeplabv3_plus)
      CUR_BS="${BS_DEEPLAB:-$BS_DEFAULT}"
      ;;
    pspnet)
      CUR_BS="${BS_PSPNET:-$BS_DEFAULT}"
      ;;
  esac

  LOG_FILE="${LOG_DIR}/${MODEL}.log"
  echo "\n==== Running: ${MODEL} (BS=${CUR_BS}) ====" | tee -a "$LOG_FILE"

  CMD=(python -u train.py \
    --train-img "$TRAIN_IMG" \
    --train-mask "$TRAIN_MASK" \
    --val-img "$VAL_IMG" \
    --val-mask "$VAL_MASK" \
    -e "$EPOCHS" \
    -b "$CUR_BS" \
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
