#!/usr/bin/env bash
################################################################################
# 消融实验训练脚本
# 
# 实验一: SSAF模块有效性验证
# 实验二: 组合损失有效性验证
#
# 使用方法:
#   1. 前台运行所有实验: ./run_ablation_experiments.sh
#   2. 后台运行所有实验: ./run_ablation_experiments.sh --nohup
#   3. 运行特定实验: ./run_ablation_experiments.sh --exp exp1
#   4. 仅显示命令不执行: ./run_ablation_experiments.sh --dry-run
#
# 后台运行时的日志位置:
#   - 主日志: logs/ablation_<timestamp>.log
#   - 各模型日志: logs/ablation/<model_name>.log
################################################################################

set -euo pipefail

# ==================== 配置区 ====================

# GPU设置
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# 数据集路径
DATASET_NAME="dat_4bands"
TRAIN_IMG="/home/lucianlu/data/dat_4bands/train1/images/"
TRAIN_MASK="/home/lucianlu/data/dat_4bands/train1/labels/"
VAL_IMG="/home/lucianlu/data/dat_4bands/val/images/"
VAL_MASK="/home/lucianlu/data/dat_4bands/val/labels/"

# 训练超参数
IN_CH=4
EPOCHS=400
BATCH_SIZE=8
LR=1e-3
SCALE=1.0
WARMUP=5

# 日志目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/ablation_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# 检查点目录
CHECKPOINT_DIR="checkpoints_ablation"
mkdir -p "$CHECKPOINT_DIR"

# ==================== 实验定义 ====================

# 实验一: SSAF模块有效性验证
declare -A EXP1_MODELS=(
    ["HRNet_BCE"]="hrnet"
    ["MS-HRNet_BCE"]="ms_hrnet"
)

declare -A EXP1_LOSS=(
    ["HRNet_BCE"]="bce"
    ["MS-HRNet_BCE"]="bce"
)

# 实验二: 组合损失有效性验证
declare -A EXP2_MODELS=(
    ["HRNet_BCE"]="hrnet"
    ["HRNet_Dice"]="hrnet"
    ["HRNet_Focal"]="hrnet"
    ["HRNet_BCE+Dice"]="hrnet"
    ["HRNet_Dice+Focal"]="hrnet"
    ["HRNet_Combined"]="hrnet"
)

declare -A EXP2_LOSS=(
    ["HRNet_BCE"]="bce"
    ["HRNet_Dice"]="dice"
    ["HRNet_Focal"]="focal"
    ["HRNet_BCE+Dice"]="bce+dice"
    ["HRNet_Dice+Focal"]="dice+focal"
    ["HRNet_Combined"]="combined"
)

# ==================== 辅助函数 ====================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# 检查checkpoint是否存在
check_checkpoint() {
    local model_name="$1"
    local loss_type="$2"
    
    # 将损失类型转换为标识符
    local loss_id
    case "$loss_type" in
        "bce")
            loss_id="bce"
            ;;
        "dice")
            loss_id="dice"
            ;;
        "focal")
            loss_id="focal"
            ;;
        "bce+dice")
            loss_id="bce+dice"
            ;;
        "dice+focal")
            loss_id="dice+focal"
            ;;
        "combined")
            loss_id="combined"
            ;;
        *)
            # 默认使用原始类型字符串
            loss_id="$loss_type"
            ;;
    esac
    
    local checkpoint="${CHECKPOINT_DIR}/BEST_${model_name}_${loss_id}_${DATASET_NAME}.pth"
    
    if [[ -f "$checkpoint" ]]; then
        return 0  # 存在
    else
        return 1  # 不存在
    fi
}

# 损失配置转换为Python参数
loss_to_args() {
    local loss_type="$1"
    
    case "$loss_type" in
        "bce")
            echo "--loss-weights bce=1.0"
            ;;
        "dice")
            echo "--loss-weights dice=1.0"
            ;;
        "focal")
            echo "--loss-weights focal=1.0"
            ;;
        "bce+dice")
            echo "--loss-weights bce=1.0 dice=1.0"
            ;;
        "dice+focal")
            echo "--loss-weights dice=1.0 focal=1.0"
            ;;
        "combined")
            echo "--loss-weights bce=1.0 dice=1.0 focal=0.5 boundary=0.3"
            ;;
        *)
            log_error "Unknown loss type: $loss_type"
            exit 1
            ;;
    esac
}

# 训练单个模型
train_model() {
    local exp_name="$1"
    local run_name="$2"
    local model_type="$3"
    local loss_type="$4"
    local dry_run="${5:-false}"
    local skip_existing="${6:-true}"
    
    log_section "Training: $run_name"
    
    # 检查是否已存在
    if [[ "$skip_existing" == "true" ]] && check_checkpoint "$model_type" "$loss_type"; then
        log_warn "Checkpoint exists for $model_type with loss $loss_type, skipping..."
        echo "$run_name: SKIPPED" >> "$LOG_DIR/summary.txt"
        return 0
    fi
    
    # 获取损失函数参数
    local loss_args
    loss_args=$(loss_to_args "$loss_type")
    
    # 构建训练命令
    local cmd=(
        python -u train.py
        --train-img "$TRAIN_IMG"
        --train-mask "$TRAIN_MASK"
        --val-img "$VAL_IMG"
        --val-mask "$VAL_MASK"
        -e "$EPOCHS"
        -b "$BATCH_SIZE"
        -l "$LR"
        -s "$SCALE"
        --model "$model_type"
        --in-ch "$IN_CH"
        --dataset "$DATASET_NAME"
        --warmup-epochs "$WARMUP"
        $loss_args
    )
    
    # 日志文件
    local log_file="${LOG_DIR}/${run_name}.log"
    
    # 显示信息
    log_info "Experiment: $exp_name"
    log_info "Run Name: $run_name"
    log_info "Model: $model_type"
    log_info "Loss: $loss_type"
    log_info "Log: $log_file"
    echo ""
    log_info "Command: ${cmd[*]}"
    echo ""
    
    # Dry run模式
    if [[ "$dry_run" == "true" ]]; then
        log_warn "DRY RUN - Not executing"
        echo "$run_name: DRY_RUN" >> "$LOG_DIR/summary.txt"
        return 0
    fi
    
    # 执行训练
    log_info "Starting training..."
    if "${cmd[@]}" 2>&1 | tee "$log_file"; then
        log_info "✓ Training completed successfully: $run_name"
        echo "$run_name: SUCCESS" >> "$LOG_DIR/summary.txt"
        return 0
    else
        log_error "✗ Training failed: $run_name"
        echo "$run_name: FAILED" >> "$LOG_DIR/summary.txt"
        return 1
    fi
}

# 运行实验一
run_exp1() {
    local dry_run="${1:-false}"
    local skip_existing="${2:-true}"
    
    log_section "实验一: SSAF模块有效性验证"
    
    local total=0
    local success=0
    local failed=0
    local skipped=0
    
    for run_name in "${!EXP1_MODELS[@]}"; do
        total=$((total + 1))
        
        model="${EXP1_MODELS[$run_name]}"
        loss="${EXP1_LOSS[$run_name]}"
        
        if train_model "exp1" "$run_name" "$model" "$loss" "$dry_run" "$skip_existing"; then
            if check_checkpoint "$model" "$loss"; then
                success=$((success + 1))
            else
                skipped=$((skipped + 1))
            fi
        else
            failed=$((failed + 1))
        fi
    done
    
    # 打印实验一总结
    log_section "实验一总结"
    log_info "Total: $total | Success: $success | Failed: $failed | Skipped: $skipped"
}

# 运行实验二
run_exp2() {
    local dry_run="${1:-false}"
    local skip_existing="${2:-true}"
    
    log_section "实验二: 组合损失有效性验证"
    
    local total=0
    local success=0
    local failed=0
    local skipped=0
    
    for run_name in "${!EXP2_MODELS[@]}"; do
        total=$((total + 1))
        
        model="${EXP2_MODELS[$run_name]}"
        loss="${EXP2_LOSS[$run_name]}"
        
        if train_model "exp2" "$run_name" "$model" "$loss" "$dry_run" "$skip_existing"; then
            if check_checkpoint "$model" "$loss"; then
                success=$((success + 1))
            else
                skipped=$((skipped + 1))
            fi
        else
            failed=$((failed + 1))
        fi
    done
    
    # 打印实验二总结
    log_section "实验二总结"
    log_info "Total: $total | Success: $success | Failed: $failed | Skipped: $skipped"
}

# 打印最终总结
print_final_summary() {
    log_section "消融实验完成"
    
    echo ""
    echo "实验结果汇总:"
    echo "============================================"
    
    if [[ -f "$LOG_DIR/summary.txt" ]]; then
        cat "$LOG_DIR/summary.txt"
    fi
    
    echo "============================================"
    echo ""
    echo "日志目录: $LOG_DIR"
    echo "Checkpoint目录: $CHECKPOINT_DIR"
    echo ""
    
    # 统计
    local total_runs
    total_runs=$(wc -l < "$LOG_DIR/summary.txt")
    local success_runs
    success_runs=$(grep -c "SUCCESS" "$LOG_DIR/summary.txt" || true)
    local failed_runs
    failed_runs=$(grep -c "FAILED" "$LOG_DIR/summary.txt" || true)
    local skipped_runs
    skipped_runs=$(grep -c "SKIPPED" "$LOG_DIR/summary.txt" || true)
    
    log_info "总计: $total_runs | 成功: $success_runs | 失败: $failed_runs | 跳过: $skipped_runs"
}

# ==================== 主函数 ====================

main() {
    local nohup_mode=false
    local dry_run=false
    local skip_existing=true
    local exp_to_run="all"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --nohup)
                nohup_mode=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --no-skip)
                skip_existing=false
                shift
                ;;
            --exp)
                exp_to_run="$2"
                shift 2
                ;;
            --help|-h)
                cat << EOF
消融实验训练脚本

用法: $0 [OPTIONS]

选项:
  --nohup          后台运行实验 (使用nohup)
  --dry-run        仅显示命令,不实际执行
  --no-skip        不跳过已有的checkpoint
  --exp <name>     运行特定实验 (exp1, exp2, all)
  --help, -h       显示此帮助信息

实验说明:
  exp1: SSAF模块有效性验证 (HRNet vs MS-HRNet, 使用BCE Loss)
  exp2: 组合损失有效性验证 (不同损失函数组合对比)
  all:  运行所有实验 (默认)

示例:
  # 前台运行所有实验
  $0

  # 后台运行所有实验
  $0 --nohup

  # 仅运行实验一
  $0 --exp exp1

  # 查看命令但不执行
  $0 --dry-run

后台运行说明:
  日志位置: logs/ablation_<timestamp>/
  查看进程: ps aux | grep train.py
  停止训练: pkill -f "train.py"
EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
        esac
    done
    
    # 显示配置
    log_section "消融实验配置"
    echo "GPU: $GPU"
    echo "Dataset: $DATASET_NAME"
    echo "Epochs: $EPOCHS"
    echo "Batch Size: $BATCH_SIZE"
    echo "Learning Rate: $LR"
    echo "Log Directory: $LOG_DIR"
    echo "Nohup Mode: $nohup_mode"
    echo "Dry Run: $dry_run"
    echo "Skip Existing: $skip_existing"
    echo "Experiments to Run: $exp_to_run"
    
    # 如果是nohup模式,重定向输出
    if [[ "$nohup_mode" == "true" ]]; then
        local main_log="${LOG_DIR}/main.log"
        log_info "Running in background mode..."
        log_info "Main log: $main_log"
        
        # 重新执行脚本,但不带--nohup参数
        local new_args=()
        [[ "$dry_run" == "true" ]] && new_args+=(--dry-run)
        [[ "$skip_existing" == "false" ]] && new_args+=(--no-skip)
        [[ "$exp_to_run" != "all" ]] && new_args+=(--exp "$exp_to_run")
        
        nohup "$0" "${new_args[@]}" > "$main_log" 2>&1 &
        local pid=$!
        
        log_info "Started background process with PID: $pid"
        log_info "Monitor progress: tail -f $main_log"
        log_info "Stop training: kill $pid"
        
        exit 0
    fi
    
    # 初始化summary文件
    echo "# Ablation Experiments Summary" > "$LOG_DIR/summary.txt"
    echo "# Started at: $(date)" >> "$LOG_DIR/summary.txt"
    echo "" >> "$LOG_DIR/summary.txt"
    
    # 运行实验
    case "$exp_to_run" in
        exp1)
            run_exp1 "$dry_run" "$skip_existing"
            ;;
        exp2)
            run_exp2 "$dry_run" "$skip_existing"
            ;;
        all)
            run_exp1 "$dry_run" "$skip_existing"
            run_exp2 "$dry_run" "$skip_existing"
            ;;
        *)
            log_error "Unknown experiment: $exp_to_run"
            echo "Valid options: exp1, exp2, all"
            exit 1
            ;;
    esac
    
    # 记录完成时间
    echo "" >> "$LOG_DIR/summary.txt"
    echo "# Finished at: $(date)" >> "$LOG_DIR/summary.txt"
    
    # 打印总结
    print_final_summary
    
    log_info "All experiments completed!"
}

# 执行主函数
main "$@"