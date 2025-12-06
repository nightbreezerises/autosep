#!/bin/bash
# ============================================================
# AutoSEP Prompt 优化脚本
# 仅运行 autosep/main.py 进行 Prompt 优化
# 读取 config.yaml 配置，支持后台运行
# ============================================================

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查是否安装了 yq
if ! command -v yq &> /dev/null; then
    echo "错误: 需要安装 yq 来解析 YAML 配置文件"
    echo "安装方法: pip install yq 或 conda install -c conda-forge yq"
    exit 1
fi

# ============================================================
# 读取环境配置
# ============================================================
CONDA_PATH=$(yq -r '.environment.conda_path' "$CONFIG_FILE")
CONDA_ENV=$(yq -r '.environment.conda_env' "$CONFIG_FILE")

# 激活 conda 环境
if [ "$CONDA_PATH" != "null" ] && [ -n "$CONDA_PATH" ]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "已激活 conda 环境: $CONDA_ENV"
fi

# ============================================================
# 读取配置
# ============================================================
DATA_DIR=$(yq -r '.paths.data_dir' "$CONFIG_FILE")
LOG_DIR=$(yq -r '.paths.log_dir' "$CONFIG_FILE")
GPU=$(yq -r '.paths.gpu' "$CONFIG_FILE")
DATASET=$(yq -r '.dataset' "$CONFIG_FILE")
MODEL=$(yq -r '.model // "Qwen2.5-VL-7B-Instruct"' "$CONFIG_FILE")
OUT_NUM=$(yq -r '.experiment.out_num' "$CONFIG_FILE")

# 处理相对路径
if [[ "$LOG_DIR" == ./* ]]; then
    LOG_DIR="${PROJECT_ROOT}/${LOG_DIR#./}"
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# ============================================================
# 生成日志文件名
# ============================================================
get_log_filename() {
    local base_name="optimize_${DATASET}"
    local log_file="${LOG_DIR}/${base_name}.log"
    
    if [ ! -f "$log_file" ]; then
        echo "$log_file"
        return
    fi
    
    local counter=1
    while [ -f "${LOG_DIR}/${base_name}(${counter}).log" ]; do
        ((counter++))
    done
    echo "${LOG_DIR}/${base_name}(${counter}).log"
}

LOG_FILE=$(get_log_filename)

# ============================================================
# 定义主执行函数
# ============================================================
run_optimize() {
    set -e

    echo "============================================================"
    echo "AutoSEP Prompt 优化"
    echo "配置文件: $CONFIG_FILE"
    echo "项目根目录: $PROJECT_ROOT"
    echo "日志文件: $LOG_FILE"
    echo "============================================================"

    # 设置 GPU
    if [ "$GPU" != "null" ] && [ -n "$GPU" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU"
        echo "设置 CUDA_VISIBLE_DEVICES=$GPU"
    fi

    # 读取 AutoSEP 参数
    N_TRAIN=$(yq -r '.autosep.n_train' "$CONFIG_FILE")
    N_VAL=$(yq -r '.autosep.n_val' "$CONFIG_FILE")
    N_TEST_AUTOSEP=$(yq -r '.autosep.n_test' "$CONFIG_FILE")
    ROUNDS=$(yq -r '.autosep.rounds' "$CONFIG_FILE")
    BEAM_SIZE=$(yq -r '.autosep.beam_size' "$CONFIG_FILE")
    MINIBATCH_SIZE=$(yq -r '.autosep.minibatch_size' "$CONFIG_FILE")
    N_GRADIENTS=$(yq -r '.autosep.n_gradients' "$CONFIG_FILE")
    MC_SAMPLES=$(yq -r '.autosep.mc_samples_per_step' "$CONFIG_FILE")
    MAX_EXPANSION=$(yq -r '.autosep.max_expansion_factor' "$CONFIG_FILE")
    TEST_EVAL=$(yq -r '.autosep.test_eval' "$CONFIG_FILE")
    AUTOSEP_TEMP=$(yq -r '.autosep.temperature' "$CONFIG_FILE")
    AUTOSEP_THREADS=$(yq -r '.autosep.max_threads' "$CONFIG_FILE")
    TRAIN_RATIO=$(yq -r '.autosep.train_ratio // 100.0' "$CONFIG_FILE")

    # 按百分比缩放样本数量（至少为 1）
    N_TRAIN=$(python3 -c "print(max(1, int($N_TRAIN * $TRAIN_RATIO / 100.0)))")
    N_VAL=$(python3 -c "print(max(1, int($N_VAL * $TRAIN_RATIO / 100.0)))")
    N_TEST_AUTOSEP=$(python3 -c "print(max(1, int($N_TEST_AUTOSEP * $TRAIN_RATIO / 100.0)))")

    echo "调试比例: train_ratio=${TRAIN_RATIO}%"
    echo "有效样本数: n_train=$N_TRAIN, n_val=$N_VAL, n_test=$N_TEST_AUTOSEP"

    # 设置模型环境变量
    export AUTOSEP_MODEL="$MODEL"
    echo "设置 AUTOSEP_MODEL=$MODEL"

    # 切换到项目根目录
    cd "$PROJECT_ROOT"

    # 设置 PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    # 获取数据集详细信息
    DATASET_INFO=$(python -c "
from data import get_dataset_dir, get_dataset_info
import os

dataset = '$DATASET'
data_dir = '$DATA_DIR'
dir_name = get_dataset_dir(dataset)
info = get_dataset_info(dataset)

if data_dir.startswith('./'):
    full_path = os.path.join('$PROJECT_ROOT', data_dir[2:], dir_name)
else:
    full_path = os.path.join(data_dir, dir_name)

path_exists = '✓' if os.path.exists(full_path) else '✗ (不存在)'

print(f'目录名: {dir_name}')
print(f'全称: {info[\"name\"]}')
print(f'类别数: {info[\"classes\"]}')
print(f'类型: {info[\"type\"]}')
print(f'路径: {full_path} {path_exists}')
" 2>/dev/null || echo "无法获取数据集信息")

    echo ""
    echo "============================================================"
    echo "配置信息"
    echo "============================================================"
    echo "数据集简称: $DATASET"
    echo "$DATASET_INFO"
    echo ""
    echo "运行参数:"
    echo "  数据目录: $DATA_DIR"
    echo "  模型: $MODEL"
    echo "  GPU: ${CUDA_VISIBLE_DEVICES:-未指定}"
    echo "  实验编号: $OUT_NUM"
    echo "  优化轮数: $ROUNDS"
    echo "  Beam Size: $BEAM_SIZE"
    echo "============================================================"
    echo ""
    echo "开始 Prompt 优化: 数据集=$DATASET, 模型=$MODEL"
    echo "============================================================"

    # 构建 test_eval 参数
    TEST_EVAL_FLAG=""
    if [ "$TEST_EVAL" = "true" ]; then
        TEST_EVAL_FLAG="--test_eval"
    fi

    # 运行 AutoSEP 优化
    # 注意: main.py 的 --model 参数只接受 sglang_qwen，实际模型由 AUTOSEP_MODEL 环境变量控制
    python autosep/main.py \
        --data_dir "$DATA_DIR" \
        --task_name "$DATASET" \
        --model "sglang_qwen" \
        --gradient_model "sglang_qwen" \
        --n_train "$N_TRAIN" \
        --n_val "$N_VAL" \
        --n_test "$N_TEST_AUTOSEP" \
        --rounds "$ROUNDS" \
        --beam_size "$BEAM_SIZE" \
        --minibatch_size "$MINIBATCH_SIZE" \
        --n_gradients "$N_GRADIENTS" \
        --mc_samples_per_step "$MC_SAMPLES" \
        --max_expansion_factor "$MAX_EXPANSION" \
        --temperature "$AUTOSEP_TEMP" \
        --max_threads "$AUTOSEP_THREADS" \
        --out_num "$OUT_NUM" \
        $TEST_EVAL_FLAG

    # 输出结果文件位置
    RESULT_DIR="autosep/results/${OUT_NUM}_${DATASET}"
    echo ""
    echo "============================================================"
    echo "Prompt 优化完成!"
    echo "============================================================"
    echo "结果目录: $RESULT_DIR"
    echo ""
    echo "生成的文件:"
    echo "  - ${RESULT_DIR}/apo_multi_${DATASET}_${OUT_NUM}.txt (优化后的 Prompt)"
    echo "  - ${RESULT_DIR}/${OUT_NUM}_train_attr.json (训练集属性缓存)"
    echo "  - ${RESULT_DIR}/${OUT_NUM}_test_attr.json (测试集属性缓存)"
    echo ""
    echo "下一步: 运行 run_classification.sh 进行分类评估"
    echo "============================================================"
    
    # 检查文件是否生成
    if [ -f "${RESULT_DIR}/${OUT_NUM}_test_attr.json" ]; then
        echo "✓ 属性缓存已保存，可以运行分类评估"
    else
        echo "✗ 警告: 属性缓存未生成，请检查日志"
    fi
}

# ============================================================
# 主入口
# ============================================================
if [ "$1" = "--foreground" ] || [ "$1" = "-f" ]; then
    run_optimize 2>&1 | tee "$LOG_FILE"
elif [ "$1" = "--background-worker" ]; then
    run_optimize
else
    echo "============================================================"
    echo "AutoSEP Prompt 优化 - 后台启动"
    echo "============================================================"
    echo "数据集: $DATASET"
    echo "模型: $MODEL"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "启动后台任务..."
    
    nohup bash "$0" --background-worker > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "后台进程 PID: $PID"
    echo ""
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止任务: kill $PID"
    echo "============================================================"
fi
