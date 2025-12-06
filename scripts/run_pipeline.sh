#!/bin/bash
# ============================================================
# AutoSEP Pipeline Runner
# 读取 config.yaml 配置，依次运行 autosep/main.py 和 classification.py
# 支持后台运行 (nohup) 和自动激活 conda 环境
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

# 检查是否安装了 yq（用于解析 YAML）
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
# 生成日志文件名（支持递增命名）
# ============================================================
get_log_filename() {
    local base_name="pipeline_${DATASET}"
    local log_file="${LOG_DIR}/${base_name}.log"
    
    if [ ! -f "$log_file" ]; then
        echo "$log_file"
        return
    fi
    
    # 文件已存在，查找递增编号
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
run_pipeline() {
    set -e  # 遇到错误立即退出

    echo "============================================================"
    echo "AutoSEP Pipeline"
    echo "配置文件: $CONFIG_FILE"
    echo "项目根目录: $PROJECT_ROOT"
    echo "日志文件: $LOG_FILE"
    echo "============================================================"

    # 设置 GPU (CUDA_VISIBLE_DEVICES)
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

    # 读取 Classification 参数
    CLASS_MODE=$(yq -r '.classification.mode' "$CONFIG_FILE")
    CLASS_N_TEST=$(yq -r '.classification.n_test' "$CONFIG_FILE")
    PROMPT_IDX=$(yq -r '.classification.prompt_idx' "$CONFIG_FILE")
    CLASS_PARALLEL=$(yq -r '.classification.parallel' "$CONFIG_FILE")
    CLASS_GENERATE=$(yq -r '.classification.generate' "$CONFIG_FILE")
    CLASS_ATTRIBUTES=$(yq -r '.classification.attributes' "$CONFIG_FILE")
    CLASS_TEMP=$(yq -r '.classification.temperature' "$CONFIG_FILE")
    CLASS_THREADS=$(yq -r '.classification.max_threads' "$CONFIG_FILE")
    TEST_RATIO=$(yq -r '.classification.test_ratio // 100.0' "$CONFIG_FILE")

    # 按百分比缩放样本数量（至少为 1）
    N_TRAIN=$(python3 -c "print(max(1, int($N_TRAIN * $TRAIN_RATIO / 100.0)))")
    N_VAL=$(python3 -c "print(max(1, int($N_VAL * $TRAIN_RATIO / 100.0)))")
    N_TEST_AUTOSEP=$(python3 -c "print(max(1, int($N_TEST_AUTOSEP * $TRAIN_RATIO / 100.0)))")
    CLASS_N_TEST=$(python3 -c "print(max(1, int($CLASS_N_TEST * $TEST_RATIO / 100.0)))")

    echo "调试比例: train_ratio=${TRAIN_RATIO}%, test_ratio=${TEST_RATIO}%"
    echo "AutoSEP 有效样本数: n_train=$N_TRAIN, n_val=$N_VAL, n_test=$N_TEST_AUTOSEP"
    echo "分类评估有效样本数: n_test=$CLASS_N_TEST"

    # 设置模型环境变量（用于 api_utils.py 动态选择模型）
    export AUTOSEP_MODEL="$MODEL"
    echo "设置 AUTOSEP_MODEL=$MODEL"

    # 切换到项目根目录
    cd "$PROJECT_ROOT"

    # 设置 PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    # 获取数据集详细信息
    DATASET_INFO=$(python -c "
from data import get_dataset_dir, get_dataset_info, DATASET_MAPPING
import os

dataset = '$DATASET'
data_dir = '$DATA_DIR'

# 获取目录名和信息
dir_name = get_dataset_dir(dataset)
info = get_dataset_info(dataset)

# 构建完整路径
if data_dir.startswith('./'):
    full_path = os.path.join('$PROJECT_ROOT', data_dir[2:], dir_name)
else:
    full_path = os.path.join(data_dir, dir_name)

# 检查路径是否存在
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
    echo ""
    echo "支持的模型:"
    echo "  - Qwen2.5-VL-7B-Instruct (默认)"
    echo "  - Qwen3-VL-8B-Instruct"
    echo ""
    echo "支持的数据集:"
    echo "  细粒度: cub, dog, flower, car, pet, aircraft, food, birdsnap"
    echo "  通用:   caltech101, caltech256, dtd, eurosat, ucf, sun397"
    echo "  ImageNet: imagenet_1k, imagenet_a, imagenet_r, imagenet_sketch, imagenet_v2"
    echo "============================================================"
    echo ""
    echo "开始处理: 数据集=$DATASET, 模型=$MODEL"
    echo "============================================================"

    # 构建 test_eval 参数
    TEST_EVAL_FLAG=""
    if [ "$TEST_EVAL" = "true" ]; then
        TEST_EVAL_FLAG="--test_eval"
    fi

    # ============================================================
    # Step 1: 运行 AutoSEP 优化 (autosep/main.py)
    # ============================================================
    echo ""
    echo "[Step 1/2] 运行 AutoSEP 优化..."
    echo ""

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

    echo ""
    echo "[Step 1/2] AutoSEP 优化完成!"
    echo ""

    # ============================================================
    # Step 2: 运行分类评估 (classification.py)
    # ============================================================
    echo "[Step 2/2] 运行分类评估..."

    # 构建可选参数
    CLASS_FLAGS=""
    if [ "$CLASS_PARALLEL" = "true" ]; then
        CLASS_FLAGS="$CLASS_FLAGS --parallel"
    fi
    if [ "$CLASS_GENERATE" = "true" ]; then
        CLASS_FLAGS="$CLASS_FLAGS --generate"
    fi
    if [ "$CLASS_ATTRIBUTES" = "true" ]; then
        CLASS_FLAGS="$CLASS_FLAGS --attributes"
    fi

    # 注意: classification.py 的 --model 参数只接受 sglang_qwen，实际模型由 AUTOSEP_MODEL 环境变量控制
    python classification.py \
        --result_folder autosep \
        --data_dir "$DATA_DIR" \
        --task_name "$DATASET" \
        --model "sglang_qwen" \
        --exp "$OUT_NUM" \
        --mode "$CLASS_MODE" \
        --n_test "$CLASS_N_TEST" \
        --prompt_idx "$PROMPT_IDX" \
        --temperature "$CLASS_TEMP" \
        --max_threads "$CLASS_THREADS" \
        --out_num "$OUT_NUM" \
        $CLASS_FLAGS

    echo ""
    echo "[Step 2/2] 分类评估完成!"
    echo ""

    # 输出结果文件位置
    RESULT_FILE="autosep/results/${OUT_NUM}_${DATASET}/evaluate/${OUT_NUM}_${DATASET}_${PROMPT_IDX}_${OUT_NUM}.txt"
    echo "============================================================"
    echo "实验完成: 数据集=$DATASET"
    echo "结果文件: $RESULT_FILE"
    if [ -f "$RESULT_FILE" ]; then
        echo ""
        echo "分类结果摘要:"
        grep -E "Accuracy|F1" "$RESULT_FILE" | head -1
    fi
    echo "============================================================"
}

# ============================================================
# 主入口：后台运行或前台运行
# ============================================================
if [ "$1" = "--foreground" ] || [ "$1" = "-f" ]; then
    # 前台运行（用于调试）
    run_pipeline 2>&1 | tee "$LOG_FILE"
elif [ "$1" = "--background-worker" ]; then
    # 后台工作进程（内部使用）
    run_pipeline
else
    # 后台运行 (nohup)
    echo "============================================================"
    echo "AutoSEP Pipeline - 后台启动"
    echo "============================================================"
    echo "数据集: $DATASET"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "启动后台任务..."
    
    # 使用 nohup 后台运行自身，输出重定向到日志文件
    nohup bash "$0" --background-worker > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "后台进程 PID: $PID"
    echo ""
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止任务: kill $PID"
    echo "============================================================"
fi
