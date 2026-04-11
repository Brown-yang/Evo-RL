#!/bin/bash

# LeRobot LoRA微调训练脚本
# 用于pi0.5_base模型在libero_10数据集上的离线训练

set -e  # 遇到错误立即退出

# ============ 环境变量配置 ============
export HF_HOME=/mydata/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/mydata/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ============ 路径配置 ============
PROJECT_DIR="/mydata/Evo-RL"
CONFIG_FILE="train_config.json"
MODEL_PATH="/mydata/models/pi0.5_base"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ============ 创建日志目录 ============
mkdir -p "${LOG_DIR}"

# ============ 打印配置信息 ============
echo "=========================================="
echo "LeRobot LoRA微调训练脚本"
echo "=========================================="
echo "项目目录: ${PROJECT_DIR}"
echo "配置文件: ${CONFIG_FILE}"
echo "模型路径: ${MODEL_PATH}"
echo "日志文件: ${LOG_FILE}"
echo "时间戳: ${TIMESTAMP}"
echo "=========================================="
echo ""

# ============ 进入项目目录 ============
cd "${PROJECT_DIR}"

# ============ 验证配置文件 ============
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 配置文件 ${CONFIG_FILE} 不存在"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 模型路径 ${MODEL_PATH} 不存在"
    exit 1
fi

echo "✓ 配置文件和模型路径验证通过"
echo ""

# ============ 开始训练 ============
echo "开始训练..."
echo "时间: $(date)"
echo ""

lerobot-train \
    --config_path="${CONFIG_FILE}" \
    --policy.path="${MODEL_PATH}" \
    --policy.repo_id=local/my_libero_pi05 \
    --policy.push_to_hub=false \
    --policy.freeze_vision_encoder=true \
    --policy.gradient_checkpointing=true \
    2>&1 | tee -a "${LOG_FILE}"

# ============ 训练完成 ============
TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ 训练完成！"
else
    echo "✗ 训练失败，退出码: ${TRAIN_EXIT_CODE}"
fi
echo "时间: $(date)"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

exit $TRAIN_EXIT_CODE
