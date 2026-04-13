#!/bin/bash

# LeRobot Expert-Only 微调训练脚本
# 冻结VLM，只训练Action Expert和投影层

set -e

export HF_HOME=/mydata/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/mydata/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

PROJECT_DIR="/mydata/Evo-RL"
CONFIG_FILE="train_config.json"
MODEL_PATH="/mydata/models/pi0.5_base"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_expert_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

echo "=========================================="
echo "Pi0.5 Expert-Only 微调"
echo "模型: ${MODEL_PATH}"
echo "配置: ${CONFIG_FILE}"
echo "日志: ${LOG_FILE}"
echo "=========================================="

lerobot-train --config_path="${CONFIG_FILE}" --policy.path="${MODEL_PATH}" --policy.repo_id=local/pi05_expert_libero10_task0 --policy.push_to_hub=false --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.gradient_checkpointing=true 2>&1 | tee -a "${LOG_FILE}"

TRAIN_EXIT_CODE=$?
echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "训练完成！"
else
    echo "训练失败，退出码: ${TRAIN_EXIT_CODE}"
fi
exit $TRAIN_EXIT_CODE
