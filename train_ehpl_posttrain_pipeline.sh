#!/bin/bash
# EHPL：用训练好的 Judge 给 pairs 打 margin，再基于 base VLA 做 post-training（lerobot-train --ehpl.enable）
#
# 使用前请按需修改下面变量路径。

set -euo pipefail

export HF_HOME="${HF_HOME:-/mydata/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mydata/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

# --- 数据与 base VLA（SFT / expert）---
DATASET_ROOT="/mydata/LIBERO/libero_10_image_task_0"
BASE_POLICY="/mydata/Evo-RL/outputs/train/pi05_expert_libero10_task0/checkpoints/010000/pretrained_model"

# --- 偏好对（原始）与 Judge checkpoint ---
PAIRS_IN="/mydata/Evo-RL/ehpl/pairs/libero10_p2_full/pairs.parquet"
JUDGE_CKPT="${JUDGE_CKPT:?请设置 JUDGE_CKPT 为 train_judge.py 保存的 .pt 路径}"

# --- 输出：带 margin 的 pairs（infer_judge 会追加 score_w/score_l/margin/pred_wins）---
PAIRS_SCORED="/mydata/Evo-RL/ehpl/pairs/libero10_p2_full/pairs_scored_by_judge.parquet"

# --- EHPL post-training 输出 ---
OUT_DIR="${OUT_DIR:-outputs/train/ehpl_pi05_posttrain_from_judge}"
JOB_NAME="${JOB_NAME:-ehpl_pi05_posttrain_from_judge}"

# libero10_p2_full 一般为 h_seg=32, act_dim=7（与 pairs 一致即可）
ACTION_DIM="${ACTION_DIM:-7}"
H_SEG="${H_SEG:-32}"

# 可选：只用 margin 足够大的 pair，减少噪声（需要 PAIRS_SCORED 含 margin 列）
MIN_PAIR_MARGIN="${MIN_PAIR_MARGIN:-0.0}"

echo "[1/2] Scoring pairs with trained judge -> ${PAIRS_SCORED}"
python -m lerobot.ehpl.infer_judge \
  --ckpt "${JUDGE_CKPT}" \
  --parquet "${PAIRS_IN}" \
  --out_parquet "${PAIRS_SCORED}" \
  --context_key context \
  --action_w_key action_w \
  --action_l_key action_l \
  --action_dim "${ACTION_DIM}" \
  --T "${H_SEG}" \
  --batch_size 512 \
  --num_workers 0

echo "[2/2] EHPL post-training base VLA (pi0.5) with scored pairs"
lerobot-train \
  --dataset.repo_id=local/ehpl_libero10 \
  --dataset.root="${DATASET_ROOT}" \
  --policy.path="${BASE_POLICY}" \
  --policy.repo_id=local/ehpl_pi05 \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --batch_size=4 \
  --num_workers=0 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=200 \
  --output_dir="${OUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --wandb.enable=false \
  --ehpl.enable=true \
  --ehpl.pairs_parquet="${PAIRS_SCORED}" \
  --ehpl.beta=1.0 \
  --ehpl.min_pair_margin="${MIN_PAIR_MARGIN}" \
  --ehpl.hf_home="${HF_HOME}" \
  --ehpl.offline=true

echo "Done. Checkpoints under: ${OUT_DIR}"
