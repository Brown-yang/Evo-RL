#!/bin/bash
# EHPL 偏好训练脚本 - 快速验证版
# 使用已有的 libero10_p2_small 偏好对 (37对, 前5个episode)

set -e

export HF_HOME=/mydata/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/mydata/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

lerobot-train --dataset.repo_id=local/ehpl_libero10 --dataset.root=/mydata/LIBERO/libero_10_image_task_0 --policy.path=/mydata/Evo-RL/outputs/train/pi05_expert_libero10_task0/checkpoints/010000/pretrained_model --policy.repo_id=local/ehpl_pi05 --policy.push_to_hub=false --policy.freeze_vision_encoder=true --policy.gradient_checkpointing=true --batch_size=4 --num_workers=0 --steps=200 --save_freq=100 --log_freq=10 --output_dir=outputs/train/ehpl_pi05_sanity --job_name=ehpl_pi05_sanity --wandb.enable=false --ehpl.enable=true --ehpl.pairs_parquet=/mydata/Evo-RL/ehpl/pairs/libero10_task0_p2/pairs.parquet --ehpl.beta=1.0 --ehpl.hf_home=/mydata/hub --ehpl.offline=true
