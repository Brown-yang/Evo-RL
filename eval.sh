#!/bin/bash
export HF_HOME=/mydata/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/mydata/hub

lerobot-eval --output_dir=./eval_logs_expert_task0/ --env.type=libero --env.task=libero_10 --env.task_ids='[0]' --eval.batch_size=1 --eval.n_episodes=10 --policy.path=/mydata/Evo-RL/outputs/train/pi05_expert_libero10_task0/checkpoints/010000/pretrained_model --policy.n_action_steps=10 --env.max_parallel_tasks=1 --rename_map='{"observation.images.image": "observation.images.base_0_rgb", "observation.images.image2": "observation.images.left_wrist_0_rgb"}'
