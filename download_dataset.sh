#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mydata/hub

huggingface-cli download continuallearning/libero_10_image_task_0 --repo-type dataset --local-dir /mydata/LIBERO/libero_10_image_task_0
