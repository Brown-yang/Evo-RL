#!/usr/bin/env python3
"""
简单调试脚本：查看训练时的图片数据
"""

import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/mydata/hub'

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def debug_training_data():
    """查看训练时的实际输入数据"""

    # 创建调试目录
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("调试训练数据 - 查看输入图片")
    print("=" * 60)

    # 加载数据集
    print("\n1. 加载数据集...")
    dataset = LeRobotDataset(
        repo_id="local/libero_10_no_noops_1.0.0_lerobot",
        root="/mydata/LIBERO/libero_10_no_noops_1.0.0_lerobot"
    )
    print(f"   ✓ 数据集加载成功，共 {len(dataset)} 帧")

    # 获取第一个样本
    print("\n2. 获取第一个样本...")
    sample = dataset[0]
    print(f"   ✓ 样本获取成功")
    print(f"   样本中的键: {list(sample.keys())}")

    # 查看图片键
    print("\n3. 查看可用的图片键...")
    image_keys = [k for k in sample.keys() if 'image' in k.lower()]
    for key in image_keys:
        data = sample[key]
        if isinstance(data, torch.Tensor):
            print(f"   - {key}: shape={data.shape}, dtype={data.dtype}")
        else:
            print(f"   - {key}: type={type(data)}")

    # 保存原始图片
    print("\n4. 保存原始图片...")

    # 保存全局图片 (image)
    if "observation.images.image" in sample:
        img_data = sample["observation.images.image"]
        save_image(img_data, debug_dir / "01_image.png", "全局图片 (observation.images.image)")

    # 保存腕部图片 (wrist_image)
    if "observation.images.wrist_image" in sample:
        img_data = sample["observation.images.wrist_image"]
        save_image(img_data, debug_dir / "02_wrist_image.png", "腕部图片 (observation.images.wrist_image)")

    # 保存多个样本用于对比
    print("\n5. 保存多个样本用于对比...")
    for sample_idx in range(1, min(5, len(dataset))):
        print(f"\n   处理样本 {sample_idx}...")
        try:
            sample = dataset[sample_idx]
            sample_dir = debug_dir / f"sample_{sample_idx}"
            sample_dir.mkdir(exist_ok=True)

            # 保存全局图片
            if "observation.images.image" in sample:
                img_data = sample["observation.images.image"]
                save_image(img_data, sample_dir / "image.png", f"样本{sample_idx} - 全局图片")

            # 保存腕部图片
            if "observation.images.wrist_image" in sample:
                img_data = sample["observation.images.wrist_image"]
                save_image(img_data, sample_dir / "wrist_image.png", f"样本{sample_idx} - 腕部图片")
        except Exception as e:
            print(f"     ✗ 处理样本{sample_idx}时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"✓ 调试完成！图片已保存到: {debug_dir}")
    print("=" * 60)
    print("\n查看图片:")
    print(f"  ls -la {debug_dir}/")
    print(f"\n你可以在IDE中打开这些图片文件查看")

def save_image(img_tensor, save_path, description):
    """保存图片"""
    try:
        # 转换为numpy
        if isinstance(img_tensor, torch.Tensor):
            img_array = img_tensor.cpu().numpy()
        else:
            img_array = np.array(img_tensor)

        print(f"   处理 {description}...")
        print(f"     - 原始形状: {img_array.shape}")
        print(f"     - 数据类型: {img_array.dtype}")
        print(f"     - 值范围: [{img_array.min():.2f}, {img_array.max():.2f}]")

        # 处理数据类型和范围
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

        # 处理通道顺序 (CHW -> HWC)
        if len(img_array.shape) == 3:
            if img_array.shape[0] in [3, 4]:  # 如果第一维是通道数
                img_array = np.transpose(img_array, (1, 2, 0))

        # 如果是RGBA，转换为RGB
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        print(f"     - 最终形状: {img_array.shape}")

        # 保存
        img = Image.fromarray(img_array)
        img.save(save_path)
        print(f"   ✓ 已保存: {save_path}")

    except Exception as e:
        print(f"   ✗ 保存失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_data()
