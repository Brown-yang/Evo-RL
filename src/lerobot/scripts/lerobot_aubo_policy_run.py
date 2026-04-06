#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a trained LeRobot policy on a real AUBO arm (no dataset recording).

Example:

```
lerobot-aubo-policy-run \\
  --robot.type=aubo_arm \\
  --robot.robot_ip=192.168.1.10 \\
  --robot.id=my_aubo \\
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
  --dataset.repo_id=local/my_ds \\
  --dataset.root=/path/to/dataset \\
  --policy.path=/path/to/checkpoints/030000/pretrained_model \\
  --single_task="pick the cube" \\
  --control_time_s=120 \\
  --fps=30
```

`--policy.path` must point to a folder containing `config.json` and `model.safetensors`
(typically `.../checkpoints/<step>/pretrained_model`).

`--dataset.repo_id` / `--dataset.root` must refer to the **same** dataset (or compatible
features/stats) used for training so normalization matches the policy.

If your training robot type differs from `aubo_arm`, set `--strict_dataset_compat=false`
(you are responsible for observation/action feature alignment).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lerobot.robots.aubo_arm  # noqa: F401  # registers AuboArmConfig for CLI
from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.scripts.recording_hil import (
    ACPInferenceConfig,
    _capture_policy_runtime_state,
    _predict_policy_action_with_acp_inference,
)
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, sanity_check_dataset_robot_compatibility
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class AuboPolicyDatasetRefConfig:
    repo_id: str
    root: str | Path | None = None
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class AuboPolicyRunConfig:
    robot: RobotConfig
    dataset: AuboPolicyDatasetRefConfig
    policy: PreTrainedConfig | None = None
    single_task: str | None = None
    fps: int = 30
    control_time_s: int | float = 60
    display_data: bool = False
    display_compressed_images: bool = False
    acp_inference: ACPInferenceConfig = field(default_factory=ACPInferenceConfig)
    strict_dataset_compat: bool = False
    communication_retry_timeout_s: float = 2.0
    communication_retry_interval_s: float = 0.1

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("This script requires `--policy.path=...` to a pretrained checkpoint directory.")
        cli_overrides = parser.get_cli_overrides("policy")
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = policy_path

        if self.acp_inference.use_cfg and not self.acp_inference.enable:
            raise ValueError("`acp_inference.use_cfg=true` requires `acp_inference.enable=true`.")
        if self.acp_inference.cfg_beta < 0:
            raise ValueError("`acp_inference.cfg_beta` must be >= 0.")
        if self.communication_retry_timeout_s < 0:
            raise ValueError("`communication_retry_timeout_s` must be >= 0.")
        if self.communication_retry_interval_s <= 0:
            raise ValueError("`communication_retry_interval_s` must be > 0.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _run_with_connection_retry(
    action_name: str,
    fn,
    *,
    timeout_s: float,
    interval_s: float,
):
    timeout_s = max(timeout_s, 0.0)
    interval_s = max(interval_s, 0.0)
    deadline_t = time.perf_counter() + timeout_s
    attempts = 0
    first_error: ConnectionError | None = None

    while True:
        attempts += 1
        try:
            return fn()
        except ConnectionError as error:
            if first_error is None:
                first_error = error
                logging.warning(
                    "%s failed (%s); retrying up to %.2fs",
                    action_name,
                    error,
                    timeout_s,
                )
            if timeout_s <= 0.0:
                raise
            remaining_s = deadline_t - time.perf_counter()
            if remaining_s <= 0.0:
                raise
            sleep_s = interval_s if interval_s > 0.0 else remaining_s
            time.sleep(min(sleep_s, remaining_s))


@parser.wrap()
def aubo_policy_run(cfg: AuboPolicyRunConfig) -> None:
    init_logging()
    assert cfg.policy is not None
    logging.info(
        "lerobot-aubo-policy-run: robot.type=%s dataset.repo_id=%s policy.path=%s strict_dataset_compat=%s",
        cfg.robot.type,
        cfg.dataset.repo_id,
        cfg.policy.pretrained_path,
        cfg.strict_dataset_compat,
    )

    robot = make_robot_from_config(cfg.robot)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)

    if cfg.strict_dataset_compat:
        use_videos = bool(dataset.video_keys)
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=robot_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=use_videos,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=use_videos,
            ),
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, dataset_features)
    else:
        logging.warning(
            "strict_dataset_compat=false: skipping dataset vs robot metadata check. "
            "Ensure camera keys, action names, and joint semantics match training data."
        )

    policy = make_policy(
        cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.dataset.rename_map or None,
    )
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
        },
    )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    cond_state: dict[str, Any] | None = None
    uncond_state: dict[str, Any] | None = None
    if cfg.acp_inference.enable and cfg.acp_inference.use_cfg:
        cond_state = _capture_policy_runtime_state(policy)
        uncond_state = _capture_policy_runtime_state(policy)

    listener, events = init_keyboard_listener()
    if cfg.display_data:
        init_rerun(session_name="aubo_policy_run")

    robot.connect()

    device = get_safe_torch_device(cfg.policy.device)
    start_t = time.perf_counter()
    try:
        while time.perf_counter() - start_t < cfg.control_time_s:
            loop_t = time.perf_counter()
            if events.get("exit_early"):
                events["exit_early"] = False
                logging.info("Exit requested (keyboard); stopping control loop.")
                break

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )

            policy_action = _predict_policy_action_with_acp_inference(
                observation_frame=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=cfg.policy.use_amp,
                task=cfg.single_task,
                robot_type=robot.robot_type,
                acp_inference=cfg.acp_inference,
                cond_runtime_state=cond_state,
                uncond_runtime_state=uncond_state,
            )
            act = make_robot_action(policy_action, dataset.features)
            robot_action_to_send = robot_action_processor((act, obs))

            _run_with_connection_retry(
                "robot.send_action",
                lambda: robot.send_action(robot_action_to_send),
                timeout_s=cfg.communication_retry_timeout_s,
                interval_s=cfg.communication_retry_interval_s,
            )

            if cfg.display_data:
                log_rerun_data(
                    observation=obs_processed,
                    action=act,
                    compress_images=cfg.display_compressed_images,
                )

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1.0 / max(cfg.fps, 1) - dt_s, 0.0))
    finally:
        if listener is not None and hasattr(listener, "stop"):
            listener.stop()
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    aubo_policy_run()


if __name__ == "__main__":
    main()
