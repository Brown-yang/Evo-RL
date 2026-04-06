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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("aubo_arm")
@dataclass
class AuboArmConfig(RobotConfig):
    """LeRobot robot config for AUBO arms using `pyaubo_sdk` (RPC + RTDE)."""

    # AUBO controller endpoints
    robot_ip: str = "192.168.123.213"
    rpc_port: int = 30004
    rtde_port: int = 30010
    username: str = "aubo"
    password: str = "123456"

    # Control loop
    control_frequency: float = 125.0
    servo_lookahead: float = 0.0
    servo_gain: int = 0

    # Optional initial moveJ target (6 joints, radians)
    initial_target: Sequence[float] | None = None

    # Safety / smoothing
    max_jump: float = 0.08
    vel_limit: float = 80.0

    # Optional gripper (Lebai)
    use_gripper: bool = True
    gripper_ip: str = "192.168.123.118"
    gripper_port: int = 6888
    gripper_auto_calibrate: bool = True
    gripper_is_dataset_robot: bool = False

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

