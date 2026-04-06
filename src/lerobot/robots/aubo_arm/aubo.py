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

import logging
import threading
import time
from functools import cached_property
from typing import Optional, Sequence

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_aubo_arm import AuboArmConfig

MAX_JUMP = 0.08
DT_RT = 0.004
VEL_LIMIT = 80.0

logger = logging.getLogger(__name__)


class AuboArm(Robot):
    """AUBO arm robot implementation via `pyaubo_sdk`."""

    config_class = AuboArmConfig
    name = "aubo_arm"

    def __init__(self, config: AuboArmConfig) -> None:
        super().__init__(config)
        self.config = config

        self._control_frequency = float(config.control_frequency)
        self._control_period = 1.0 / self._control_frequency
        self._servo_lookahead = float(config.servo_lookahead)
        self._servo_gain = int(config.servo_gain)
        self._max_jump = float(config.max_jump if config.max_jump is not None else MAX_JUMP)
        self._vel_limit = float(config.vel_limit if config.vel_limit is not None else VEL_LIMIT)

        self._pyaubo_sdk = None
        self._rpc_client = None
        self._rtde_client = None
        self._robot_name = None
        self._robot_interface = None
        self._motion_control = None
        self._rtde_topic = None

        self._joint_lock = threading.Lock()
        self._joint_positions: Optional[np.ndarray] = None
        self._servo_enabled: bool = False
        self._last_servo_t: Optional[float] = None
        self._last_servo_warn_t: float = 0.0

        self._use_gripper = bool(config.use_gripper)
        self._gripper = None
        self._publish_thread: threading.Thread | None = None
        self._gripper_loop_alive = False
        self._target_gripper_pos_norm: float = 0.0

        self.cameras = make_cameras_from_configs(config.cameras)

    # ----------------------
    # Robot protocol methods
    # ----------------------
    @property
    def _arm_joint_names(self) -> list[str]:
        return [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "wrist_yaw",
        ]

    @property
    def _has_gripper(self) -> bool:
        return self._use_gripper

    @property
    def is_connected(self) -> bool:
        arm_connected = self._rpc_client is not None and self._rtde_client is not None
        cams_connected = all(cam.is_connected for cam in self.cameras.values())
        return bool(arm_connected and cams_connected)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        ft: dict[str, type | tuple] = {f"{j}.pos": float for j in self._arm_joint_names}
        if self._has_gripper:
            ft["gripper.pos"] = float
        for cam_key, cam_cfg in self.config.cameras.items():
            ft[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
        return ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        ft: dict[str, type] = {f"{j}.pos": float for j in self._arm_joint_names}
        if self._has_gripper:
            ft["gripper.pos"] = float
        return ft

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        try:
            import pyaubo_sdk  # type: ignore
        except ImportError as exc:  # pragma: no cover - hardware specific
            raise ImportError(
                "pyaubo_sdk is required for AuboArm. Please install the AUBO SDK."
            ) from exc

        self._pyaubo_sdk = pyaubo_sdk

        self._rpc_client = pyaubo_sdk.RpcClient()
        self._rpc_client.connect(self.config.robot_ip, int(self.config.rpc_port))
        self._rpc_client.login(self.config.username, self.config.password)
        self._rpc_client.setRequestTimeout(1000)

        self._rtde_client = pyaubo_sdk.RtdeClient()
        self._rtde_client.connect(self.config.robot_ip, int(self.config.rtde_port))
        self._rtde_client.login(self.config.username, self.config.password)

        robot_names = self._rpc_client.getRobotNames()
        if not robot_names:
            raise RuntimeError("No AUBO robot detected on the controller")
        self._robot_name = robot_names[0]
        self._robot_interface = self._rpc_client.getRobotInterface(self._robot_name)
        self._motion_control = self._robot_interface.getMotionControl()

        self._rtde_topic = self._setup_rtde_subscription()
        self._wait_for_initial_data()

        if self.config.initial_target is not None:
            self._movej_to_target(np.asarray(self.config.initial_target, dtype=float))

        if self._use_gripper:
            from .lebai_gripper import LebaiGripper

            self._gripper = LebaiGripper()
            self._gripper.connect(hostname=self.config.gripper_ip, port=int(self.config.gripper_port))
            self._gripper.activate(auto_calibrate=bool(self.config.gripper_auto_calibrate))
            self._target_gripper_pos_norm = float(self._gripper.get_current_position()) / 255.0
            if not self.config.gripper_is_dataset_robot:
                self._gripper_loop_alive = True
                self._publish_thread = threading.Thread(
                    target=self._publish_gripper_command,
                    daemon=True,
                    name="LebaiGripperPublisher",
                )
                self._publish_thread.start()

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    def get_joint_state(self) -> np.ndarray:
        with self._joint_lock:
            if self._joint_positions is None:
                raise RuntimeError("Joint state not initialized yet")
            return self._joint_positions.copy()

    def _publish_gripper_command(self):
        while self._gripper_loop_alive and self._gripper is not None:
            gripper_pos = int(np.clip(self._target_gripper_pos_norm, 0.0, 1.0) * 255.0)
            if gripper_pos < 150:
                gripper_pos = 0
            self._gripper.move(gripper_pos, 255, 10)
            time.sleep(0.1)

    def limit_joint_jump(self, q_current, q_target):
        q_current = np.asarray(q_current)
        q_target = np.asarray(q_target)
        delta = np.clip(q_target - q_current, -self._max_jump, self._max_jump)
        return (q_current + delta).tolist()

    def clip_velocity(self, current_q, target_q, vel_limit=None, dt=DT_RT):
        current_q = np.asarray(current_q, float)
        target_q = np.asarray(target_q, float)
        if vel_limit is None:
            vel_limit = self._vel_limit
        max_delta = vel_limit * dt
        delta = target_q - current_q
        clipped = np.clip(delta, -max_delta, max_delta)
        return (current_q + clipped).tolist()

    def _command_joint_targets(self, target_joint: np.ndarray) -> None:
        target_joint = np.asarray(target_joint, dtype=float)
        if target_joint.shape != (6,):
            raise ValueError(f"Expected 6 joint targets, got shape={target_joint.shape}")

        current_joint_state = self.get_joint_state()
        if not np.allclose(target_joint, current_joint_state, atol=0.001):
            target_joint = np.asarray(self.limit_joint_jump(current_joint_state, target_joint), dtype=float)
            target_joint = np.asarray(self.clip_velocity(current_joint_state, target_joint), dtype=float)
        else:
            target_joint = current_joint_state

        if self._motion_control is None:
            raise RuntimeError("AUBO motion control not initialized. Did you call connect()?")

        now = time.time()
        servo_mode_ok = self._motion_control.isServoModeEnabled()
        if not servo_mode_ok:
            self._enable_servo_mode()
            servo_mode_ok = self._motion_control.isServoModeEnabled()

        if not servo_mode_ok:
            if self._servo_enabled:
                logger.warning("Servo mode is disabled; joint commands will have no effect.")
                self._servo_enabled = False
            if now - self._last_servo_warn_t > 1.0:
                self._last_servo_warn_t = now
                logger.warning("Servo mode not enabled. Check controller state / safety IO / pendant.")
            return

        if not self._servo_enabled:
            self._servo_enabled = True
            logger.info("Servo mode enabled; resuming joint commands.")

        if self._last_servo_t is None:
            period = self._control_period
        else:
            period = now - self._last_servo_t
        self._last_servo_t = now
        t_move = float(np.clip(period, 0.005, 0.1))

        self._motion_control.servoJoint(
            target_joint.tolist(),
            self._control_period,
            0.0,
            t_move,
            self._servo_lookahead,
            self._servo_gain,
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        joints = self.get_joint_state()
        obs: RobotObservation = {f"{name}.pos": float(val) for name, val in zip(self._arm_joint_names, joints)}

        if self._use_gripper and self._gripper is not None:
            raw_pos = float(self._gripper.get_current_position())
            obs["gripper.pos"] = float(np.clip(raw_pos / 255.0, 0.0, 1.0))

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal = np.array([float(action[f"{j}.pos"]) for j in self._arm_joint_names], dtype=float)
        self._command_joint_targets(goal)

        sent: RobotAction = {f"{j}.pos": float(v) for j, v in zip(self._arm_joint_names, goal)}

        if self._use_gripper and "gripper.pos" in action:
            self._target_gripper_pos_norm = float(np.clip(float(action["gripper.pos"]), 0.0, 1.0))
            sent["gripper.pos"] = self._target_gripper_pos_norm

        return sent

    # ----------------------
    # Internal helpers
    # ----------------------
    def _setup_rtde_subscription(self):
        topic = self._rtde_client.setTopic(
            False, ["R1_actual_q"], self._control_frequency, 0
        )
        self._rtde_client.subscribe(topic, self._rtde_callback)
        return topic

    def _rtde_callback(self, parser):
        data = parser.popVectorDouble()
        if data is None:
            return
        joints = np.asarray(data, dtype=float)
        with self._joint_lock:
            self._joint_positions = joints

    def _wait_for_initial_data(self, timeout: float = 50.0) -> np.ndarray:
        start = time.time()
        while True:
            with self._joint_lock:
                if self._joint_positions is not None:
                    return self._joint_positions
            if time.time() - start > timeout:
                raise TimeoutError("AUBORobot joint data timeout")
            time.sleep(0.01)

    def _enable_servo_mode(self) -> None:
        self._motion_control.setServoMode(True)
        for _ in range(50):
            if self._motion_control.isServoModeEnabled():
                logger.info("Servo mode enabled")
                return
            time.sleep(0.01)
        raise RuntimeError("Failed to enable servo mode on AUBO robot")

    def _disable_servo_mode(self) -> None:
        self._motion_control.setServoMode(False)
        for _ in range(50):
            if not self._motion_control.isServoModeEnabled():
                return
            time.sleep(0.01)

    def _wait_arrival(self) -> None:
        while self._motion_control.getExecId() != -1:
            time.sleep(0.05)

    def _movej_to_target(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float).tolist()
        self._motion_control.setSpeedFraction(1.0)
        self._motion_control.moveJoint(target, 2.0, 1.0, 0, 0)
        self._wait_arrival()
        logger.info("Reached initial target joint position.")

    # ----------------------
    # Lifecycle helpers
    # ----------------------
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._gripper_loop_alive = False
        if self._publish_thread is not None and self._publish_thread.is_alive():
            self._publish_thread.join(timeout=2.0)

        try:
            if self._motion_control is not None:
                self._disable_servo_mode()
        except Exception:
            logger.exception("Failed to disable servo mode during disconnect")

        try:
            if self._rtde_topic is not None and self._rtde_client is not None:
                self._rtde_client.removeTopic(False, self._rtde_topic)
        except Exception:
            logger.exception("Failed to remove RTDE topic during disconnect")

        try:
            if self._rtde_client is not None:
                self._rtde_client.disconnect()
        except Exception:
            logger.exception("Failed to disconnect RTDE client")

        try:
            if self._rpc_client is not None:
                self._rpc_client.disconnect()
        except Exception:
            logger.exception("Failed to disconnect RPC client")

        try:
            if self._gripper is not None:
                self._gripper.disconnect()
        except Exception:
            logger.exception("Failed to disconnect gripper")

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                logger.exception("Failed to disconnect a camera")

        self._rpc_client = None
        self._rtde_client = None
        self._motion_control = None
        self._robot_interface = None
        self._robot_name = None
        self._rtde_topic = None
        self._joint_positions = None

        logger.info(f"{self} disconnected.")

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            pass
