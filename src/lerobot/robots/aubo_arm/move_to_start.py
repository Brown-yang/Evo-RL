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

"""Move AUBO arm (and optional gripper) to a fixed start pose."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from lerobot.robots.aubo_arm.aubo import AuboArm
from lerobot.robots.aubo_arm.config_aubo_arm import AuboArmConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_START_JOINTS = [0.0, 0.85, -1.309, -0.87, -1.57, 0.0, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move AUBO to start joints.")
    parser.add_argument("--robot-ip", default="192.168.123.213")
    parser.add_argument("--rpc-port", type=int, default=30004)
    parser.add_argument("--rtde-port", type=int, default=30010)
    parser.add_argument("--username", default="aubo")
    parser.add_argument("--password", default="123456")

    parser.add_argument("--use-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-ip", default="192.168.123.118")
    parser.add_argument("--gripper-port", type=int, default=6888)
    parser.add_argument("--gripper-auto-calibrate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-is-dataset-robot", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--post-hold-s", type=float, default=1.0)
    parser.add_argument("--joint-tol", type=float, default=0.02)
    parser.add_argument("--gripper-tol", type=float, default=0.05)
    parser.add_argument("--max-jump", type=float, default=0.01, help="Per-step max joint jump (rad)")
    parser.add_argument("--vel-limit", type=float, default=2.0, help="Joint speed limit (rad/s)")
    parser.add_argument(
        "--skip-disconnect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip explicit robot.disconnect() to avoid SDK crash during teardown",
    )
    parser.add_argument(
        "--start-joints",
        type=float,
        nargs=7,
        default=DEFAULT_START_JOINTS,
        help="7 values: 6 arm joints (rad) + gripper [0,1]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_arm = np.asarray(args.start_joints[:6], dtype=float)
    target_gripper = float(args.start_joints[6])

    cfg = AuboArmConfig(
        robot_ip=args.robot_ip,
        rpc_port=args.rpc_port,
        rtde_port=args.rtde_port,
        username=args.username,
        password=args.password,
        use_gripper=bool(args.use_gripper),
        gripper_ip=args.gripper_ip,
        gripper_port=args.gripper_port,
        gripper_auto_calibrate=bool(args.gripper_auto_calibrate),
        gripper_is_dataset_robot=bool(args.gripper_is_dataset_robot),
        max_jump=float(args.max_jump),
        vel_limit=float(args.vel_limit),
        cameras={},
    )

    robot = AuboArm(cfg)
    period = 1.0 / max(args.hz, 1.0)

    logger.info(f"Target arm joints: {target_arm.tolist()}")
    logger.info(f"Target gripper: {target_gripper:.3f}")
    logger.info(f"Control rate: {args.hz:.1f} Hz")
    logger.info(f"Smoothing params: max_jump={args.max_jump:.4f}, vel_limit={args.vel_limit:.3f}")

    try:
        robot.connect()
        logger.info("Robot connected.")

        steady_hits = 0
        steady_need = max(1, int(args.hz * 0.5))
        t0 = time.time()

        while time.time() - t0 < args.timeout_s:
            loop_t = time.time()

            action = {
                "shoulder_pan.pos": float(target_arm[0]),
                "shoulder_lift.pos": float(target_arm[1]),
                "elbow_flex.pos": float(target_arm[2]),
                "wrist_flex.pos": float(target_arm[3]),
                "wrist_roll.pos": float(target_arm[4]),
                "wrist_yaw.pos": float(target_arm[5]),
            }
            if args.use_gripper:
                action["gripper.pos"] = target_gripper

            robot.send_action(action)
            obs = robot.get_observation()

            current_arm = np.array(
                [
                    obs["shoulder_pan.pos"],
                    obs["shoulder_lift.pos"],
                    obs["elbow_flex.pos"],
                    obs["wrist_flex.pos"],
                    obs["wrist_roll.pos"],
                    obs["wrist_yaw.pos"],
                ],
                dtype=float,
            )
            arm_err = float(np.max(np.abs(current_arm - target_arm)))

            gripper_ok = True
            if args.use_gripper and "gripper.pos" in obs:
                gripper_err = abs(float(obs["gripper.pos"]) - target_gripper)
                gripper_ok = gripper_err <= args.gripper_tol

            if arm_err <= args.joint_tol and gripper_ok:
                steady_hits += 1
            else:
                steady_hits = 0

            if steady_hits >= steady_need:
                logger.info("Reached start pose and held steady.")
                if args.post_hold_s > 0:
                    logger.info(f"Holding target pose for {args.post_hold_s:.1f}s")
                    time.sleep(args.post_hold_s)
                return

            dt = time.time() - loop_t
            if dt < period:
                time.sleep(period - dt)

        logger.warning("Timeout reached before fully converging to start pose.")

    finally:
        if args.skip_disconnect:
            logger.warning("Skipping robot.disconnect() by request (--skip-disconnect).")
            return
        try:
            robot.disconnect()
        except Exception as exc:
            logger.warning(f"Disconnect raised an exception: {exc}")
        logger.info("Robot disconnected.")


if __name__ == "__main__":
    main()
