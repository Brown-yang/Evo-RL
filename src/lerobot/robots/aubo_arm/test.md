RTC 推理：
python src/lerobot/scripts/aubo_executor.py --policy.path=/home/lab/model/checkpoints/010000/pretrained_model --policy.device=cuda --device=cuda --rtc.enabled=true --rtc.execution_horizon=20 --robot.type=aubo_arm --robot.robot_ip=192.168.123.213 --robot.rpc_port=30004 --robot.rtde_port=30010 --robot.username=aubo --robot.password=123456 --robot.use_gripper=true --robot.gripper_ip=192.168.123.118 --robot.gripper_port=6888 --robot.cameras="{ head: {type: zmq, server_address: 192.168.123.118, port: 5555, camera_name: head, width: 640, height: 480, fps: 30, warmup_s: 2, timeout_ms: 5000}, right_wrist: {type: zmq, server_address: 192.168.123.118, port: 5555, camera_name: right_wrist, width: 640, height: 480, fps: 30, warmup_s: 2, timeout_ms: 5000} }" --task="Pick up the watermelon and place it into the white box." --fps=10 --duration=900 --debug_print_obs_keys_once=true --debug_print_joint_obs=true --debug_print_joint_interval_s=1.0 --tokenizer_local_path=/home/lab/.cache/huggingface/hub/models--leo009--paligemma-3b-pt-224 --tokenizer_force_offline=true

## A/B 对照：为什么RTC执行更慢

### A 组：RTC 开启（提速参数）

```bash
python src/lerobot/scripts/aubo_executor.py \
  --policy.path=/home/lab/model/checkpoints/010000/pretrained_model \
  --policy.device=cuda \
  --device=cuda \
  --rtc.enabled=true \
  --rtc.execution_horizon=10 \
  --robot.type=aubo_arm \
  --robot.robot_ip=192.168.123.213 \
  --robot.rpc_port=30004 \
  --robot.rtde_port=30010 \
  --robot.username=aubo \
  --robot.password=123456 \
  --robot.use_gripper=true \
  --robot.gripper_ip=192.168.123.118 \
  --robot.gripper_port=6888 \
  --robot.max_jump=0.12 \
  --robot.vel_limit=120 \
  --robot.cameras="{ head: {type: zmq, server_address: 192.168.123.118, port: 5555, camera_name: head, width: 640, height: 480, fps: 30, warmup_s: 2, timeout_ms: 5000}, right_wrist: {type: zmq, server_address: 192.168.123.118, port: 5555, camera_name: right_wrist, width: 640, height: 480, fps: 30, warmup_s: 2, timeout_ms: 5000} }" \
  --task="Pick up the watermelon and place it into the white box." \
  --fps=20 \
  --duration=600 \
  --tokenizer_local_path=/home/lab/.cache/huggingface/hub/models--leo009--paligemma-3b-pt-224 \
  --tokenizer_force_offline=true
```

### B 组：RTC 关闭（对照执行链路）

```bash
python src/lerobot/scripts/aubo_executor.py \
  --policy.path=/home/lab/model/checkpoints/010000/pretrained_model \
  --policy.device=cuda \
  --device=cuda \
  --rtc.enabled=false \
  --rtc.execution_horizon=10 \
  --robot.type=aubo_arm \
  --robot.robot_ip=192.168.123.213 \
  --robot.rpc_port=30004 \
  --robot.rtde_port=30010 \
  --robot.username=aubo \
  --robot.password=123456 \
  --robot.use_gripper=true \
  --robot.gripper_ip=192.168.123.118 \
  --robot.gripper_port=6888 \
  --robot.max_jump=0.12 \
  --robot.vel_limit=120 \
  --robot.cameras="{ head: {type: zmq, server_address: 192.168.123.118, port: 5555, camera_name: head, width: 640, height: 480, fps: 30, warmup_s: 2, timeout_ms: 5000}, right_wrist: {type: zmq, server_address: 192.168.123.118, port: 5555, camera_name: right_wrist, width: 640, height: 480, fps: 30, warmup_s: 2, timeout_ms: 5000} }" \
  --task="Pick up the watermelon and place it into the white box." \
  --fps=20 \
  --duration=600 \
  --tokenizer_local_path=/home/lab/.cache/huggingface/hub/models--leo009--paligemma-3b-pt-224 \
  --tokenizer_force_offline=true
```




