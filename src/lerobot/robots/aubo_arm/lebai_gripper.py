"""TCP socket driver for Lebai (乐白) grippers used with AUBO setups."""

import logging
import socket
import threading
import time
from collections import OrderedDict
from enum import Enum
from typing import Tuple, Union

logger = logging.getLogger(__name__)


class LebaiGripper:
    """乐白机械臂夹爪控制类。
    
    通过 Socket 直接与乐白机械臂夹爪通信，使用字符串命令和变量名进行控制。
    支持激活、位置控制、速度控制、力控制等功能。
    """

    # ========== 可写变量（也可读取）==========
    ACT = (
        "ACT"  # 激活标志：1 表示已激活，可重置为 0 以清除故障状态
    )
    GTO = (
        "GTO"  # 执行移动命令：设置为 1 时，夹爪将根据 POS、FOR、SPE 参数执行移动
    )
    ATR = "ATR"  # 自动释放：紧急情况下的慢速移动
    ADR = (
        "ADR"  # 自动释放方向：1 表示打开方向，0 表示闭合方向
    )
    FOR = "FOR"  # 力控制：范围 0-255，数值越大夹持力越大
    SPE = "SPE"  # 速度控制：范围 0-255，数值越大移动速度越快
    POS = "POS"  # 位置控制：范围 0-255，0 表示完全打开，255 表示完全闭合
    # ========== 只读变量 ==========
    STA = "STA"  # 状态：0 = 已重置，1 = 激活中，3 = 已激活
    PRE = "PRE"  # 位置请求：上次命令的位置值（回显）
    OBJ = "OBJ"  # 物体检测：0 = 移动中，1 = 外部夹持，2 = 内部夹持，3 = 静止无物体
    FLT = "FLT"  # 故障代码：0 = 正常，非零值表示故障

    ENCODING = "UTF-8"  # 使用 UTF-8 编码

    class GripperStatus(Enum):
        """夹爪状态枚举。
        
        由夹爪硬件报告的状态值，整数值必须与夹爪发送的值匹配。
        """

        RESET = 0  # 已重置状态
        ACTIVATING = 1  # 激活中
        # UNUSED = 2  # 此值当前未被夹爪固件使用
        ACTIVE = 3  # 已激活，可以正常工作

    class ObjectStatus(Enum):
        """物体检测状态枚举。
        
        由夹爪硬件报告的物体检测状态，整数值必须与夹爪发送的值匹配。
        """

        MOVING = 0  # 正在移动中
        STOPPED_OUTER_OBJECT = 1  # 因检测到外部物体而停止（物体在手指外部）
        STOPPED_INNER_OBJECT = 2  # 因检测到内部物体而停止（物体在手指内部）
        AT_DEST = 3  # 已到达目标位置，静止状态

    def __init__(self):
        """构造函数，初始化夹爪控制对象。
        
        初始化 Socket 连接、线程锁以及位置、速度、力的范围参数。
        """
        self.socket = None  # Socket 连接对象，初始为 None
        self.command_lock = threading.Lock()  # 命令发送/接收的线程锁，确保命令原子性
        self._min_position = 0  # 最小位置值（完全闭合）
        self._max_position = 255  # 最大位置值（完全打开）
        self._min_speed = 0  # 最小速度值
        self._max_speed = 100  # 最大速度值
        self._min_force = 0  # 最小力值
        self._max_force = 100  # 最大力值

    def connect(self, hostname: str, port: int, socket_timeout: float = 10.0) -> None:
        """连接到指定地址的乐白机械臂夹爪。

        :param hostname: 夹爪的主机名或 IP 地址（通常是乐白机械臂 IP）
        :param port: 夹爪的端口号（乐白机械臂通常使用 5888 端口）
        :param socket_timeout: Socket 阻塞操作的超时时间（秒），默认 10.0 秒
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert self.socket is not None
        self.socket.connect((hostname, port))  # 建立 TCP 连接
        self.socket.settimeout(socket_timeout)  # 设置超时时间

    def disconnect(self) -> None:
        """关闭与夹爪的连接。"""
        if self.socket is not None:
            self.socket.close()  # 关闭 Socket 连接
            self.socket = None

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]):
        """通过 Socket 发送命令设置多个变量的值，并等待响应。

        :param var_dict: 要设置的变量字典，格式为 {变量名: 值}
        :return: 成功接收到确认返回 True，否则返回 False
        """
        assert self.socket is not None
        # 构造命令字符串，格式：SET VAR1 VALUE1 VAR2 VALUE2 ...
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += "\n"  # 换行符是命令结束的标志
        # 使用线程锁确保命令发送/接收的原子性
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))  # 发送命令
            data = self.socket.recv(1024)  # 接收响应
        return self._is_ack(data)  # 检查是否为确认响应

    def _set_var(self, variable: str, value: Union[int, float]):
        """通过 Socket 发送命令设置单个变量的值，并等待响应。

        :param variable: 要设置的变量名
        :param value: 要设置的变量值
        :return: 成功接收到确认返回 True，否则返回 False
        """
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str):
        """发送命令从夹爪读取指定变量的值，阻塞直到收到响应或超时。

        :param variable: 要读取的变量名
        :return: 变量的整数值
        """
        assert self.socket is not None
        # 使用线程锁确保命令发送/接收的原子性
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)

        # 期望响应格式为 'VAR x'，其中 VAR 是变量名的回显，X 是值
        try:
            decoded_data = data.decode(self.ENCODING).strip()
            var_name, value_str = decoded_data.split()
            if var_name != variable:
                raise ValueError(
                    f"意外响应 {data} ({decoded_data}): 与 '{variable}' 不匹配"
                )
            value = int(value_str)
            return value
        except Exception as e:
            raise ValueError(f"解析响应失败: {data}, 错误: {str(e)}")

    @staticmethod
    def _is_ack(data: bytes):
        """检查接收到的数据是否为确认响应。
        
        :param data: 接收到的字节数据
        :return: 如果是确认响应则返回 True，否则返回 False
        """
        return data.strip() == b"ack"

    def _reset(self):
        """重置夹爪。
        
        执行重置逻辑：
        1. 设置 ACT 和 ATR 为 0
        2. 等待夹爪确认已重置
        3. 等待 0.5 秒确保重置完成
        """
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        while not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0:
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
        time.sleep(0.5)

    def activate(self, auto_calibrate: bool = True):
        """激活乐白机械臂夹爪。
        
        重置夹爪的激活标志，然后将其设置回 1，清除之前的故障标志。
        
        :param auto_calibrate: 是否根据实际运动校准最小和最大位置
        """
        if not self.is_active():
            self._reset()
            while not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0:
                time.sleep(0.01)

            self._set_var(self.ACT, 1)
            time.sleep(1.0)
            while not self._get_var(self.ACT) == 1 or not self._get_var(self.STA) == 3:
                time.sleep(0.01)

        # 如果需要，自动校准位置范围
        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self):
        """检查夹爪是否已激活。
        
        :return: 如果夹爪已激活返回 True，否则返回 False
        """
        status = self._get_var(self.STA)
        return (
            LebaiGripper.GripperStatus(status) == LebaiGripper.GripperStatus.ACTIVE
        )

    def get_min_position(self) -> int:
        """获取夹爪能到达的最小位置（闭合打位置）。
        
        :return: 最小位置值
        """
        return self._min_position

    def get_max_position(self) -> int:
        """获取夹爪能到达的最大位置（开位置）。
        
        :return: 最大位置值
        """
        return self._max_position

    def get_open_position(self) -> int:
        """获取夹爪的打开位置（最大位置值）。
        
        :return: 打开位置值
        """
        return self.get_max_position()

    def get_closed_position(self) -> int:
        """获取夹爪的闭合位置（最小位置值）。
        
        :return: 闭合位置值
        """
        return self.get_min_position()

    def is_open(self):
        """检查当前是否处于完全打开状态。
        
        :return: 如果当前位置小于等于打开位置返回 True，否则返回 False
        """
        return self.get_current_position() >= self.get_open_position()

    def is_closed(self):
        """检查当前是否处于完全闭合状态。
        
        :return: 如果当前位置大于等于闭合位置返回 True，否则返回 False
        """
        return self.get_current_position() <= self.get_closed_position()

    def get_current_position(self) -> int:
        """获取硬件报告的当前位置。
        
        :return: 当前位置值
        """
        return self._get_var(self.POS)

    def auto_calibrate(self, log: bool = True) -> None:
        """自动校准打开和闭合位置。
        
        通过缓慢闭合和打开夹爪来校准实际位置范围。
        
        :param log: 是否打印校准结果到日志
        :raises RuntimeError: 如果校准过程中遇到物体或失败
        """
        # 首先尝试打开，以防夹持有物体
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if LebaiGripper.ObjectStatus(status) != LebaiGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"校准失败，打开时遇到物体: {str(status)}")

        # 尝试尽可能闭合，记录位置值
        (position, status) = self.move_and_wait_for_pos(
            self.get_closed_position(), 64, 1
        )
        if LebaiGripper.ObjectStatus(status) != LebaiGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(
                f"校准失败，因为遇到物体: {str(status)}"
            )
        assert position >= self._min_position
        self._min_position = position

        # 尝试尽可能打开，记录位置值
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if LebaiGripper.ObjectStatus(status) != LebaiGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(
                f"校准失败，因为遇到物体: {str(status)}"
            )
        assert position <= self._max_position
        self._max_position = position

        if log:
            logger.info(
                "Lebai gripper calibrated to range [%s, %s]",
                self.get_min_position(),
                self.get_max_position(),
            )

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """发送移动命令到指定位置，使用指定的速度和力。
        
        :param position: 目标位置 [min_position, max_position]
        :param speed: 移动速度 [min_speed, max_speed]
        :param force: 夹持力 [min_force, max_force]
        :return: 元组 (是否成功发送命令, 调整后的实际请求位置)
        """
        position = int(position)
        speed = int(speed)
        force = int(force)

        def clip_val(min_val, val, max_val):
            """将值限制在最小值和最大值之间。"""
            return max(min_val, min(val, max_val))

        # 将输入值限制在合理范围内
        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        clip_for = clip_val(self._min_force, force, self._max_force)

        # 设置位置、速度、力参数，并执行移动
        var_dict = OrderedDict(
            [
                (self.POS, clip_pos),
                (self.SPE, clip_spe),
                (self.FOR, clip_for),
                (self.GTO, 1),  # 设置为 1 执行移动
            ]
        )
        succ = self._set_vars(var_dict)
        time.sleep(0.008)  # 需要等待（可能是硬件处理延迟）
        return succ, clip_pos

    def move_and_wait_for_pos(
        self, position: int, speed: int, force: int
    ) -> Tuple[int, "LebaiGripper.ObjectStatus"]:
        """发送移动命令并等待移动完成。
        
        :param position: 目标位置 [min_position, max_position]
        :param speed: 移动速度 [min_speed, max_speed]
        :param force: 夹持力 [min_force, max_force]
        :return: 元组 (最终位置, 移动结束状态)
        :raises RuntimeError: 如果设置移动变量失败
        """
        position = int(position)
        speed = int(speed)
        force = int(force)

        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("设置移动变量失败。")

        # 等待夹爪确认将尝试移动到请求的位置
        while self._get_var(self.PRE) != cmd_pos:
            time.sleep(0.001)

        # 等待直到夹爪停止移动
        cur_obj = self._get_var(self.OBJ)
        while (
            LebaiGripper.ObjectStatus(cur_obj) == LebaiGripper.ObjectStatus.MOVING
        ):
            cur_obj = self._get_var(self.OBJ)

        # 返回实际位置和物体状态
        final_pos = self._get_var(self.POS)
        final_obj = cur_obj
        return final_pos, LebaiGripper.ObjectStatus(final_obj)

    def emergency_release(self, direction: int = 1):
        """执行紧急释放。
        
        :param direction: 释放方向，1=打开方向，0=闭合方向
        """
        self._set_var(self.ADR, direction)
        self._set_var(self.ATR, 1)
        time.sleep(1.0)  # 等待释放完成

    def get_fault_code(self):
        """获取故障代码。
        
        :return: 故障代码，0=正常，其他值表示故障
        """
        return self._get_var(self.FLT)


def main():
    """主函数，用于测试乐白机械臂夹爪控制功能。"""
    # 测试打开和闭合夹爪
    gripper = LebaiGripper()
    
    # 连接到乐白机械臂（默认端口 6888）
    try:
        gripper.connect(hostname="192.168.123.118", port=6888)
        print("连接成功")
        
        # 激活夹爪
        print("激活夹爪...")
        gripper.activate()
        
        # 获取当前位置
        current_pos = gripper.get_current_position()
        print(f"当前位置: {current_pos}")
        
        # 移动到半开位置
        print("移动到半开位置...")
        gripper.move(128, 128, 64)
        time.sleep(1.0)
        
        # 检查是否打开
        if gripper.is_open():
            print("夹爪已完全打开")
        elif gripper.is_closed():
            print("夹爪已完全闭合")
        else:
            print(f"夹爪处于中间位置: {gripper.get_current_position()}")
            
        # 完全打开夹爪
        time.sleep(1)
        print("完全打开夹爪...")
        gripper.move_and_wait_for_pos(gripper.get_open_position(), 255, 64)
        time.sleep(1)
        print("完全闭合夹爪")
        gripper.move_and_wait_for_pos(gripper.get_closed_position(), 255, 64)

        print(f"最终位置: {gripper.get_current_position()}")
        
        gripper.disconnect()
        print("断开连接")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        if gripper.socket is not None:
            gripper.disconnect()


if __name__ == "__main__":
    main()