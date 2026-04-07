"""
Standalone Real-time IMU Visualizer for E1 Robot
独立运行的 IMU 实时监控脚本 (不会干扰控制进程)
"""

import sys
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==================== 导入真机 SDK ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from base.Base import NanoSleep
from base.RobotBase import RobotBase
from base.ConfigE1 import Config  # 你的配置文件
# ==========================================================

def main():
    print("=" * 60)
    print("启动 E1 机器人 IMU 实时监控...")
    print("注意：本脚本仅读取数据，不发送控制指令，绝对安全。")
    print("=" * 60)

    # 1. 初始化真机 SDK
    robot = RobotBase(Config)

    # 2. Matplotlib 实时画图设置
    plt.ion()  # 开启交互模式 (Interactive Mode)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('E1 Robot Real-time IMU')

    # 使用 deque 维持固定窗口大小的数据，防止内存爆炸
    # 假设 50Hz 刷新率，250 个点刚好是过去 5 秒的历史曲线
    max_points = 250
    time_data = deque(maxlen=max_points)
    roll_data = deque(maxlen=max_points)
    pitch_data = deque(maxlen=max_points)
    yaw_data  = deque(maxlen=max_points)

    # 绘制三条线
    line_r, = ax.plot([], [], label='Roll (翻滚角)', color='red', linewidth=2)
    line_p, = ax.plot([], [], label='Pitch (俯仰角)', color='green', linewidth=2)
    line_y, = ax.plot([], [], label='Yaw (偏航角)', color='blue', linewidth=2)

    ax.set_title("Real-time IMU Euler Angles (Degrees)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (Degrees)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle='--', alpha=0.7)

    start_time = time.perf_counter()
    timer = NanoSleep(20)  # 限制刷新率为 50Hz (20ms)，节省 CPU 性能

    print("[INFO] 开始实时绘制... (关闭窗口或按 Ctrl+C 退出)")

    try:
        # 当图表窗口处于打开状态时，持续循环
        while plt.fignum_exists(fig.number):
            loop_start = time.perf_counter()

            # --- 核心：获取机器人状态 ---
            robot.get_robot_state()

            # 提取并处理 IMU 欧拉角
            base_euler = np.array(robot.legState.imu_euler)
            base_euler[base_euler > math.pi] -= 2 * math.pi  # 处理越界跳变

            # 为了人类直观可读，将弧度 (rad) 转换为角度 (deg)
            roll_deg  = math.degrees(base_euler[0])
            pitch_deg = math.degrees(base_euler[1])
            yaw_deg   = math.degrees(base_euler[2])

            current_time = time.perf_counter() - start_time

            # --- 追加新数据 ---
            time_data.append(current_time)
            roll_data.append(roll_deg)
            pitch_data.append(pitch_deg)
            yaw_data.append(yaw_deg)

            # --- 更新图表曲线 ---
            line_r.set_data(time_data, roll_data)
            line_p.set_data(time_data, pitch_data)
            line_y.set_data(time_data, yaw_data)

            # 动态调整 X 轴：始终显示最新的 5 秒
            ax.set_xlim(max(0, current_time - 5), max(5, current_time + 0.5))

            # 动态调整 Y 轴：根据当前窗口内的最大/最小值自适应缩放
            all_angles = list(roll_data) + list(pitch_data) + list(yaw_data)
            if all_angles:
                min_a, max_a = min(all_angles), max(all_angles)
                # 留出 10 度的上下边距，看着更舒服
                ax.set_ylim(min_a - 10, max_a + 10)

            # 极速刷新画布 (不使用会导致闪烁的 cla() )
            fig.canvas.draw()
            fig.canvas.flush_events()

            # 精确休眠
            timer.waiting(loop_start)

    except KeyboardInterrupt:
        print("\n[INFO] 捕捉到 Ctrl+C，安全退出监控...")
    finally:
        plt.ioff()
        plt.close('all')

if __name__ == '__main__':
    main()