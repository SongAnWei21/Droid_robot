"""
Standalone Real-time IMU & 21-DOF Joint Visualizer for E1 Robot
独立运行的传感器实时监控脚本 (IMU + 21关节位置 + 21关节速度)
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

# 21 自由度关节的简写名称 (用于图例展示)
JOINT_NAMES = [
    # 腿部 12
    'L_Hip_P', 'L_Hip_R', 'L_Hip_Y', 'L_Knee', 'L_Ankle_P', 'L_Ankle_R',
    'R_Hip_P', 'R_Hip_R', 'R_Hip_Y', 'R_Knee', 'R_Ankle_P', 'R_Ankle_R',
    # 腰部 1
    'Waist_Y', 
    # 手臂 8
    'L_Shoulder_P', 'L_Shoulder_R', 'L_Shoulder_Y', 'L_Elbow',
    'R_Shoulder_P', 'R_Shoulder_R', 'R_Shoulder_Y', 'R_Elbow'
]
NUM_JOINTS = len(JOINT_NAMES)

def main():
    print("=" * 60)
    print("启动 E1 机器人 21-DOF 全状态实时监控...")
    print("注意：本脚本仅读取数据，不发送控制指令，绝对安全。")
    print("=" * 60)

    # 1. 初始化真机 SDK
    try:
        robot = RobotBase(Config)
    except Exception as e:
        print(f"[ERROR] SDK 初始化失败，请检查底层服务是否启动: {e}")
        return

    # 2. Matplotlib 实时画图设置
    plt.ion()  # 开启交互模式
    # 创建 3 行 1 列的子图结构，稍微加大画布尺寸适应 21 个图例
    fig, (ax_imu, ax_pos, ax_vel) = plt.subplots(3, 1, figsize=(14, 10))
    fig.canvas.manager.set_window_title('E1 Robot 21-DOF Real-time Monitor')
    fig.tight_layout(pad=4.0, rect=[0, 0, 0.85, 1]) # 留出右侧空间给图例

    # 使用 deque 维持固定窗口大小的数据 (50Hz * 5秒 = 250个点)
    max_points = 250
    time_data = deque(maxlen=max_points)
    
    # IMU 数据队列
    roll_data = deque(maxlen=max_points)
    pitch_data = deque(maxlen=max_points)
    yaw_data  = deque(maxlen=max_points)
    
    # 关节数据队列列表 (包含 21 个 deque)
    pos_data = [deque(maxlen=max_points) for _ in range(NUM_JOINTS)]
    vel_data = [deque(maxlen=max_points) for _ in range(NUM_JOINTS)]

    # 生成 21 种不同的颜色 (tab20有20种，最后追加一个黑色)
    cmap = plt.get_cmap('tab20')
    line_colors = [cmap(i) for i in range(20)] + ['black']

    # --- 初始化 IMU 曲线 ---
    line_r, = ax_imu.plot([], [], label='Roll', color='red', linewidth=2)
    line_p, = ax_imu.plot([], [], label='Pitch', color='green', linewidth=2)
    line_y, = ax_imu.plot([], [], label='Yaw', color='blue', linewidth=2)
    ax_imu.set_title("IMU Euler Angles (Degrees)")
    ax_imu.set_ylabel("Angle (deg)")
    ax_imu.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax_imu.grid(True, linestyle='--', alpha=0.7)

    # --- 初始化 关节位置 (Position) 曲线 ---
    lines_pos = []
    for i in range(NUM_JOINTS):
        line, = ax_pos.plot([], [], label=JOINT_NAMES[i], color=line_colors[i], linewidth=1.5)
        lines_pos.append(line)
    ax_pos.set_title("Joint Positions (Radians)")
    ax_pos.set_ylabel("Position (rad)")
    # 分两列显示图例，防止超出屏幕
    ax_pos.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol=2)
    ax_pos.grid(True, linestyle='--', alpha=0.7)

    # --- 初始化 关节速度 (Velocity) 曲线 ---
    lines_vel = []
    for i in range(NUM_JOINTS):
        line, = ax_vel.plot([], [], label=JOINT_NAMES[i], color=line_colors[i], linewidth=1.5)
        lines_vel.append(line)
    ax_vel.set_title("Joint Velocities (Rad/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (rad/s)")
    ax_vel.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol=2)
    ax_vel.grid(True, linestyle='--', alpha=0.7)

    start_time = time.perf_counter()
    timer = NanoSleep(20)  # 限制刷新率为 50Hz (20ms)

    print("[INFO] 开始实时绘制... (关闭窗口或按 Ctrl+C 退出)")

    try:
        while plt.fignum_exists(fig.number):
            loop_start = time.perf_counter()

            # --- 核心：获取机器人状态 ---
            robot.get_robot_state()

            # 1. 提取 IMU 欧拉角并转换
            base_euler = np.array(robot.legState.imu_euler)
            base_euler[base_euler > math.pi] -= 2 * math.pi  
            roll_deg  = math.degrees(base_euler[0])
            pitch_deg = math.degrees(base_euler[1])
            yaw_deg   = math.degrees(base_euler[2])

            # 2. 提取 21 自由度的关节数据
            q_real = np.zeros(21, dtype=np.float32)
            dq_real = np.zeros(21, dtype=np.float32)
            
            # E1 SDK: 前 13 个自由度 (腿+腰) 在 legState
            for i in range(13):
                q_real[i] = robot.legState.position[i]
                dq_real[i] = robot.legState.velocity[i]
            # E1 SDK: 后 8 个自由度 (双臂) 在 armState
            for i in range(8):
                q_real[13 + i] = robot.armState.position[i]
                dq_real[13 + i] = robot.armState.velocity[i]

            current_time = time.perf_counter() - start_time

            # --- 追加新数据 ---
            time_data.append(current_time)
            
            roll_data.append(roll_deg)
            pitch_data.append(pitch_deg)
            yaw_data.append(yaw_deg)
            
            for i in range(NUM_JOINTS):
                pos_data[i].append(q_real[i])
                vel_data[i].append(dq_real[i])

            # --- 更新图表曲线 ---
            # 更新 IMU
            line_r.set_data(time_data, roll_data)
            line_p.set_data(time_data, pitch_data)
            line_y.set_data(time_data, yaw_data)
            
            # 更新 Joints
            for i in range(NUM_JOINTS):
                lines_pos[i].set_data(time_data, pos_data[i])
                lines_vel[i].set_data(time_data, vel_data[i])

            # --- 动态调整 X 轴 ---
            x_min = max(0, current_time - 5)
            x_max = max(5, current_time + 0.5)
            ax_imu.set_xlim(x_min, x_max)
            ax_pos.set_xlim(x_min, x_max)
            ax_vel.set_xlim(x_min, x_max)

            # --- 动态调整 Y 轴 (避免因为数值跳变导致图形消失) ---
            if len(time_data) > 1:
                # IMU Y 轴
                all_imus = list(roll_data) + list(pitch_data) + list(yaw_data)
                ax_imu.set_ylim(min(all_imus) - 10, max(all_imus) + 10)
                
                # Position Y 轴
                all_pos = [p for d in pos_data for p in d]
                ax_pos.set_ylim(min(all_pos) - 0.5, max(all_pos) + 0.5)
                
                # Velocity Y 轴
                all_vel = [v for d in vel_data for v in d]
                ax_vel.set_ylim(min(all_vel) - 1.0, max(all_vel) + 1.0)

            # 极速刷新画布
            fig.canvas.draw_idle()
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