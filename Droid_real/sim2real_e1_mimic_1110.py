"""Unified sim2real script for E1 robot (10-frame Term-wise History, 1110 obs, 硬编码无TimeStep版)."""

import sys
import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import onnx
import torch
from scipy.spatial.transform import Rotation as R

# ==================== 导入真机 SDK ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from base.Base import NanoSleep, euler_to_quaternion
from base.RobotBase import RobotBase
from base.ConfigE1 import Config  # 你的配置文件
# ==========================================================

# ================= 纯数学与空间变换工具 =================
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)

def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return quat_conjugate_np(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)

def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)

# ================= 真实硬件底层物理顺序 (21-DOF) =================
RealJointOrder = [
    # LegBase (13个)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 
    # ArmBase (8个)
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
]

def get_robot_state_arrays(robot):
    """从机器人的状态中提取干净的 numpy 数组"""
    q_real = np.zeros(21, dtype=np.float32)
    dq_real = np.zeros(21, dtype=np.float32)

    for i in range(13):
        q_real[i] = robot.legState.position[i]
        dq_real[i] = robot.legState.velocity[i]
    for i in range(8):
        q_real[13 + i] = robot.armState.position[i]
        dq_real[13 + i] = robot.armState.velocity[i]

    base_euler = np.array(robot.legState.imu_euler)
    base_euler[base_euler > math.pi] -= 2 * math.pi
    
    # 【真机护盾 1：IMU 硬件误差补偿】
    # 如果真机起步后往后倒，可以尝试改成 0.05 左右；现在默认设为 0。
    PITCH_OFFSET = 0.0  
    base_euler[1] += PITCH_OFFSET  
    
    eq = euler_to_quaternion(base_euler[0], base_euler[1], base_euler[2])
    quat_w_first = np.array(eq, dtype=np.double)  # 格式为 [w, x, y, z]
    
    omega = np.array(robot.legState.imu_gyro, dtype=np.float32)

    return q_real, dq_real, quat_w_first, omega


def run_real(policy_path, motion_file):
    """
    面向过程的真机运行主循环 (全自动，1110维，硬编码参数，无 TimeStep)
    """
    print("=" * 60)
    print("初始化 E1-21DOF BeyondMimic 真机部署 (1110 obs, 修正防摔版)...")
    print("安全提示：运行期间在终端按 [Ctrl + C] 可随时紧急停止！")
    print("=" * 60)
    
    num_actions = 21
    decimation = 4
    dt = 0.005  # 200Hz
    timestep = 0  # 绝对不跳帧，从 0 开始
    alpha = 1.0  # 低通滤波范围 0.0 ~ 1.0，越小动作越丝滑，但会有微小的延迟；越大动作越干脆。

    # ── 1. 载入动作数据 (NPZ) ──
    print(f"[INFO]: Loading motion file from {motion_file}")
    motion = np.load(motion_file)
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]
    num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0])

    # ── 2. 载入 ONNX 及 Metadata 解析 ──
    print(f"[INFO]: Parsing ONNX metadata from {policy_path}")
    model = onnx.load(policy_path)
    
    # 增加 stiffness_onnx 和 damping_onnx 用于接收解析数据
    joint_seq, default_pos_onnx, action_scale_onnx = None, None, None
    stiffness_onnx, damping_onnx = None, None

    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        elif prop.key == "default_joint_pos":
            default_pos_onnx = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)
        elif prop.key == "action_scale":
            action_scale_onnx = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)
        elif prop.key == "joint_stiffness":
            stiffness_onnx = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)
        elif prop.key == "joint_damping":
            damping_onnx = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)

    idx_to_onnx = [RealJointOrder.index(j) for j in joint_seq]
    idx_to_real = [joint_seq.index(j) for j in RealJointOrder]

    # 将解析出的 kp 和 kd 转换为真实硬件顺序
    stiffness_real = stiffness_onnx[idx_to_real]
    damping_real = damping_onnx[idx_to_real]

    policy = ort.InferenceSession(policy_path)
    has_time_step = 'time_step' in [inp.name for inp in policy.get_inputs()]

    # ── 3. 初始化真机 SDK ──
    print("[INFO]: Initializing Real Robot SDK...")
    robot = RobotBase(Config)
    robot.legActions = 13
    robot.armActions = 8
    
    while len(robot.legCommand.position) < robot.legActions:
        robot.legCommand.position.append(0.0)
        robot.legCommand.kp.append(0.0)
        robot.legCommand.kd.append(0.0)
    while len(robot.armCommand.position) < robot.armActions:
        robot.armCommand.position.append(0.0)
        robot.armCommand.kp.append(0.0)
        robot.armCommand.kd.append(0.0)

    # ── 4. 平滑初始化到第 0 帧 ──
    print("[INFO] Smoothly initializing to the 0-th frame...")
    robot.get_robot_state()
    q0_leg = [robot.legState.position[i] for i in range(13)]
    q0_arm = [robot.armState.position[i] for i in range(8)]

    # 提取第 0 帧作为硬件的起始目标姿态
    first_frame_onnx = motioninputpos[timestep, :]
    first_frame_real = first_frame_onnx[idx_to_real]

    T_init = 2.0
    tt_init = 0.0
    timer_init = NanoSleep(2)
    while tt_init < T_init:
        start_time = time.perf_counter()
        robot.get_robot_state()
        st = min(tt_init / T_init, 1.0)
        s0 = 0.5 * (1.0 + math.cos(math.pi * st))
        s1 = 1 - s0
        
        for i in range(13):
            robot.legCommand.position[i] = s0 * q0_leg[i] + s1 * first_frame_real[i]
            robot.legCommand.kp[i] = stiffness_real[i]
            robot.legCommand.kd[i] = damping_real[i]
        for i in range(8):
            robot.armCommand.position[i] = s0 * q0_arm[i] + s1 * first_frame_real[13 + i]
            robot.armCommand.kp[i] = stiffness_real[13 + i]
            robot.armCommand.kd[i] = damping_real[13 + i]
            
        robot.set_robot_command()
        tt_init += 0.002
        timer_init.waiting(start_time)

    # ── 5. 倒计时准备 ──
    # wait_seconds = 1.0
    # print(f"\n[SUCCESS] 就位完成！倒计时 {int(wait_seconds)} 秒后 AI 接管控制权...")
    # wait_time = 0.0
    # timer_wait = NanoSleep(5) # 5ms
    
    # while wait_time < wait_seconds:
    #     start_time = time.perf_counter()
    #     robot.get_robot_state()
        
    #     # 保持死锁在第 0 帧姿态
    #     for i in range(13): 
    #         robot.legCommand.position[i] = first_frame_real[i]
    #         robot.legCommand.kp[i] = stiffness_real[i]
    #         robot.legCommand.kd[i] = damping_real[i]
    #     for i in range(8):  
    #         robot.armCommand.position[i] = first_frame_real[13 + i]
    #         robot.armCommand.kp[i] = stiffness_real[13 + i]
    #         robot.armCommand.kd[i] = damping_real[13 + i]
    #     robot.set_robot_command()
            
    #     wait_time += 0.005
    #     timer_wait.waiting(start_time)

    # ==================== 真机护盾 2：获取真实下垂姿态并“欺骗”AI ====================
    # 机器人落地后，电机在重力压迫下会存在一点点下垂误差（Gravity Sag）
    robot.get_robot_state()
    settled_q_real = np.zeros(21, dtype=np.float32)
    for i in range(13): settled_q_real[i] = robot.legState.position[i]
    for i in range(8):  settled_q_real[13 + i] = robot.armState.position[i]
    
    settled_q_onnx = settled_q_real[idx_to_onnx]

    # 反推初始 action_buffer，让 AI 以为自己上一帧发出的就是维持当前物理状态的指令！
    # 这彻底清除了第一帧神经网络惊跳后仰的问题。
    action_buffer = (settled_q_onnx - default_pos_onnx) / action_scale_onnx
    action_buffer = action_buffer.astype(np.float32)
    # ===============================================================================

    # ── 6. 初始化 10 帧历史堆叠 (1110维) ──
    H = 10
    hist_cmd       = np.zeros((H, 42), dtype=np.float32)
    hist_proj_grav = np.zeros((H, 3),  dtype=np.float32)
    hist_ang_vel   = np.zeros((H, 3),  dtype=np.float32)
    hist_joint_pos = np.zeros((H, 21), dtype=np.float32)
    hist_joint_vel = np.zeros((H, 21), dtype=np.float32)
    hist_actions   = np.zeros((H, 21), dtype=np.float32)
    is_first_frame = True

    # ── 7. 核心循环 ──
    print("\n[INFO] AI 动作跟随已启动！(按 Ctrl+C 随时急停)")
    # 注意：目标姿态从真实的物理下垂姿态开始起步
    target_q_real = settled_q_real.copy() 
    count_lowlevel = 0
    pre_tic = 0
    
    timer_main = NanoSleep(dt * 1000) # 5ms
    start = time.perf_counter()
    pbar = tqdm(desc="Tracking...")

    try:
        while True:
            start_time = time.perf_counter()
            robot.get_robot_state()
                
            # --- 50Hz 策略计算层 ---
            if count_lowlevel % decimation == 0:
                q_real, dq_real, quat_w_first, omega = get_robot_state_arrays(robot)
                
                # 转为网络认知顺序
                q_onnx = q_real[idx_to_onnx]
                dq_onnx = dq_real[idx_to_onnx]

                # 计算 3D 重力投影 (从 IMU 四元数计算)
                quat_scipy = np.array([quat_w_first[1], quat_w_first[2], quat_w_first[3], quat_w_first[0]])
                r_scipy = R.from_quat(quat_scipy)
                gvec = r_scipy.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float32)

                # 收集当前帧各特征项
                cur_cmd        = np.concatenate((motioninputpos[timestep, :], motioninputvel[timestep, :]), axis=0).astype(np.float32)
                cur_proj_grav  = gvec
                cur_ang_vel    = omega
                cur_joint_pos  = (q_onnx - default_pos_onnx)
                cur_joint_vel  = dq_onnx
                cur_action     = action_buffer

                # 历史滑动窗口更新
                if is_first_frame:
                    hist_cmd[:]        = cur_cmd
                    hist_proj_grav[:]  = cur_proj_grav
                    hist_ang_vel[:]    = cur_ang_vel
                    hist_joint_pos[:]  = cur_joint_pos
                    hist_joint_vel[:]  = cur_joint_vel
                    hist_actions[:]    = cur_action
                    is_first_frame = False
                else:
                    hist_cmd       = np.roll(hist_cmd,       -1, axis=0); hist_cmd[-1]       = cur_cmd
                    hist_proj_grav = np.roll(hist_proj_grav, -1, axis=0); hist_proj_grav[-1] = cur_proj_grav
                    hist_ang_vel   = np.roll(hist_ang_vel,   -1, axis=0); hist_ang_vel[-1]   = cur_ang_vel
                    hist_joint_pos = np.roll(hist_joint_pos, -1, axis=0); hist_joint_pos[-1] = cur_joint_pos
                    hist_joint_vel = np.roll(hist_joint_vel, -1, axis=0); hist_joint_vel[-1] = cur_joint_vel
                    hist_actions   = np.roll(hist_actions,   -1, axis=0); hist_actions[-1]   = cur_action

                # 特征平铺展平 (1110维: 420 + 30 + 30 + 210 + 210 + 210)
                obs = np.concatenate([
                    hist_cmd.reshape(-1),
                    hist_proj_grav.reshape(-1),
                    hist_ang_vel.reshape(-1),
                    hist_joint_pos.reshape(-1),
                    hist_joint_vel.reshape(-1),
                    hist_actions.reshape(-1)
                ]).reshape(1, -1).astype(np.float32)

                # 神经网络推理
                feed_dict = {'obs': obs}
                if has_time_step:
                    feed_dict['time_step'] = np.array([[timestep]], dtype=np.float32)
                
                action_onnx = policy.run(['actions'], feed_dict)[0][0]
                action_onnx = np.asarray(action_onnx).reshape(-1)
                raw_action = action_onnx.copy()

                # 应用动作平滑滤波
                action_buffer = alpha * raw_action + (1.0 - alpha) * action_buffer
                
                # 转换回真实硬件顺序
                target_pos_onnx = action_onnx * action_scale_onnx + default_pos_onnx
                raw_target_q_real = target_pos_onnx[idx_to_real]

                # 【真机护盾 3：AI 软启动离合 (Fade-in)】
                # 在 AI 接管的前 100 个控制帧（2 秒）内，让目标关节从当前被压弯的僵直态缓慢向 AI 目标态靠近
                FADE_IN_STEPS = 100.0
                current_ctrl_step = count_lowlevel // decimation
                if current_ctrl_step < FADE_IN_STEPS:
                    alpha = current_ctrl_step / FADE_IN_STEPS
                    target_q_real = (1.0 - alpha) * settled_q_real + alpha * raw_target_q_real
                else:
                    target_q_real = raw_target_q_real

                # 【真机护盾 4：NaN 紧急熔断】
                # 一旦出现 NaN 必须立刻掐断指令下发，保护电机！
                if np.any(np.isnan(target_q_real)):
                    print("\n[CRITICAL ERROR] 传感器或网络输出 NaN！已紧急熔断，保护真机安全。")
                    break

                timestep = (timestep + 1) % num_frames # 一轮结束重置timestep

            # --- 200Hz 物理执行层 ---
            for i in range(13):
                robot.legCommand.position[i] = target_q_real[i]
                robot.legCommand.kp[i] = stiffness_real[i]
                robot.legCommand.kd[i] = damping_real[i]
                
            for i in range(8):
                robot.armCommand.position[i] = target_q_real[13 + i]
                robot.armCommand.kp[i] = stiffness_real[13 + i]
                robot.armCommand.kd[i] = damping_real[13 + i]
                
            robot.set_robot_command()
            count_lowlevel += 1
            
            pbar.set_postfix(
                Cycle=f"{robot.legState.system_tic - pre_tic}ms", 
                Time=f"{(time.perf_counter() - start):.2f}s",
                Frame=timestep
            )
            pre_tic = robot.legState.system_tic
            timer_main.waiting(start_time)
            
    except KeyboardInterrupt:
        print("\n[INFO] 捕捉到 Ctrl+C (KeyboardInterrupt)！安全退出流程启动...")

    print("Robot process finished safely.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default="/home/saw/droidup/E1_BeyondMimic/logs/rsl_rl/e1_flat/2026-03-19_19-39-25/exported/model_49999.onnx")
    parser.add_argument("--motion_file", type=str, default="motion/MJ_dance.npz")
    args = parser.parse_args()

    run_real(policy_path=args.policy_path, motion_file=args.motion_file)