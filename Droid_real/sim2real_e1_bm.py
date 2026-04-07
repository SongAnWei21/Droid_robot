import os
import sys
import math
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort
from pynput import keyboard

# ==================== 真机 SDK ============================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base.Base import NanoSleep, euler_to_quaternion
from base.RobotBase import RobotBase
from base.ConfigE1_bm import Config 
# ==========================================================

class Cmd:
    reset_requested = False

def on_press(key_evt):
    try:
        # 按 '0' 键作为紧急停止！
        if key_evt.char == '0':
            Cmd.reset_requested = True
    except AttributeError:
        pass

def on_release(key):
    pass

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener

def get_obs_real(robot):
    """从真实的 RobotBase SDK 中提取观测量，替代原来的 mujoco data提取"""
    q = np.zeros(21, dtype=np.double)
    dq = np.zeros(21, dtype=np.double)
    
    # 提取 13 个腿部/腰部关节
    for i in range(13):
        q[i] = robot.legState.position[i]
        dq[i] = robot.legState.velocity[i]
    # 提取 8 个手臂关节 (21 DOF 版本)
    for i in range(8):
        q[13+i] = robot.armState.position[i]
        dq[13+i] = robot.armState.velocity[i]

    # IMU 处理：欧拉角 -> 四元数 -> 旋转矩阵
    euler = np.array(robot.legState.imu_euler)
    # 处理可能的角度跳变
    euler[euler > math.pi] -= 2 * math.pi
    quat_wxyz = euler_to_quaternion(euler[0], euler[1], euler[2])
    # R.from_quat 需要 [x, y, z, w] 顺序
    quat = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    r = R.from_quat(quat)
    
    # 获取投影重力和角速度
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    omega = np.array(robot.legState.imu_gyro).astype(np.double)
    v = np.zeros(3, dtype=np.double) # 真机无可靠线速度，此处全0占位
    
    return q, dq, quat, v, omega, gvec


def run_real(policy, cfg, loop=False, motion_file=None):
    """
    真机运行逻辑：完全对齐原有的 run_mujoco，仅替换执行器
    """
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1

    print("=" * 60)
    print("Keyboard control instructions:")
    print("  0 key: EMERGENCY STOP (紧急停止)")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()

    # ── Load motion reference data ────────────────────────────────────────────
    motion      = np.load(motion_file)
    m_joint_pos = motion["joint_pos"]    # (T, 21) in Isaac Lab joint order
    m_joint_vel = motion["joint_vel"]    # (T, 21) in Isaac Lab joint order
    num_frames  = min(m_joint_pos.shape[0], m_joint_vel.shape[0])

    num_actions = cfg.robot_config.num_actions

    # ── Initialize Real Robot SDK ─────────────────────────────────────────────
    print("[INFO] Initializing Real Robot SDK...")
    robot = RobotBase(Config)
    robot.legActions = 13
    robot.armActions = 8
    
    # 补齐 command 数组防越界
    while len(robot.legCommand.position) < robot.legActions:
        robot.legCommand.position.append(0.0)
        robot.legCommand.kp.append(0.0)
        robot.legCommand.kd.append(0.0)
    while len(robot.armCommand.position) < robot.armActions:
        robot.armCommand.position.append(0.0)
        robot.armCommand.kp.append(0.0)
        robot.armCommand.kd.append(0.0)


    print("[INFO] Smoothly initializing to the first frame...")
    robot.get_robot_state()
    q0 = np.zeros(num_actions)
    for i in range(13): q0[i] = robot.legState.position[i]
    for i in range(8):  q0[13+i] = robot.armState.position[i]

    # 将第一帧从 Isaac 顺序转为 Real顺序
    first_frame_isaac = m_joint_pos[0]
    first_frame_real = np.zeros(num_actions)
    for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
        first_frame_real[mujoco_idx] = first_frame_isaac[isaac_idx]
    target_pos_abs_init = first_frame_real + cfg.robot_config.default_pos

    T_init = 2.0
    tt_init = 0.0
    timer_init = NanoSleep(2)  # 2ms 循环过渡
    while tt_init < T_init:
        start_time = time.perf_counter()
        robot.get_robot_state()
        st = min(tt_init / T_init, 1.0)
        s0 = 0.5 * (1.0 + math.cos(math.pi * st))
        s1 = 1 - s0

        for i in range(13):
            robot.legCommand.position[i] = s0 * q0[i] + s1 * target_pos_abs_init[i]
        for i in range(8):
            robot.armCommand.position[i] = s0 * q0[13+i] + s1 * target_pos_abs_init[13+i]
        
        robot.set_robot_command()
        tt_init += 0.002
        timer_init.waiting(start_time)
    print("[SUCCESS] Initialization complete. Starting policy loop...")

    # ── Per-term history buffers (H, dim), oldest → newest ───────────────────
    H = cfg.robot_config.frame_stack   # 10
    hist_cmd       = np.zeros((H, 2 * num_actions), dtype=np.float32)  # 42
    hist_proj_grav = np.zeros((H, 3),               dtype=np.float32)
    hist_ang_vel   = np.zeros((H, 3),               dtype=np.float32)
    hist_joint_pos = np.zeros((H, num_actions),     dtype=np.float32)
    hist_joint_vel = np.zeros((H, num_actions),     dtype=np.float32)
    hist_actions   = np.zeros((H, num_actions),     dtype=np.float32)
    is_first_frame = True

    count_lowlevel = 0
    motion_t       = 0
    
    target_pos  = np.zeros(num_actions, dtype=np.double)
    action      = np.zeros(num_actions, dtype=np.double)   # Isaac Lab order

    # 底层执行器设定：dt = 0.005 (200Hz)
    timer_main = NanoSleep(cfg.sim_config.dt * 1000)

    try:
        # 使用 while True 保证与真机的持续通信
        while True:
            start_time = time.perf_counter()

            if Cmd.reset_requested:
                print('EMERGENCY STOP (紧急停止)')
                break

            robot.get_robot_state()
            q, dq, quat, v, omega, gvec = get_obs_real(robot)

            # ── Policy step (50 Hz, count_lowlevel % 4 == 0) ─────────────────
            if count_lowlevel % cfg.sim_config.decimation == 0:
                idx = frame_idx(motion_t)

                # Real(MuJoCo) joint order → Isaac Lab joint order
                q_rel  = q - cfg.robot_config.default_pos
                q_obs  = np.zeros(num_actions, dtype=np.double)
                dq_obs = np.zeros(num_actions, dtype=np.double)
                for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                    q_obs[isaac_idx]  = q_rel[mujoco_idx]
                    dq_obs[isaac_idx] = dq[mujoco_idx]

                # Current-frame per-term observations
                cur_cmd       = np.concatenate([m_joint_pos[idx], m_joint_vel[idx]]).astype(np.float32)
                cur_proj_grav = gvec.astype(np.float32)
                cur_ang_vel   = omega.astype(np.float32)
                cur_joint_pos = q_obs.astype(np.float32)
                cur_joint_vel = dq_obs.astype(np.float32)
                cur_action    = action.astype(np.float32)

                # Sliding window update
                if is_first_frame:
                    hist_cmd[:]       = cur_cmd
                    hist_proj_grav[:] = cur_proj_grav
                    hist_ang_vel[:]   = cur_ang_vel
                    hist_joint_pos[:] = cur_joint_pos
                    hist_joint_vel[:] = cur_joint_vel
                    hist_actions[:]   = cur_action
                    is_first_frame = False
                else:
                    hist_cmd       = np.roll(hist_cmd,       -1, axis=0); hist_cmd[-1]       = cur_cmd
                    hist_proj_grav = np.roll(hist_proj_grav, -1, axis=0); hist_proj_grav[-1] = cur_proj_grav
                    hist_ang_vel   = np.roll(hist_ang_vel,   -1, axis=0); hist_ang_vel[-1]   = cur_ang_vel
                    hist_joint_pos = np.roll(hist_joint_pos, -1, axis=0); hist_joint_pos[-1] = cur_joint_pos
                    hist_joint_vel = np.roll(hist_joint_vel, -1, axis=0); hist_joint_vel[-1] = cur_joint_vel
                    hist_actions   = np.roll(hist_actions,   -1, axis=0); hist_actions[-1]   = cur_action

                # Term-wise flattened input
                policy_input = np.concatenate([
                    hist_cmd.reshape(-1),
                    hist_proj_grav.reshape(-1),
                    hist_ang_vel.reshape(-1),
                    hist_joint_pos.reshape(-1),
                    hist_joint_vel.reshape(-1),
                    hist_actions.reshape(-1),
                ]).reshape(1, -1).astype(np.float32)

                # ONNX Runtime inference
                input_name = policy.get_inputs()[0].name
                action[:] = policy.run(None, {input_name: policy_input})[0][0]

                # Isaac Lab order → Real(MuJoCo) order target positions
                target_q = action * cfg.robot_config.action_scale
                for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                    target_pos[mujoco_idx] = target_q[isaac_idx]
                target_pos_abs = target_pos + cfg.robot_config.default_pos

                motion_t += 1

            # ── 无论网络是否更新，底层 200Hz 持续下发指令 ──────────────────────────
            # 将 cfg.robot_config 的软 PD 参数应用到真机 SDK
            for i in range(13):
                robot.legCommand.position[i] = target_pos_abs[i]
                robot.legCommand.kp[i] = cfg.robot_config.kps[i]
                robot.legCommand.kd[i] = cfg.robot_config.kds[i]
            for i in range(8):
                robot.armCommand.position[i] = target_pos_abs[13+i]
                robot.armCommand.kp[i] = cfg.robot_config.kps[13+i]
                robot.armCommand.kd[i] = cfg.robot_config.kds[13+i]

            
            robot.set_robot_command()
            count_lowlevel += 1
            
            # 使用真机的 NanoSleep 精确控制 200Hz (dt=0.005) 的节拍
            timer_main.waiting(start_time)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected, shutting down...")

    keyboard_listener.stop()
    print("Robot process finished safely.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E1 BeyondMimic Sim2Real 21DOF deployment script.')
    parser.add_argument('--policy_model', type=str, default='data/policies/bm/policy.onnx',
                        help='Path to exported policy.onnx')
    parser.add_argument('--motion_file', type=str, default='data/motions/e1_bm/MJ_dance.npz',
                        help='Path to reference motion .npz' )
    parser.add_argument('--loop', action='store_true',
                        help='Loop the reference motion indefinitely')
    args = parser.parse_args()

    class Sim2RealCfg:
        class sim_config:
            dt           = 0.005   # 200 Hz physics
            decimation   = 4       # 50 Hz policy

        class robot_config:
            # ── PD gains (MuJoCo/Real actuator order) ─────────────────────────────
            kps = np.array([
                150, 150, 100, 150, 20, 20,   # L leg
                150, 150, 100, 150, 20, 20,   # R leg
                100,                           # waist_yaw
                40,  40,  40,  40,            # L arm
                40,  40,  40,  40,            # R arm
            ], dtype=np.double)
            kds = np.array([
                3,   3,   3,   5,   2,  2,    # L leg
                3,   3,   3,   5,   2,  2,    # R leg
                3,                             # waist_yaw
                2,   2,   2,   2,             # L arm
                2,   2,   2,   2,             # R arm
            ], dtype=np.double)

            # ── Default joint positions (MuJoCo/Real order) ───────────────────────
            default_pos = np.array([
                -0.1,  0.0,  0.0,  0.2,  -0.1,  0.0,   # L leg
                -0.1,  0.0,  0.0,  0.2,  -0.1,  0.0,   # R leg
                 0.0,                                    # waist_yaw
                 0.18, 0.06, 0.06, 0.78,                # L arm
                 0.18, 0.06, 0.06, 0.78,                # R arm
            ], dtype=np.double)

            # ── Isaac Lab BFS order → MuJoCo/Real DFS order ───────────────────────
            usd2urdf = [0, 6, 12, 1, 7, 13, 17, 2, 8, 14, 18, 3, 9, 15, 19, 4, 10, 16, 20, 5, 11]

            num_actions    = 21
            action_scale   = 0.25
            frame_stack    = 10    

    policy = ort.InferenceSession(args.policy_model)
    run_real(policy, Sim2RealCfg(), args.loop, args.motion_file)