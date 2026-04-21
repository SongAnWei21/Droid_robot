# 文件名: sim2real_e1_21dof_amp.py

import argparse
import os
import time
import sys
import numpy as np
import onnxruntime
import math
from scipy.spatial.transform import Rotation as R

# ==================== 导入硬件 SDK 与手柄 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("[INFO] 正在导入手柄模块...")
from gamepad_controller import GamepadController

print("[INFO] 正在导入真机 SDK...")
from base.Base import NanoSleep, euler_to_quaternion
from base.RobotBase import RobotBase
from base.ConfigE1 import Config  # 请确保你的配置路径正确

# ====================================================================
# 🌟 绝对真理：ONNX TRUE JOINT ORDER (21 DOF) - 训练时的顺序
# ====================================================================
ONNX_JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 
    'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 
    'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 
    'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 
    'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint'
]

# ====================================================================
# 🌟 硬件底层物理顺序 (21 DOF) - 实际发送给电机的顺序
# ====================================================================
REAL_JOINT_NAMES = [
    # LegBase (13个: 12个腿 + 1个腰)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 
    # ArmBase (8个)
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
]

class SimToRealCfg:
    class sim:
        dt = 0.005        # 控制底层循环 200Hz
        decimation = 4    # 策略网络 50Hz (0.005 * 4)
        
        num_action = 21
        num_single_obs = 72
        frame_stack = 1
        
        action_scale = 0.25
        clip_observations = 100.0
        clip_actions = 100.0

        # 训练时的 ONNX 顺序参数
        kp_onnx = np.array([100, 100, 100, 100, 100, 30, 30, 50, 50, 30, 30, 100, 100, 30, 30, 20, 20, 30, 30, 20, 20], dtype=np.float32)
        kd_onnx = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.5, 2.5, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        default_pos_onnx = np.array([-0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, -0.25, 0.3, 0.3, 0.0, 0.0, -0.1, -0.1, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)


class RealRunner:
    def __init__(self, cfg: SimToRealCfg, policy_path: str):
        self.cfg = cfg
        
        print("[INFO] 正在加载 ONNX 模型...")
        self.session = onnxruntime.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name
        print("[SUCCESS] ONNX 模型加载成功！")

        self.init_hardware_mapping()
        self.init_variables()
        
        print("[INFO] 正在初始化手柄...")
        self.gamepad = GamepadController(deadzone=0.15)
        
        print("[INFO] 正在建立 gRPC 硬件连接...")
        self.robot = RobotBase(Config)
        self.robot.legActions = 13  
        self.robot.armActions = 8   
        
        # 预填充指令数组
        while len(self.robot.legCommand.position) < self.robot.legActions:
            self.robot.legCommand.position.append(0.0)
            self.robot.legCommand.kp.append(0.0)
            self.robot.legCommand.kd.append(0.0)
        while len(self.robot.armCommand.position) < self.robot.armActions:
            self.robot.armCommand.position.append(0.0)
            self.robot.armCommand.kp.append(0.0)
            self.robot.armCommand.kd.append(0.0)
            
        print("[SUCCESS] 底层通讯初始化完成！")

    def init_hardware_mapping(self):
        """建立网络输出和实际物理电机的双向映射"""
        self.idx_to_onnx = [REAL_JOINT_NAMES.index(j) for j in ONNX_JOINT_NAMES]
        self.idx_to_real = [ONNX_JOINT_NAMES.index(j) for j in REAL_JOINT_NAMES]

    def init_variables(self) -> None:
        self.action = np.zeros(self.cfg.sim.num_action, dtype=np.float32)
        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.obs_history = np.zeros((self.cfg.sim.frame_stack, self.cfg.sim.num_single_obs), dtype=np.float32)
        
        # 转换配置参数到真机顺序
        self.default_pos_real = self.cfg.sim.default_pos_onnx[self.idx_to_real]
        self.kp_real = self.cfg.sim.kp_onnx[self.idx_to_real]
        self.kd_real = self.cfg.sim.kd_onnx[self.idx_to_real]

    def get_obs(self) -> np.ndarray:
        """从传感器读取数据，拼接成 72 维 AMP 观测向量"""
        # 1. 抓取底层状态
        q_real = np.zeros(21, dtype=np.float32)
        dq_real = np.zeros(21, dtype=np.float32)
        for i in range(13):
            q_real[i] = self.robot.legState.position[i]
            dq_real[i] = self.robot.legState.velocity[i]
        for i in range(8):
            q_real[13 + i] = self.robot.armState.position[i]
            dq_real[13 + i] = self.robot.armState.velocity[i]

        # 2. 转换为网络认知顺序
        q_onnx = q_real[self.idx_to_onnx]
        dq_onnx = dq_real[self.idx_to_onnx]

        # 3. IMU 处理
        base_euler = np.array(self.robot.legState.imu_euler)
        base_euler[base_euler > math.pi] -= 2 * math.pi
        eq = euler_to_quaternion(base_euler[0], base_euler[1], base_euler[2])
        quat_w = np.array(eq, dtype=np.double)
        r_scipy = R.from_quat([quat_w[1], quat_w[2], quat_w[3], quat_w[0]])
        gvec = r_scipy.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float32)
        omega = np.array(self.robot.legState.imu_gyro, dtype=np.float32)

        # 4. 拼接 72 维特征
        obs = np.zeros(self.cfg.sim.num_single_obs, dtype=np.float32)
        obs[0:3] = omega
        obs[3:6] = gvec
        obs[6:9] = self.command_vel
        obs[9:30] = q_onnx - self.cfg.sim.default_pos_onnx
        obs[30:51] = dq_onnx
        obs[51:72] = self.action

        obs = np.clip(obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)
        
        # 更新滑动窗口
        self.obs_history = np.roll(self.obs_history, shift=-1, axis=0)
        self.obs_history[-1] = obs
        
        return self.obs_history.flatten().reshape(1, -1)

    def run(self) -> None:
        print("\n" + "="*60)
        print("🚀 E1-21DOF Sim2Real 启动！(AMP 模型)")
        print("🟢 激活 AI 接管:  LT(左扳机) + A(按键0)")
        print("🛑 紧急脱力熔断:  LT(左扳机) + B(按键1)")
        print("="*60 + "\n")

        # ==================== 第一阶段：真机平滑预热到初始站姿 ====================
        print("[INFO] 正在平滑过渡至预备站姿 (耗时 2 秒)...")
        self.robot.get_robot_state()
        
        q0_leg = [self.robot.legState.position[i] for i in range(13)]
        q0_arm = [self.robot.armState.position[i] for i in range(8)]
        
        T_init = 2.0
        tt_init = 0.0
        timer_init = NanoSleep(2) # 2ms
        
        while tt_init < T_init:
            start_time = time.perf_counter()
            self.robot.get_robot_state()
            
            st = min(tt_init / T_init, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            
            for i in range(13):
                self.robot.legCommand.position[i] = s0 * q0_leg[i] + s1 * self.default_pos_real[i]
                self.robot.legCommand.kp[i] = self.kp_real[i]
                self.robot.legCommand.kd[i] = self.kd_real[i]
            for i in range(8):
                self.robot.armCommand.position[i] = s0 * q0_arm[i] + s1 * self.default_pos_real[13 + i]
                self.robot.armCommand.kp[i] = self.kp_real[13 + i]
                self.robot.armCommand.kd[i] = self.kd_real[13 + i]
                
            self.robot.set_robot_command()
            tt_init += 0.002
            timer_init.waiting(start_time)
            
        print("[SUCCESS] ✅ 已安全到达预备姿态！")

        # ==================== 第二阶段：维持姿态，等待操作员确认 ====================
        print("[INFO] 💤 机器人已锁定。当前处于【待机状态】。")
        print("[INFO] 👉 请确认周围安全，准备好后按下 [LT + A] 激活 AI 神经网络！")
        
        timer_wait = NanoSleep(5) # 5ms
        while True:
            start_wait = time.perf_counter()
            self.gamepad.get_commands()
            
            if self.gamepad.get_button_a() and self.gamepad.get_button_lt():
                print("\n[SUCCESS] 🔓 接收到启动指令！AI 正式接管控制权！")
                break
            
            # 持续发指令防掉电
            for i in range(13):
                self.robot.legCommand.position[i] = self.default_pos_real[i]
            for i in range(8):
                self.robot.armCommand.position[i] = self.default_pos_real[13 + i]
            self.robot.set_robot_command()
            timer_wait.waiting(start_wait)

        # ==================== 第三阶段：AI 神经网络主循环 ====================
        # 获取机器人真实压迫下的姿态，防止启动第一帧惊跳
        self.robot.get_robot_state()
        settled_q_real = np.zeros(21, dtype=np.float32)
        for i in range(13): settled_q_real[i] = self.robot.legState.position[i]
        for i in range(8):  settled_q_real[13+i] = self.robot.armState.position[i]
        
        settled_q_onnx = settled_q_real[self.idx_to_onnx]
        self.action = (settled_q_onnx - self.cfg.sim.default_pos_onnx) / self.cfg.sim.action_scale

        target_q_real = settled_q_real.copy()
        count_lowlevel = 0
        timer_main = NanoSleep(self.cfg.sim.dt * 1000) 
        
        # 填充初始缓冲区
        for _ in range(self.cfg.sim.frame_stack): self.get_obs()

        try:
            while True:
                step_start = time.perf_counter()
                self.robot.get_robot_state()

                pad_x, pad_y, pad_yaw = self.gamepad.get_commands()
                btn_b = self.gamepad.get_button_b()   
                btn_lt = self.gamepad.get_button_lt() 

                # 🔴 终极紧急熔断：LT + B
                if btn_b and btn_lt:
                    print("\n\n" + "="*55)
                    print("🚨 [EMERGENCY STOP] 🚨")
                    print("🛑 检测到 LT + B！触发【原地阻尼急停】！")
                    print("🛑 正在切断电机刚度...")
                    print("="*55 + "\n")
                    
                    self.robot.get_robot_state()
                    for i in range(13):
                        self.robot.legCommand.position[i] = self.robot.legState.position[i]
                        self.robot.legCommand.kp[i] = 0.0
                        self.robot.legCommand.kd[i] = 4.0  
                    for i in range(8):
                        self.robot.armCommand.position[i] = self.robot.armState.position[i]
                        self.robot.armCommand.kp[i] = 0.0
                        self.robot.armCommand.kd[i] = 2.0
                        
                    self.robot.set_robot_command()
                    time.sleep(0.1) 
                    break 

                self.command_vel = np.array([pad_x, pad_y, pad_yaw], dtype=np.float32)
                print(f"\r[🎮] 速度 -> X: {self.command_vel[0]:5.2f} | Y: {self.command_vel[1]:5.2f} | Yaw: {self.command_vel[2]:5.2f}        ", end="", flush=True)

                # --- 50Hz 策略网络层 ---
                if count_lowlevel % self.cfg.sim.decimation == 0:
                    onnx_input = {self.input_name: self.get_obs()}
                    
                    # 推理
                    raw_action = self.session.run(None, onnx_input)[0].flatten()
                    self.action[:] = np.clip(raw_action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

                    target_dof_pos_onnx = self.action * self.cfg.sim.action_scale + self.cfg.sim.default_pos_onnx
                    raw_target_q_real = target_dof_pos_onnx[self.idx_to_real]

                    # 刚启动时的缓入融合 (Fade-in)，防止动作剧烈跳变
                    FADE_IN_STEPS = 100.0
                    current_ctrl_step = count_lowlevel // self.cfg.sim.decimation
                    if current_ctrl_step < FADE_IN_STEPS:
                        alpha = current_ctrl_step / FADE_IN_STEPS
                        target_q_real = (1.0 - alpha) * settled_q_real + alpha * raw_target_q_real
                    else:
                        target_q_real = raw_target_q_real

                    if np.any(np.isnan(target_q_real)):
                        print("\n[CRITICAL ERROR] 网络输出 NaN！已紧急熔断。")
                        break

                # --- 200Hz 底层下发层 ---
                for i in range(13):
                    self.robot.legCommand.position[i] = target_q_real[i]
                    self.robot.legCommand.kp[i] = self.kp_real[i]
                    self.robot.legCommand.kd[i] = self.kd_real[i]
                for i in range(8):
                    self.robot.armCommand.position[i] = target_q_real[13 + i]
                    self.robot.armCommand.kp[i] = self.kp_real[13 + i]
                    self.robot.armCommand.kd[i] = self.kd_real[13 + i]
                    
                self.robot.set_robot_command()
                count_lowlevel += 1

                timer_main.waiting(step_start)
                
        except KeyboardInterrupt:
            print("\n[INFO] 捕捉到 Ctrl+C！安全退出流程启动...")
        finally:
            print("Robot process finished safely.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认指向你之前跑通的 ONNX 模型路径
    parser.add_argument("--policy", type=str, default="/home/saw/droidup/atom01_train/logs/rsl_rl/e1_21dof_amp/2026-04-16_20-22-02/exported/policy.onnx")
    args = parser.parse_args()
    
    runner = RealRunner(cfg=SimToRealCfg(), policy_path=args.policy)
    runner.run()