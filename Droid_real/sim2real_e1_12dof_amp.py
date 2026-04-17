# 文件名: sim2real_e1_12dof_amp.py
print("[DEBUG] 1. 开始导入依赖库...")
import argparse
import os
import time
import sys
import numpy as np
import onnxruntime
import math

print("[DEBUG] 2. 导入手柄模块...")
from gamepad_controller import GamepadController

print("[DEBUG] 3. 导入真机 SDK...")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from base.Base import NanoSleep, euler_to_quaternion
from base.RobotBase import RobotBase
from base.ConfigE1 import Config

class SimToRealCfg:
    class sim:
        sim_duration = 10000.0  
        num_action = 12       
        num_obs_per_step = 51   
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

        kp = np.array([
            100, 100,  # hip_pitch
            100, 100,  # hip_roll
             50,  50,  # hip_yaw
            100, 100,  # knee
             20,  20,  # ankle_pitch
             20,  20   # ankle_roll
        ], dtype=np.float32)
        
        kd = np.array([5, 5, 5, 5, 3, 3, 5, 5, 2, 2, 2, 2], dtype=np.float32)

    class robot:
        gait_cycle: float = 0.8 
        gait_air_ratio_l: float = 0.37
        gait_air_ratio_r: float = 0.37
        gait_phase_offset_l: float = 0.37
        gait_phase_offset_r: float = 0.87

class RealRunner:
    def __init__(self, cfg: SimToRealCfg, policy_path):
        print("[DEBUG] 5. 初始化 RealRunner...")
        self.cfg = cfg
        
        print("[DEBUG] 6. 正在加载 ONNX 模型 (可能需要十几秒)...")
        self.session = onnxruntime.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name
        print("[DEBUG] 7. ONNX 模型加载成功！")

        self.joint_names = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint', 
            'left_hip_roll_joint', 'right_hip_roll_joint', 
            'left_hip_yaw_joint', 'right_hip_yaw_joint', 
            'left_knee_joint', 'right_knee_joint', 
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
            'left_ankle_roll_joint', 'right_ankle_roll_joint'
        ]
        
        self.real_joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]
        
        self.init_hardware_mapping()
        self.init_variables()
        
        print("[DEBUG] 8. 正在初始化手柄...")
        self.gamepad = GamepadController(deadzone=0.15)
        
        print("[DEBUG] 9. 正在初始化 RobotBase 真机底层 SDK...")
        self.robot = RobotBase(Config)
        self.robot.legActions = 13  
        self.robot.armActions = 8   
        
        while len(self.robot.legCommand.position) < self.robot.legActions:
            self.robot.legCommand.position.append(0.0)
            self.robot.legCommand.kp.append(0.0)
            self.robot.legCommand.kd.append(0.0)
        while len(self.robot.armCommand.position) < self.robot.armActions:
            self.robot.armCommand.position.append(0.0)
            self.robot.armCommand.kp.append(0.0)
            self.robot.armCommand.kd.append(0.0)
        print("[DEBUG] 10. SDK 初始化完成！")

    def init_hardware_mapping(self):
        self.idx_to_onnx = [self.real_joint_names.index(j) for j in self.joint_names]
        self.idx_to_real = [self.joint_names.index(j) for j in self.real_joint_names]

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action, dtype=np.float32)
        self.dof_vel = np.zeros(self.cfg.sim.num_action, dtype=np.float32)
        self.action = np.zeros(self.cfg.sim.num_action, dtype=np.float32)
        
        self.default_dof_pos = np.array([-0.3, -0.3, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, -0.2, -0.2, 0.0, 0.0], dtype=np.float32)
        self.default_dof_pos_real = self.default_dof_pos[self.idx_to_real]
        self.kp_real = self.cfg.sim.kp[self.idx_to_real]
        self.kd_real = self.cfg.sim.kd[self.idx_to_real]
        
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2, dtype=np.float32)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r], dtype=np.float32)
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r], dtype=np.float32)

        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.obs_history = np.zeros((self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        q_real = np.zeros(12, dtype=np.float32)
        dq_real = np.zeros(12, dtype=np.float32)
        for i in range(12):
            q_real[i] = self.robot.legState.position[i]
            dq_real[i] = self.robot.legState.velocity[i]

        self.dof_pos = q_real[self.idx_to_onnx]
        self.dof_vel = dq_real[self.idx_to_onnx]

        base_euler = np.array(self.robot.legState.imu_euler)
        base_euler[base_euler > math.pi] -= 2 * math.pi
        
        eq = euler_to_quaternion(base_euler[0], base_euler[1], base_euler[2])
        quat_w = np.array(eq, dtype=np.double)
        obs_gravity = self.quat_rotate_inverse(quat_w, np.array([0, 0, -1]))
        omega = np.array(self.robot.legState.imu_gyro, dtype=np.float32)

        current_obs = np.concatenate([
            omega, obs_gravity, self.command_vel,  
            (self.dof_pos - self.default_dof_pos), self.dof_vel, self.action,
            np.sin(2 * np.pi * self.gait_phase).reshape(2,),
            np.cos(2 * np.pi * self.gait_phase).reshape(2,), self.phase_ratio,
        ], axis=0).astype(np.float32)

        current_obs = np.clip(current_obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = current_obs.copy()
        return self.obs_history

    def run(self) -> None:
        print("\n" + "="*50)
        print("[INFO] 🚀 Sim2Real 12DOF AMP Started.")
        print("[INFO] 🟢 启动 AI: LT(按键8) + A(按键0)")
        print("[INFO] 🛑 紧急熔断: LT(按键8) + B(按键1)")
        print("="*50 + "\n")

        # ==================== 第一阶段：真机平滑预热到初始蹲姿 ====================
        print("[INFO] 正在平滑过渡至预备蹲姿 (耗时 2 秒)...")
        self.robot.get_robot_state()
        q0_real = [self.robot.legState.position[i] for i in range(12)]
        
        T_init = 2.0
        tt_init = 0.0
        timer_init = NanoSleep(2) # 2ms = 500Hz 控制频率
        while tt_init < T_init:
            start_time = time.perf_counter()
            self.robot.get_robot_state()
            st = min(tt_init / T_init, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            
            for i in range(12):
                self.robot.legCommand.position[i] = s0 * q0_real[i] + s1 * self.default_dof_pos_real[i]
                self.robot.legCommand.kp[i] = self.kp_real[i]
                self.robot.legCommand.kd[i] = self.kd_real[i]
            
            self.robot.legCommand.position[12] = 0.0
            self.robot.legCommand.kp[12], self.robot.legCommand.kd[12] = 100.0, 4.0
            for i in range(8):
                self.robot.armCommand.position[i] = 0.0
                self.robot.armCommand.kp[i], self.robot.armCommand.kd[i] = 30.0, 2.0
                
            self.robot.set_robot_command()
            tt_init += 0.002
            timer_init.waiting(start_time)
            
        print("[SUCCESS] ✅ 已到达预备蹲姿！")

        # ==================== 第二阶段：维持蹲姿，等待操作员确认 ====================
        print("[INFO] 💤 机器人已锁定在蹲姿。当前处于【待机锁定】状态。")
        print("[INFO] 👉 请确认周围安全，准备好后按下 [LT + A] 激活 AI 神经网络！")
        
        timer_wait = NanoSleep(5) # 5ms = 200Hz
        while True:
            start_wait = time.perf_counter()
            self.gamepad.get_commands()
            
            # 检测 LT + A 解锁
            if self.gamepad.get_button_a() and self.gamepad.get_button_lt():
                print("\n[SUCCESS] 🔓 接收到启动指令！AI 正式接管底盘控制！")
                break
            
            # 必须持续发送维持当前蹲姿的指令，防止电机掉电
            for i in range(12):
                self.robot.legCommand.position[i] = self.default_dof_pos_real[i]
                self.robot.legCommand.kp[i] = self.kp_real[i]
                self.robot.legCommand.kd[i] = self.kd_real[i]
                
            self.robot.legCommand.position[12] = 0.0
            self.robot.legCommand.kp[12], self.robot.legCommand.kd[12] = 100.0, 4.0
            for i in range(8):
                self.robot.armCommand.position[i] = 0.0
                self.robot.armCommand.kp[i], self.robot.armCommand.kd[i] = 30.0, 2.0
                
            self.robot.set_robot_command()
            timer_wait.waiting(start_wait)

        # ==================== 第三阶段：AI 神经网络主循环 ====================
        self.robot.get_robot_state()
        settled_q_real = np.zeros(12, dtype=np.float32)
        for i in range(12): settled_q_real[i] = self.robot.legState.position[i]
        
        settled_q_onnx = settled_q_real[self.idx_to_onnx]
        self.action = (settled_q_onnx - self.default_dof_pos) / self.cfg.sim.action_scale

        target_q_real = settled_q_real.copy()
        count_lowlevel = 0
        timer_main = NanoSleep(self.cfg.sim.dt * 1000) 
        
        for _ in range(self.cfg.sim.actor_obs_history_length): self.get_obs()

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
                    print("🛑 电机已锁定当前位置，缓慢卸力中...")
                    print("="*55 + "\n")
                    
                    self.robot.get_robot_state()
                    
                    for i in range(12):
                        self.robot.legCommand.position[i] = self.robot.legState.position[i]
                        self.robot.legCommand.kp[i] = 0.0
                        self.robot.legCommand.kd[i] = 4.0  
                        
                    for i in range(8):
                        self.robot.armCommand.kp[i] = 0.0
                        self.robot.armCommand.kd[i] = 1.0
                        
                    self.robot.set_robot_command()
                    time.sleep(0.1) 
                    break 

                self.command_vel = np.array([pad_x, pad_y, pad_yaw], dtype=np.float32)

                print(f"\r[🎮 手柄] 目标速度 -> X: {self.command_vel[0]:5.2f} | Y: {self.command_vel[1]:5.2f} | Yaw: {self.command_vel[2]:5.2f}        ", end="", flush=True)

                if count_lowlevel % self.cfg.sim.decimation == 0:
                    obs = self.get_obs()
                    onnx_input = {self.input_name: obs.reshape(1, -1)}
                    
                    raw_action = self.session.run(None, onnx_input)[0].flatten()[:12]
                    self.action[:] = np.clip(raw_action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

                    target_dof_pos_onnx = self.action * self.cfg.sim.action_scale + self.default_dof_pos
                    raw_target_q_real = target_dof_pos_onnx[self.idx_to_real]

                    FADE_IN_STEPS = 100.0
                    current_ctrl_step = count_lowlevel // self.cfg.sim.decimation
                    if current_ctrl_step < FADE_IN_STEPS:
                        alpha = current_ctrl_step / FADE_IN_STEPS
                        target_q_real = (1.0 - alpha) * settled_q_real + alpha * raw_target_q_real
                    else:
                        target_q_real = raw_target_q_real

                    if np.any(np.isnan(target_q_real)):
                        print("\n[CRITICAL ERROR] 传感器或网络输出 NaN！已紧急熔断。")
                        break

                    self.episode_length_buf += 1
                    self.calculate_gait_para()

                for i in range(12):
                    self.robot.legCommand.position[i] = target_q_real[i]
                    self.robot.legCommand.kp[i] = self.kp_real[i]
                    self.robot.legCommand.kd[i] = self.kd_real[i]
                
                self.robot.legCommand.position[12] = 0.0
                for i in range(8): self.robot.armCommand.position[i] = 0.0
                
                self.robot.set_robot_command()
                count_lowlevel += 1

                timer_main.waiting(step_start)
                
        except KeyboardInterrupt:
            print("\n[INFO] 捕捉到 Ctrl+C！安全退出流程启动...")
        finally:
            print("Robot process finished safely.")

    def quat_rotate_inverse(self, q, v):
        q_w = q[0]; q_vec = q[1:4]
        return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0

    def calculate_gait_para(self) -> None:
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

if __name__ == "__main__":
    print("[DEBUG] 4. 解析命令行参数...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    args = parser.parse_args()
    
    runner = RealRunner(cfg=SimToRealCfg(), policy_path=args.policy)
    runner.run()