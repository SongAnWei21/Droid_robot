import sys
import os
import math
import time
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

# ==================== 真机 SDK ====================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base.Base import NanoSleep, euler_to_quaternion
from base.LegBase import LegBase  
from tools.Gamepad import GamepadHandler  

# ============================== 配置路径 ==============================
onnx_model_path = "/home/saw/RL/humanoid_robot/Droid/Droid_robot/Droid_real/data/policies/amp/policy.onnx"

# ============================== 12-DOF映射关系 ==============================
# Isaac Lab AMP 训练环境里的关节顺序 
IsaacLabJointOrder = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint',
    'left_hip_roll_joint',  'right_hip_roll_joint',
    'left_hip_yaw_joint',   'right_hip_yaw_joint',
    'left_knee_joint',      'right_knee_joint',
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_ankle_roll_joint',  'right_ankle_roll_joint'
]

# 真实硬件底层物理顺序 LegBase 顺序
RealJointOrder = [
    # 左腿 (6个)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    # 右腿 (6个)
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
]

Isaac_to_Real_indices = [IsaacLabJointOrder.index(joint) for joint in RealJointOrder]
Real_to_Isaac_indices = [RealJointOrder.index(joint) for joint in IsaacLabJointOrder]


class Sim2Real_AMP(LegBase): # 直接继承 LegBase，抛弃手臂
    def __init__(self):
        super().__init__()   

        self.num_actions = 12
        self.history_length = 10
        self.decimation = 4
        self.dt = 0.005  # 底层 200Hz

        # 默认姿态
        self.default_joints_real = np.array([
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # 左腿
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # 右腿
        ], dtype=np.double)
        self.default_joints_isaac = self.default_joints_real[Real_to_Isaac_indices]
        self.action_scale_isaac = 0.25
        
        # 初始化 10 帧历史缓冲队列
        H = self.history_length
        self.hist_ang_vel   = np.zeros((H, 3), dtype=np.float32)               # 角速度
        self.hist_proj_grav = np.zeros((H, 3), dtype=np.float32)               # 投影重力
        self.hist_vel_cmd   = np.zeros((H, 3), dtype=np.float32)               # 速度指令 [vx, vy, dyaw]
        self.hist_joint_pos = np.zeros((H, self.num_actions), dtype=np.float32)# 关节相对位置
        self.hist_joint_vel = np.zeros((H, self.num_actions), dtype=np.float32)# 关节速度
        self.hist_actions   = np.zeros((H, self.num_actions), dtype=np.float32)# 上一步动作
        self.is_first_frame = True
        
        self.last_action_isaac = np.zeros(self.num_actions, dtype=np.double)
        self.target_q_real = self.default_joints_real.copy()
        
        # 加载 ONNX 模型与手柄
        print(f"[INFO]: Loading AMP ONNX policy from {onnx_model_path}")
        self.onnx_policy = ort.InferenceSession(onnx_model_path)
        self.rc = GamepadHandler()

    def get_velocity_command(self):
        """
        将手柄摇杆映射为线速度和角速度指令。
        """
        max_vx = 2.5
        max_vy = 0.8
        max_dyaw = 1.0

        ly = self.rc.state.LEFT_Y   # 左摇杆上下 (控制前进/后退 vx)
        lx = self.rc.state.LEFT_X   # 左摇杆左右 (控制平移 vy)
        rx = self.rc.state.RIGHT_X  # 右摇杆左右 (控制自转 dyaw)

        # 增加死区，防止摇杆回中不准导致机器人原地乱动
        ly = 0.0 if abs(ly) < 0.1 else ly
        lx = 0.0 if abs(lx) < 0.1 else lx
        rx = 0.0 if abs(rx) < 0.1 else rx

        vx = ly * max_vx
        vy = lx * max_vy
        dyaw = rx * max_dyaw

        return np.array([vx, vy, dyaw], dtype=np.float32)

    def init_robot(self):
        """AMP 初始化：平滑过渡到 default_pos 默认马步姿态"""
        print("[INFO] 开始平滑过渡到 AMP 默认准备姿态...")
        self.get_leg_state() 
    
        # 强制只处理 12 自由度腿部
        self.legActions = 12
        while len(self.legCommand.position) < self.legActions:
            self.legCommand.position.append(0.0)
            
        q0_leg = [self.legState.position[i] for i in range(self.legActions)]

        T = 2.0  # 2秒过渡
        dt = 0.002
        tt = 0.0
        timer = NanoSleep(2)
        
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            self.get_leg_state() 
            
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            
            # ========== 腿部：平滑位置 + 高刚度 ==========
            for i in range(self.legActions):
                self.legCommand.position[i] = s0 * q0_leg[i] + s1 * self.default_joints_real[i]
                
            self.set_leg_command() 
            tt += dt
            timer.waiting(start_time)
            
        print("[SUCCESS] 已就位！请按手柄 START 开始接受摇杆遥控，LT 压到底急停")
        
        # 死循环等待 START 键，期间死死锁住准备姿态
        while not self.rc.state.START:
            start_time = time.perf_counter()
            self.get_leg_state() 
            
            for i in range(self.legActions):
                self.legCommand.position[i] = self.default_joints_real[i]
                self.legCommand.kp[i] = 150.0
                self.legCommand.kd[i] = 5.0
                
            self.set_leg_command() 
            
            if self.rc.state.LT > 64:
                print("紧急停止！！！")
                exit()
            timer.waiting(start_time)

    def get_obs(self, vel_cmd):
        """提取真实物理数据，构造 AMP 所需的 450 维观测向量"""
        q_real = np.zeros(self.num_actions)
        dq_real = np.zeros(self.num_actions)

        # 提取真实腿部数据
        for i in range(self.legActions):
            q_real[i] = self.legState.position[i]
            dq_real[i] = self.legState.velocity[i]

        q_isaac = q_real[Real_to_Isaac_indices]
        dq_isaac = dq_real[Real_to_Isaac_indices]

        # 角速度
        cur_ang_vel = np.array(self.legState.imu_gyro, dtype=np.float32)
        
        # 投影重力
        base_euler = np.array(self.legState.imu_euler)
        base_euler[base_euler > math.pi] -= 2 * math.pi
        quat_wxyz = euler_to_quaternion(base_euler[0], base_euler[1], base_euler[2])
        r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        cur_proj_grav = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.float32)

        # 速度指令
        cur_vel_cmd = vel_cmd.astype(np.float32)

        # 关节位置误差与速度
        cur_joint_pos = (q_isaac - self.default_joints_isaac).astype(np.float32)
        cur_joint_vel = dq_isaac.astype(np.float32)
        
        # 上一帧动作
        cur_action = self.last_action_isaac.astype(np.float32)

        # 滑动窗口更新
        if self.is_first_frame:
            self.hist_ang_vel[:]   = cur_ang_vel
            self.hist_proj_grav[:] = cur_proj_grav
            self.hist_vel_cmd[:]   = cur_vel_cmd
            self.hist_joint_pos[:] = cur_joint_pos
            self.hist_joint_vel[:] = cur_joint_vel
            self.hist_actions[:]   = cur_action
            self.is_first_frame = False
        else:
            self.hist_ang_vel   = np.roll(self.hist_ang_vel,   -1, axis=0); self.hist_ang_vel[-1]   = cur_ang_vel
            self.hist_proj_grav = np.roll(self.hist_proj_grav, -1, axis=0); self.hist_proj_grav[-1] = cur_proj_grav
            self.hist_vel_cmd   = np.roll(self.hist_vel_cmd,   -1, axis=0); self.hist_vel_cmd[-1]   = cur_vel_cmd
            self.hist_joint_pos = np.roll(self.hist_joint_pos, -1, axis=0); self.hist_joint_pos[-1] = cur_joint_pos
            self.hist_joint_vel = np.roll(self.hist_joint_vel, -1, axis=0); self.hist_joint_vel[-1] = cur_joint_vel
            self.hist_actions   = np.roll(self.hist_actions,   -1, axis=0); self.hist_actions[-1]   = cur_action

        # 按 Term 展平拼接 -> 450维
        policy_input = np.concatenate([
            self.hist_ang_vel.reshape(-1),
            self.hist_proj_grav.reshape(-1),
            self.hist_vel_cmd.reshape(-1),
            self.hist_joint_pos.reshape(-1),
            self.hist_joint_vel.reshape(-1),
            self.hist_actions.reshape(-1),
        ]).reshape(1, -1).astype(np.float32)

        return policy_input

    def run(self):
        count_lowlevel = 0
        actual_sleep_ms = self.dt * 1000  # 5ms (200Hz) 定时器
        timer_main = NanoSleep(actual_sleep_ms)  
        
        print("[INFO] AMP 遥控模式已启动，请推动摇杆...")
        
        while True:
            start_time = time.perf_counter()
            self.get_leg_state() 
            
            if self.rc.state.LT > 64:
                print("\n[INFO] 紧急停止按键触发！安全下线。")
                break
                
            # --- 50Hz 策略网络执行层 (每 4 个底层循环跑 1 次) ---
            if count_lowlevel % self.decimation == 0:
                
                # 实时从手柄读取摇杆指令 [vx, vy, dyaw]
                vel_cmd = self.get_velocity_command()
                
                # 提取 450维 历史观测量
                obs = self.get_obs(vel_cmd)
                
                # ONNX 神经网络推理
                input_name = self.onnx_policy.get_inputs()[0].name
                action_isaac = np.array(self.onnx_policy.run(None, {input_name: obs})[0][0])
                self.last_action_isaac = action_isaac.copy()
                
                # 从网络输出转换为绝对目标角度，并转回底层顺序
                target_q_isaac = action_isaac * self.action_scale_isaac + self.default_joints_isaac
                self.target_q_real = target_q_isaac[Isaac_to_Real_indices]
                
                # 终端调试打印当前的摇杆指令
                if count_lowlevel % 100 == 0:
                     print(f"Cmd => Vx: {vel_cmd[0]:.2f} m/s, Vy: {vel_cmd[1]:.2f} m/s, Yaw: {vel_cmd[2]:.2f} rad/s")

            # 200Hz 物理执行层 (始终执行) 
            for i in range(self.legActions):
                self.legCommand.position[i] = self.target_q_real[i]
                
            self.set_leg_command() 
            count_lowlevel += 1
            timer_main.waiting(start_time)

if __name__ == '__main__':
    gBot = Sim2Real_AMP()
    gBot.init_robot()  
    gBot.run()