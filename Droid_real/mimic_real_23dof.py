import sys
import os

# 获取当前脚本所在的目录绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import math
import time
import numpy as np
from tqdm import tqdm
import onnxruntime as ort

from base.RobotBase import RobotBase
from base.Base import get_command
from base.Base import set_joint_mode
from tools.Gamepad import GamepadHandler
from base.Base import NanoSleep, euler_to_quaternion
from base.ConfigE1 import Config  


onnx_mode_path = f"/home/saw/RL/humanoid_robot/Droid/Droid_robot/Droid_real/policles/model_3.onnx"

# 用来跟踪的动作 NPZ 文件
motion_file_path = f"/home/saw/RL/humanoid_robot/Droid/Droid_robot/Droid_real/motion/dance3.npz"

# ================= 关节顺序配置 =================
# Isaac 训练环境里神经网络期望的顺序 (来自 ONNX Metadata)
IsaacLabJointOrder = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint',
    'left_wrist_roll_joint', 'right_wrist_roll_joint'
]

# 真实硬件（LegBase + ArmBase）底层拼接返回的物理顺序
RealJointOrder = [
    # LegBase (13个)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 
    # ArmBase (10个)
    'left_shoulder_pitch_joint','left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint'
]

Isaac_to_Real_indices = [IsaacLabJointOrder.index(joint) for joint in RealJointOrder]
Real_to_Isaac_indices = [RealJointOrder.index(joint) for joint in IsaacLabJointOrder]

# ================= 纯 NumPy 的四元数/矩阵运算 (替代 torch) =================
def matrix_from_quat_np(quaternions: np.ndarray) -> np.ndarray:
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
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

# ================= 主体控制逻辑 =================
class Sim2Real(RobotBase):
    def __init__(self):
        super().__init__(Config)

        # 23 DOF 维度配置
        self.num_actions = 23
        self.num_observations = 124  

        # 按照 RealJointOrder 严格排列的 default_joints
        self.default_joints = np.array([
            -0.300, 0.000, 0.000, 0.600, -0.300, 0.000,  # 左腿 6
            -0.300, 0.000, 0.000, 0.600, -0.300, 0.000,  # 右腿 6
            0.000,                                       # 腰部 1
            0.000, 0.000, 0.000, 0.000, 0.000,           # 左臂 5
            0.000, 0.000, 0.000, 0.000, 0.000            # 右臂 5
        ])
        
        # 按照 RealJointOrder 严格提取的 action_scale
        self.action_scale = np.array([
            0.045, 0.150, 0.090, 0.075, 0.450, 0.350,    # 左腿 6
            0.045, 0.150, 0.090, 0.075, 0.450, 0.350,    # 右腿 6
            0.075,                                       # 腰部 1
            0.090, 0.090, 0.175, 0.233, 0.900,           # 左臂 5
            0.090, 0.090, 0.175, 0.233, 0.900            # 右臂 5
        ])
        
        self.decimation = 10
        self.dt = 0.002
        self.run_flag = True

        self.target_q = np.zeros(self.num_actions, dtype=np.double)
        self.action = np.zeros(self.num_actions, dtype=np.double)
        
        # 载入模仿学习 NPZ 文件
        print(f"[INFO]: Loading motion file from {motion_file_path}")
        motion = np.load(motion_file_path)
        self.motionpos = motion["body_pos_w"]
        self.motionquat = motion["body_quat_w"]
        self.motioninputpos = motion["joint_pos"]
        self.motioninputvel = motion["joint_vel"]
        self.num_frames = min(self.motioninputpos.shape[0], self.motioninputvel.shape[0], self.motionpos.shape[0], self.motionquat.shape[0])
        self.timestep = 0
        self.motion_body_idx = 0  # e1的躯干索引
        
        # 初始化 ONNX 和手柄
        self.onnx_policy = ort.InferenceSession(onnx_mode_path)
        self.rc = GamepadHandler()

    def init_robot(self):
        print("default_joints: ", self.default_joints)
        
        # 先强制获取一次当前状态，探探机器人的底
        self.get_robot_state()
        
        # 读取机器人真实返回的数组长度，覆盖本地可能错误的 Config
        self.legActions = len(self.legState.position)
        self.armActions = len(self.armState.position)
        print(f"[DEBUG] 修正关节数量 -> 腿部及腰肩: {self.legActions}, 手臂: {self.armActions}")
        
        # 动态补齐底层 Command 数组长度 (防止赋值时越界)
        while len(self.legCommand.position) < self.legActions:
            self.legCommand.position.append(0.0)
        while len(self.armCommand.position) < self.armActions:
            self.armCommand.position.append(0.0)
            
        # 防呆检查
        if self.legActions + self.armActions != self.num_actions:
            print(f"严重警告：SDK读取的腿部({self.legActions}) + 手臂({self.armActions}) 数量不等于 {self.num_actions}！")
            
        T = 2.0  # 2秒平滑过渡
        dt = 0.002
        tt = 0.0
        timer = NanoSleep(2)
        
        while tt < T:
            start_time = time.perf_counter()
            self.get_robot_state()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            
            # 循环赋值
            for i in range(self.legActions):
                self.legCommand.position[i] = s0 * self.legState.position[i] + s1 * self.default_joints[i]
            for i in range(self.armActions):
                self.armCommand.position[i] = s0 * self.armState.position[i] + s1 * self.default_joints[self.legActions + i]
                
            self.set_robot_command()
            tt += dt
            timer.waiting(start_time)
            
        print("归位完成，单击 START 开始跟踪, LT 按压到底急停")
        while (self.rc.state.START == False) and (self.run_flag == True):
            start_time = time.perf_counter()
            self.get_robot_state()
            if self.rc.state.LT > 64:
                print("紧急停止！！！")
                exit()
            timer.waiting(start_time)

    def get_obs(self, timestep):
        q = np.zeros(self.num_actions)
        dq = np.zeros(self.num_actions)

        # 获取机器人的 23 个自由度的物理位置和速度 (这是 Real 顺序)
        for i in range(self.legActions):
            q[i] = self.legState.position[i]
            dq[i] = self.legState.velocity[i]
        for i in range(self.armActions):
            q[self.legActions + i] = self.armState.position[i]
            dq[self.legActions + i] = self.armState.velocity[i]

        # 将数据切片成神经网络认识的 Isaac 顺序
        q_isaac = q[Real_to_Isaac_indices]
        dq_isaac = dq[Real_to_Isaac_indices]
        default_isaac = self.default_joints[Real_to_Isaac_indices]
        action_isaac = self.action[Real_to_Isaac_indices]

        # 计算相对姿态误差 (motion_ref_ori_b)
        base_euler = np.array(self.legState.imu_euler)
        base_euler[base_euler > math.pi] -= 2 * math.pi
        eq = euler_to_quaternion(base_euler[0], base_euler[1], base_euler[2])
        q01 = np.array(eq, dtype=np.double)  # [w, x, y, z]
        
        q02 = self.motionquat[timestep, self.motion_body_idx, :]
        q10 = quat_inv_np(q01)
        q12 = quat_mul_np(q10, q02) if q02 is not None else q10
        mat = matrix_from_quat_np(q12)
        motion_ref_ori_b = mat[..., :2].reshape(6) # 取出旋转矩阵的前两列 xy

        # 获取当前需要跟踪的动作帧指令 (46维)
        motioninput = np.concatenate((
            self.motioninputpos[timestep, :], 
            self.motioninputvel[timestep, :]
        ), axis=0)

        base_ang_vel = np.array(self.legState.imu_gyro)

        # 按照 124 维结构拼接 observation
        obs = np.zeros([self.num_observations], dtype=np.float32)
        offset = 0
        
        cmd_size = len(motioninput)
        obs[offset:offset + cmd_size] = motioninput  # 46
        offset += cmd_size
        
        obs[offset:offset + 6] = motion_ref_ori_b  # 6
        offset += 6
        
        obs[offset:offset + 3] = base_ang_vel  # 3
        offset += 3
        
        obs[offset:offset + self.num_actions] = q_isaac - default_isaac  # 23
        offset += self.num_actions
        
        obs[offset:offset + self.num_actions] = dq_isaac  # 23
        offset += self.num_actions   
        
        obs[offset:offset + self.num_actions] = action_isaac  # 23
        
        obs = np.clip(obs, -100, 100)
        return obs

    def get_action(self, obs, timestep):
        obs_np = np.expand_dims(obs, axis=0)
        time_step_np = np.array([timestep], dtype=np.float32).reshape(1, 1)
        
        # 神经网络输出的是 Isaac 顺序的 Action
        action_isaac = np.array(self.onnx_policy.run(['actions'], {
            'obs': obs_np,
            'time_step': time_step_np
        })[0].tolist()[0])
        
        # 将输出转换回 Real 实机硬件顺序，存入 self.action 用于下一次观测
        self.action = np.clip(action_isaac[Isaac_to_Real_indices], -100.0, 100.0)
        
        # 按 Real 实机顺序计算最终发送的物理角度
        return self.action * self.action_scale + self.default_joints

    def run(self):
        pre_tic = 0
        # 理论推理频率 50Hz
        duration_second = self.decimation * self.dt  
        
        slowdown_factor = 1.0  # 调整频率参数
        
        # 根据实际调整这个参数，延缓动作频率（目前不清楚什么原因导致，不知道是否是和底层仿真平台通信存在延迟，导致和直接采用mujoco仿真对比存在gap）
        actual_sleep_ms = duration_second * 1000 * slowdown_factor

        # 创建定时器 传入拉等待时间
        timer = NanoSleep(actual_sleep_ms)  
        pbar = tqdm(range(int(0xfffffff0 / duration_second)), desc="E1 Motion Tracking...")

        start = time.perf_counter()

        # ================= 测试频率用： 初始化计时变量 =================
        test_step_count = 0
        test_start_time = time.time()
        # =================================================================
        
        while True:
            start_time = time.perf_counter()
            self.get_robot_state()
            
            if self.rc.state.LT > 64:
                print("紧急停止！！！")
                exit()
                
            # 提取当前观测
            obs = self.get_obs(self.timestep)
            
            # 推理目标关节角 返回的已经是 Hardware 顺序
            self.target_q = self.get_action(obs, self.timestep)
            
            # 下发关节指令
            for i in range(self.legActions):
                self.legCommand.position[i] = self.target_q[i]
            for i in range(self.armActions):
                self.armCommand.position[i] = self.target_q[self.legActions + i]
                
            self.set_robot_command()
            
            # 时间步前进，并保证循环播放 NPZ
            self.timestep = (self.timestep + 1) % self.num_frames
            
            pbar.set_postfix(
                realCycle=f"{self.legState.system_tic - pre_tic}ms", 
                runTime=f"{(time.perf_counter() - start):.3f}s",
                frameIdx=self.timestep
            )
            pre_tic = self.legState.system_tic

            timer.waiting(start_time) #动态延时

            # ================= 测试频率用：统计并打印频率 =================
            test_step_count += 1
            if test_step_count % 50 == 0:  # 每跑 50 次控制循环打印一次 (理论上正好是 1 秒)
                current_time = time.time()
                elapsed_time = current_time - test_start_time
                
                # 计算实际控制频率
                actual_control_fps = 50 / elapsed_time  # 神经网络实际 Hz (目标是 50)
                avg_step_ms = (elapsed_time / 50) * 1000  # 每次循环平均耗时 (目标是 20.0ms)
                
                print(f"[真机频率测试] 神经网络控制: {actual_control_fps:.1f} Hz (目标:50.0) | "
                      f"单步平均耗时: {avg_step_ms:.2f} ms")
                
                # 重置计时器，准备统计下一个 50 步
                test_start_time = time.time()
            # =================================================================

if __name__ == '__main__':
    mybot = Sim2Real()
    mybot.init_robot()  
    time.sleep(1)
    mybot.run()