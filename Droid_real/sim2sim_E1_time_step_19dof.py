import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime
import torch
from scipy.spatial.transform import Rotation as R

# ============================== 路径与运行配置 ==============================
# 【请修改此路径】指向你的实际文件
XML_PATH = "/home/saw/droidup/Droid_robot/droidup_mujoco/assets/droidup/e1/E1_19dof.xml"  # E1 机器人的 xml 描述文件
ONNX_POLICY_PATH = "/home/saw/droidup/Droid_robot/Droid_real/policles/model_19dof_1.onnx"
MOTION_FILE_PATH = "/home/saw/droidup/Droid_robot/Droid_real/motion/dance80_600_19dof.npz"

# 仿真运行参数
simulation_duration = 3000.0  # 足够长的仿真时间 (秒)
simulation_dt = 0.002         # 底层物理步长
control_decimation = 10       # 控制策略下发间隔 (50Hz)
# =========================================================================

# 机器人硬件配置 (E1)
ROBOT_CONFIG = {
    "num_actions": 19,
    "num_obs": 104,
    "reference_body": "torso_link",
    "joint_names": [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
    ],
    "motion_body_index": 0
}

# ----------------- 纯数学与空间变换工具函数 -----------------
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def get_obs(data):
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[0, 1, 2, 3]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)
    return (qpos, dq, quat, v, omega, gvec, state_tau)

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

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def create_observation(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions):
    cmd_size = len(motioninput)
    obs[offset:offset + cmd_size] = motioninput 
    offset += cmd_size
    obs[offset:offset + 6] = motion_ref_ori_b 
    offset += 6
    obs[offset:offset + 3] = omega 
    offset += 3
    obs[offset:offset + num_actions] = qpos_seq - joint_pos_array_seq 
    offset += num_actions
    obs[offset:offset + num_actions] = qvel_seq 
    offset += num_actions   
    obs[offset:offset + num_actions] = action_buffer 
    return obs
# ------------------------------------------------------------

def run_simulation():
    print(f"[INFO]: Loading motion file from: {MOTION_FILE_PATH}")
    motion = np.load(MOTION_FILE_PATH)
    motionpos = motion["body_pos_w"]
    motionquat = motion["body_quat_w"]
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]

    # motioninputpos[:,:5] = 0
    # motioninputvel[:,:5] = 0

    # motioninputpos[:,7:9] = 0
    # motioninputvel[:,7:9] = 0

    # motioninputpos[:,11:13] = 0
    # motioninputvel[:,11:13] = 0

    # motioninputpos[:,7:9] = 0
    # motioninputvel[:,16:] = 0
    
    num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0], motionpos.shape[0], motionquat.shape[0])
    
    # 强制循环索引
    def frame_idx(t):
        return t % num_frames if num_frames > 0 else 0
    
    print(f"[INFO]: Loading ONNX policy from: {ONNX_POLICY_PATH}")
    model = onnx.load(ONNX_POLICY_PATH)
    
    joint_seq, joint_pos_array_seq, stiffness_array_seq, damping_array_seq, action_scale = None, None, None, None, None
    for prop in model.metadata_props:
        if prop.key == "joint_names": joint_seq = prop.value.split(",")
        elif prop.key == "default_joint_pos": joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_stiffness": stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_damping": damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "action_scale": action_scale = np.array([float(x) for x in prop.value.split(",")])
    
    joint_xml = ROBOT_CONFIG["joint_names"]
    if joint_seq is None:
        raise ValueError("Model metadata does not contain 'joint_names'")
        
    joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    
    num_actions = ROBOT_CONFIG["num_actions"]
    num_obs = ROBOT_CONFIG["num_obs"]
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    print(f"[INFO]: Loading XML model from: {XML_PATH}")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    policy = onnxruntime.InferenceSession(ONNX_POLICY_PATH)
    policy_input_names = [i.name for i in policy.get_inputs()]  # 动态读取需要的输入 (自适应 time_step)

    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0 
    
    motion_body_idx = ROBOT_CONFIG["motion_body_index"]
    target_dof_pos = joint_pos_array.copy()

    print("[INFO]: Simulation started (Loop mode active). Close viewer to exit.")

    # ================= 测试频率用： 初始化计时变量 =================
    test_step_count = 0
    test_start_time = time.time()
    # =================================================================

    # ================== 全局时间锚点 ==================
    next_target_time = time.time() 
    # =========================================================

    # ================= [新增] 强制物理状态初始化 =================
    print("[INFO]: Initializing robot posture to match the first frame of motion...")
    
    # 1. 提取舞蹈动作的第 0 帧
    init_idx = 0
    init_motion_pos = motionpos[init_idx, motion_body_idx, :]
    init_motion_quat = motionquat[init_idx, motion_body_idx, :]
    init_joint_pos = motioninputpos[init_idx, :]

    # 设置机器人基座（Torso/Pelvis）在世界坐标系的初始高度和姿态
    #d.qpos[0:3] = init_motion_pos  # X, Y, Z 位置
    d.qpos[3:7] = init_motion_quat # W, X, Y, Z 四元数姿态

    # 设置 19 个关节的初始角度（必须从序列顺序映射到 XML 顺序）
    mapped_init_joint_pos = np.array([init_joint_pos[joint_seq.index(joint)] for joint in joint_xml])
    d.qpos[7:7 + num_actions] = mapped_init_joint_pos

    # 更新一次物理状态
    mujoco.mj_forward(m, d)
    # ==============================================================
    # ================= [新增] 锁定初始控制目标 =================
    # 让 PD 控制器一开始死死抱住第 0 帧的姿势，不要乱动
    target_dof_pos = mapped_init_joint_pos.copy() 
    # ========================================================

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        d.time = 0.0  # 确保仿真内部时间从 0 开始计数
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            mujoco.mj_step(m, d)
            qpos, dq, quat, v, omega, gvec, state_tau = get_obs(d)
            
            tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)
            d.ctrl[:] = tau
            counter += 1

            if counter % control_decimation == 0: # 50Hz 控制计算
                
                # ================= 等待逻辑 =================
                if d.time < 0.05:
                    # 在前 0.05*10 秒内，我们只做物理仿真和画面渲染，不进行推理！
                    pass 
                else:
                    # 时间到，开始正常的 ONNX 推理和动作推进
                    idx = frame_idx(timestep)
                    motioninput = np.concatenate((motioninputpos[idx, :], motioninputvel[idx, :]), axis=0)
                    motionquatcurrent = motionquat[idx, motion_body_idx, :]
                    
                    offset = 0
                    q01 = quat  
                    q02 = motionquatcurrent 
                    q10 = quat_inv_np(q01) 
                    
                    q12 = quat_mul_np(q10, q02) if q02 is not None else q10
                    mat = matrix_from_quat(torch.from_numpy(q12)) 
                    motion_ref_ori_b = mat[..., :2].reshape(6) 
                    
                    qpos_xml = d.qpos[7:7 + num_actions]
                    qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                    qvel_xml = d.qvel[6:6 + num_actions]
                    qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                    
                    obs = create_observation(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions)
                    
                    # ------ 动态组织模型输入 ------
                    feed_dict = {'obs': torch.from_numpy(obs).unsqueeze(0).numpy()}
                    if 'time_step' in policy_input_names:
                        feed_dict['time_step'] = np.array([idx], dtype=np.float32).reshape(1, 1)
                    
                    action = policy.run(['actions'], feed_dict)[0]
                    # -----------------------------

                    action = np.asarray(action).reshape(-1)
                    action_buffer = action.copy()
                    target_dof_pos = action * action_scale + joint_pos_array_seq
                    target_dof_pos = target_dof_pos.reshape(-1,)
                    target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
                    
                    timestep += 1  # 只有开始跳舞了，时间帧才往后走
                # ===============================================

            # 渲染必须放在外面，保证等待的 2 秒内你也能看到画面
            viewer.sync()
            
            # 判断是否循环0.002s 如果没有则强行延时time_until_next_step 
            # 实际测试 viewer.sync() 渲染的时候达不到50Hz的要求，只有38Hz左右
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


            # ================= 替换为全局时间补偿逻辑 =================
            # next_target_time += m.opt.timestep  # 目标时间严格往后推 0.002s
            # sleep_duration = next_target_time - time.time()
            
            # if sleep_duration > 0: # 提前到了时间，则睡眠多出来的时间补齐0.02s
            #     time.sleep(sleep_duration)
            # elif sleep_duration < -0.02: 
            #     # 如果电脑实在太卡，落后超过 20ms (比如拖动了窗口)，
            #     # 就重置目标时间，防止程序疯狂快进补时间
            #     next_target_time = time.time()
            # 在-0.02以内不需要else直接进入下一次循环，因为通过几轮之后可以动态补偿
            # =============================================================

            # ================= 测试频率用：统计并打印频率 =================
            test_step_count += 1
            if test_step_count % 500 == 0:  # 每跑 500 次底层循环打印一次 (理论上是 1 秒)
                current_time = time.time()
                elapsed_time = current_time - test_start_time # 时间差值
                
                # 计算实际频率
                actual_sim_fps = 500 / elapsed_time  # 底层物理实际 Hz (目标是 500)
                actual_control_fps = actual_sim_fps / control_decimation # 神经网络实际 Hz (目标是 50)
                avg_step_ms = (elapsed_time / 500) * 1000 # 每次循环平均耗时 (目标是 2.0ms)
                
                print(f"[频率测试] 物理环境刷新: {actual_sim_fps:.1f} Hz (目标:500.0) | "
                      f"神经网络控制: {actual_control_fps:.1f} Hz (目标:50.0) | "
                      f"底层单步平均耗时: {avg_step_ms:.2f} ms")
                
                # 重置计时器，准备统计下一个 500 步
                test_start_time = time.time()
            # =================================================================

if __name__ == "__main__":
    run_simulation()
