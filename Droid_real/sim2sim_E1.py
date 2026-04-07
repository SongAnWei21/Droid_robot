"""Unified sim2sim script for E1 robot.

Usage Examples:
    # For E1 robot:
    python sim2sim.py --robot e1 --motion_file source/motion/hightorque/e1/npz/dance1.npz \
    --xml_path /path/to/e1.xml --policy_path /path/to/e1_policy.onnx --save_json
"""

import argparse
import json
import time
import os

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime
import torch
from scipy.spatial.transform import Rotation as R

# Simulation parameters
simulation_duration = 300.0
simulation_dt = 0.002
control_decimation = 10

# Robot configurations
ROBOT_CONFIGS = {
    "e1": {
        "num_actions": 23,
        "num_obs": 124,
        "reference_body": "torso_link",
        "default_xml": None,  # Must be provided
        "joint_names": [
            'left_hip_pitch_joint',
            'left_hip_roll_joint',
            'left_hip_yaw_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',

            'right_hip_pitch_joint',
            'right_hip_roll_joint',
            'right_hip_yaw_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint', 

            "waist_yaw_joint",

            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",

            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 46,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 23,
            "joint_vel": 23,
            "actions": 23
        }
    }
}


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
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
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[0, 1, 2, 3]].astype(np.double)
    
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)

    return (qpos, dq, quat, v, omega, gvec, state_tau)


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v (NumPy version)."""
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    
    if q_vec.ndim == 2:
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0
    
    return a - b + c


def quaternion_multiply(q1, q2):
    """Quaternion multiplication: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together."""
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    
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


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes the inverse of a quaternion."""
    return quat_conjugate_np(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def create_observation(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions):
    """Create observation for the robot."""
    cmd_size = len(motioninput)
    obs[offset:offset + cmd_size] = motioninput # 46 joint_pos + joint_vel
    offset += cmd_size
    obs[offset:offset + 6] = motion_ref_ori_b # 6
    offset += 6
    obs[offset:offset + 3] = omega # 3
    offset += 3
    obs[offset:offset + num_actions] = qpos_seq - joint_pos_array_seq # 23
    offset += num_actions
    obs[offset:offset + num_actions] = qvel_seq # 23
    offset += num_actions   
    obs[offset:offset + num_actions] = action_buffer # 23
    return obs


def run_simulation(robot_type: str, motion_file: str, xml_path: str, policy_path: str, save_json: bool = False, loop: bool = False):
    """Run the sim2sim simulation."""
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Robot type '{robot_type}' not found in configurations.")

    config = ROBOT_CONFIGS[robot_type]
    print(f"[INFO]: Using robot configuration: {robot_type}")
    print(f"[INFO]: Actions: {config['num_actions']}, Observations: {config['num_obs']}")
    
    # Load motion data
    motion = np.load(motion_file)
    motionpos = motion["body_pos_w"]
    motionquat = motion["body_quat_w"]
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]
    
    # number of frames available across all sequences
    num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0], motionpos.shape[0], motionquat.shape[0])
    
    # safe index helper (supports looping)
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1
    
    # Save motion data to JSON if requested
    if save_json:
        motion_dict = {
            "body_pos_w": motionpos.tolist(),
            "body_quat_w": motionquat.tolist(),
            "joint_pos": motioninputpos.tolist(),
            "joint_vel": motioninputvel.tolist()
        }
        motion_dir = os.path.dirname(motion_file)
        motion_basename = os.path.basename(motion_file)
        
        # Replace 'npz' directory with 'json' directory
        if motion_dir.endswith('/npz') or motion_dir.endswith('\\npz'):
            json_dir = motion_dir[:-3] + 'json'
        else:
            json_dir = motion_dir
            
        os.makedirs(json_dir, exist_ok=True)
        
        json_basename = motion_basename.replace('.npz', '.json')
        json_filename = os.path.join(json_dir, json_basename)
        with open(json_filename, 'w') as f:
            json.dump(motion_dict, f, indent=2)
        print(f"[INFO]: Motion data saved to: {json_filename}")
    
    # Load ONNX model and extract metadata
    model = onnx.load(policy_path)
    joint_seq = None
    joint_pos_array_seq = None
    stiffness_array_seq = None
    damping_array_seq = None
    action_scale = None
    
    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        elif prop.key == "default_joint_pos":   
            joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_stiffness":
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_damping":
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "action_scale":
            action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(f"{prop.key}: {prop.value}")
    
    # Remap to XML joint order
    joint_xml = config["joint_names"]
    
    # Ensure mapping is possible
    if joint_seq:
        joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
        stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
        damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    else:
        raise ValueError("Model metadata does not contain joint_names")
 
    print("stiffness_array", stiffness_array)
    print("damping_array", damping_array)
    print("action_scale", action_scale)
    
    # Initialize variables
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # Load policy
    policy = onnxruntime.InferenceSession(policy_path)

    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0 
    motioninput = np.concatenate((motioninputpos[frame_idx(timestep), :], motioninputvel[frame_idx(timestep), :]), axis=0)
    
    motion_body_idx = config["motion_body_index"]
    motionposcurrent = motionpos[frame_idx(timestep), motion_body_idx, :]
    motionquatcurrent = motionquat[frame_idx(timestep), motion_body_idx, :]
    
    target_dof_pos = joint_pos_array.copy()
    
    # Set reference body
    body_name = config["reference_body"]
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")


    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            mujoco.mj_step(m, d)
            qpos, dq, quat, v, omega, gvec, state_tau = get_obs(d)
            tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)
        
            d.ctrl[:] = tau
            counter += 1
            
            if counter % control_decimation == 0: # 50Hz
                # Update motion data
                idx = frame_idx(timestep)
                motioninput = np.concatenate((motioninputpos[idx, :], motioninputvel[idx, :]), axis=0)
                motionquatcurrent = motionquat[idx, motion_body_idx, :]
                
                # Create observations
                offset = 0
                
                robot_quat_w = torch.from_numpy(quat).unsqueeze(0)
                q01 = quat  # 获取真实四元数
                q02 = motionquatcurrent # 当前动作四元数
                q10 = quat_inv_np(q01) # 参考系逆转，从机器人自身的视角去看世界坐标系

                # 计算“相对姿态误差”，利用四元数乘法把目标姿态转换到了机器人的局部坐标系下 
                # 强化学习策略（Policy）不应该关心机器人在世界坐标系的绝对朝向（比如面朝东还是面朝西），它只需要关心“为了达到目标动作，我当前还需要转动多少度”。q12 就是这个“还需要转动的量”。
                if q02 is not None:
                    q12 = quat_mul_np(q10, q02)
                else:
                    q12 = q10
                mat = matrix_from_quat(torch.from_numpy(q12)) # 四元数转旋转矩阵

                # 提取连续的 6D 旋转表示
                # 取旋转矩阵的前两列（即 X 轴和 Y 轴的朝向向量）， 三维空间中只要确定了 X 轴和 Y 轴的向量，Z 轴的值可以通过两者的叉乘唯一确定，减小冗余输入
                motion_ref_ori_b = mat[..., :2].reshape(6) 
                
                qpos_xml = d.qpos[7:7 + num_actions]
                qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                qvel_xml = d.qvel[6:6 + num_actions]
                qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                
                obs = create_observation(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions)
                
                # Run policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy.run(['actions'], {
                    'obs': obs_tensor.numpy(),
                    'time_step': np.array([frame_idx(timestep)], dtype=np.float32).reshape(1, 1)
                })[0]
                
                action = np.asarray(action).reshape(-1)
                action_buffer = action.copy()
                target_dof_pos = action * action_scale + joint_pos_array_seq
                target_dof_pos = target_dof_pos.reshape(-1,)
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
                
                # advance time step
                if loop or timestep + 1 < num_frames:
                    timestep += 1

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser(description="Unified sim2sim script for E1 robot.")
    parser.add_argument("--robot", type=str, choices=["e1"], required=True,
                        help="Robot type: e1")
    parser.add_argument("--motion_file", type=str, required=True, 
                        help="Path to the motion NPZ file")
    parser.add_argument("--xml_path", type=str, required=True,
                        help="Path to the robot XML file")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to the ONNX policy file")
    parser.add_argument("--save_json", action="store_true",
                        help="Save motion data to JSON file")
    parser.add_argument("--loop", action="store_true",
                        help="Loop motion/policy when reaching the end of sequence")
    
    args = parser.parse_args()
    
    print(f"[INFO]: Robot: {args.robot}")
    print(f"[INFO]: Motion file: {args.motion_file}")
    print(f"[INFO]: XML path: {args.xml_path}")
    print(f"[INFO]: Policy path: {args.policy_path}")
    
    run_simulation(args.robot, args.motion_file, args.xml_path, args.policy_path, args.save_json, args.loop)


if __name__ == "__main__":
    main()

