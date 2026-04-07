# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

"""
python scripts/mujoco/sim2sim_e1_bm.py
  or
python scripts/mujoco/sim2sim_e1_bm.py \
      --policy_model logs/rsl_rl/e1_beyondmimic/<run>/exported/policy.onnx \
      --motion_file file_path
      --loop
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort
import cv2
import time
from pynput import keyboard
from loop_rate_limiters import RateLimiter
from robolab.assets import ISAAC_DATA_DIR
from scripts.tools.foxshow import FoxShow


class Cmd:
    camera_follow = True
    reset_requested = False

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")


def on_press(key_evt):
    try:
        if key_evt.char == 'f':
            Cmd.toggle_camera_follow()
        elif key_evt.char == '0':
            Cmd.reset_requested = True
    except AttributeError:
        pass


def on_release(key):
    pass


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def get_obs(data):
    """从 MuJoCo data 中提取观测量。"""
    q     = data.qpos.astype(np.double)
    dq    = data.qvel.astype(np.double)
    quat  = data.sensor('bq').data[[1, 2, 3, 0]].astype(np.double)
    r     = R.from_quat(quat)
    v     = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('gyro').data.astype(np.double)
    gvec  = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD 力矩计算。"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg, headless=False, loop=False, motion_file=None, foxshow=False):
    """
    Run the MuJoCo simulation with the E1 BeyondMimic policy.

    Args:
        policy: ONNX InferenceSession.
        cfg: Sim2simCfg configuration object.
        headless: If True, render offscreen and save video.
        loop: If True, loop the reference motion.
        motion_file: Path to .npz motion file.
    """
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1

    print("=" * 60)
    print("Keyboard control instructions:")
    print("  0 key: Reset simulation")
    print("  F key: Toggle camera follow mode")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()

    # ── Load motion reference data ────────────────────────────────────────────
    motion      = np.load(motion_file)
    motion_pos  = motion["body_pos_w"]   # (T, N_bodies, 3)
    m_joint_pos = motion["joint_pos"]    # (T, 21) in Isaac Lab joint order
    m_joint_vel = motion["joint_vel"]    # (T, 21) in Isaac Lab joint order
    num_frames  = min(m_joint_pos.shape[0], m_joint_vel.shape[0], motion_pos.shape[0])

    # ── MuJoCo setup ─────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep  = cfg.sim_config.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    data.qpos[0:3] = motion_pos[0, 0, :]   # initialise near first-frame reference body
    mujoco.mj_step(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['MUJOCO_GL'] = 'glfw'

    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        cam = mujoco.MjvCamera()
        cam.distance  = 4.0
        cam.azimuth   = 45.0
        cam.elevation = -20.0
        cam.lookat    = [0, 0, 1.0]
        out = cv2.VideoWriter(
            'simulation_e1_bm.mp4', fourcc,
            1.0 / cfg.sim_config.dt / cfg.sim_config.decimation,
            (1920, 1080)
        )
    else:
        viewer = mujoco_viewer.MujocoViewer(model, data, mode='window', width=1920, height=1080)
        viewer.cam.distance  = 4.0
        viewer.cam.azimuth   = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat    = [0, 0, 1.0]

    num_actions = cfg.robot_config.num_actions

    if foxshow:
        urdf_path = cfg.sim_config.mujoco_model_path.replace('.xml', '.urdf')
        fox = FoxShow(urdf_path)
        fox_joint_names = fox.get_joint_names()
        mujoco_joint_names = [model.joint(i).name for i in range(1, 1 + num_actions)]
        urdf_to_mujoco = [mujoco_joint_names.index(name) for name in fox_joint_names]

    target_pos  = np.zeros(num_actions, dtype=np.double)
    action      = np.zeros(num_actions, dtype=np.double)   # Isaac Lab order
    tau         = np.zeros(num_actions, dtype=np.double)

    # ── Per-term history buffers (H, dim), oldest → newest ───────────────────
    # E1 BeyondMimic policy observation terms (E1BeyondMimicEnvCfg):
    #   command        : motion joint_pos(21) + joint_vel(21) = 42
    #   projected_gravity : 3
    #   base_ang_vel   : 3
    #   joint_pos      : joint_pos_rel(21)
    #   joint_vel      : joint_vel_rel(21)
    #   actions        : last_action(21)
    #   Total single obs = 111,  history_length = 10,  policy input = 1110
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
    control_freq   = 1.0 / (cfg.sim_config.dt * cfg.sim_config.decimation)

    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # ── Reset request ────────────────────────────────────────────────────
        if Cmd.reset_requested:
            print('Performing reset')
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            Cmd.reset_requested = False
            motion_t = 0
            hist_cmd.fill(0.0);       hist_proj_grav.fill(0.0); hist_ang_vel.fill(0.0)
            hist_joint_pos.fill(0.0); hist_joint_vel.fill(0.0); hist_actions.fill(0.0)
            is_first_frame = True

        q, dq, quat, v, omega, gvec = get_obs(data)
        q  = q[-num_actions:]
        dq = dq[-num_actions:]

        # ── Policy step (50 Hz) ──────────────────────────────────────────────
        if count_lowlevel % cfg.sim_config.decimation == 0:
            idx = frame_idx(motion_t)

            # MuJoCo joint order → Isaac Lab joint order
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

            # Sliding window update (first frame: fill entire buffer)
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

            # Term-wise flattened input (matches IsaacLab flatten_history_dim order)
            # [cmd×H(420) | proj_grav×H(30) | ang_vel×H(30) | jpos×H(210) | jvel×H(210) | act×H(210)]
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

            # Isaac Lab order → MuJoCo order target positions
            target_q = action * cfg.robot_config.action_scale
            for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                target_pos[mujoco_idx] = target_q[isaac_idx]
            target_pos_abs = target_pos + cfg.robot_config.default_pos

            if foxshow:
                tic = int(step * cfg.sim_config.dt * 1000)
                fox.update_robot_state(
                    tic=tic,
                    pos=q[urdf_to_mujoco],
                    vel=dq[urdf_to_mujoco],
                    trans=data.xpos[model.body('pelvis').id].tolist(),
                    quat=quat,
                )
                fox.update_robot_target(tic=tic, pos=target_pos_abs[urdf_to_mujoco])

            # Render
            if headless:
                renderer.update_scene(data, camera=cam)
                if Cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                out.write(renderer.render())
            else:
                if Cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    viewer.cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                viewer.render()

            motion_t += 1
            rate_limiter = RateLimiter(frequency=control_freq, warn=False)
            rate_limiter.sleep()

        else:
            target_pos_abs = target_pos + cfg.robot_config.default_pos

        # ── PD control → torques ─────────────────────────────────────────────
        target_vel = np.zeros(num_actions, dtype=np.double)
        tau = pd_control(
            target_pos_abs, q, cfg.robot_config.kps,
            target_vel,     dq, cfg.robot_config.kds,
        )
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)
        count_lowlevel += 1

    if headless:
        out.release()
    else:
        viewer.close()
    keyboard_listener.stop()
    if foxshow:
        fox.stop_server()
    print("Simulation finished.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E1 BeyondMimic Sim2Sim deployment script.')
    parser.add_argument('--policy_model', type=str, default=None,
                        help='Path to exported policy.onnx '
                             '(e.g. logs/rsl_rl/e1_beyondmimic/<run>/exported/policy.onnx)')
    parser.add_argument('--motion_file', type=str, default=None,
                        help='Path to reference motion .npz (default: data/motions/e1_bm/MJ_dance.npz)')
    parser.add_argument('--headless', action='store_true',
                        help='Render offscreen and save simulation_e1_bm.mp4')
    parser.add_argument('--loop', action='store_true',
                        help='Loop the reference motion indefinitely')
    parser.add_argument('--foxshow', action='store_true',
                        help='Enable Foxglove real-time visualization')
    args = parser.parse_args()

    policy_model = args.policy_model
    if policy_model is None:
        policy_model = os.path.join(ISAAC_DATA_DIR, 'policies', 'bm', 'policy.onnx')
    motion_file = args.motion_file
    if motion_file is None:
        motion_file = os.path.join(ISAAC_DATA_DIR, 'motions', 'e1_bm', 'MJ_dance.npz')

    class Sim2simCfg:

        class sim_config:
            # E1_21dof.xml: 21 actuated joints, gyro/bq IMU sensors
            mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/droidrobot/E1/E1_21dof.xml'
            sim_duration = 2000000.0
            dt           = 0.005   # 200 Hz physics (matches Isaac Lab BeyondMimic)
            decimation   = 4       # 50 Hz policy

        class robot_config:
            # ── PD gains (MuJoCo actuator order) ─────────────────────────────
            # MuJoCo order:
            #   [L_pitch, L_roll, L_yaw, L_knee, L_ank_pitch, L_ank_roll,
            #    R_pitch, R_roll, R_yaw, R_knee, R_ank_pitch, R_ank_roll,
            #    waist_yaw,
            #    L_sho_pitch, L_sho_roll, L_sho_yaw, L_elbow,
            #    R_sho_pitch, R_sho_roll, R_sho_yaw, R_elbow]
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

            # ── Default joint positions (MuJoCo order) ───────────────────────
            default_pos = np.array([
                -0.1,  0.0,  0.0,  0.2,  -0.1,  0.0,   # L leg
                -0.1,  0.0,  0.0,  0.2,  -0.1,  0.0,   # R leg
                 0.0,                                    # waist_yaw
                 0.18, 0.06, 0.06, 0.78,                # L arm
                 0.18, 0.06, 0.06, 0.78,                # R arm
            ], dtype=np.double)

            # ── Torque limits (MuJoCo order, from URDF actuatorfrcrange) ─────
            tau_limit = np.array([
                60,   60,   36,   59.3, 59.3, 14,    # L leg
                60,   60,   36,   59.3, 59.3, 14,    # R leg
                60,                                    # waist_yaw
                36,   36,   14,   36,                 # L arm
                36,   36,   14,   36,                 # R arm
            ], dtype=np.double)

            # ── Isaac Lab BFS order → MuJoCo DFS order ───────────────────────
            # Isaac Lab (BFS traversal of the kinematic tree):
            #   idx  0: left_hip_pitch       idx  1: right_hip_pitch
            #   idx  2: waist_yaw
            #   idx  3: left_hip_roll        idx  4: right_hip_roll
            #   idx  5: left_shoulder_pitch  idx  6: right_shoulder_pitch
            #   idx  7: left_hip_yaw         idx  8: right_hip_yaw
            #   idx  9: left_shoulder_roll   idx 10: right_shoulder_roll
            #   idx 11: left_knee            idx 12: right_knee
            #   idx 13: left_shoulder_yaw   idx 14: right_shoulder_yaw
            #   idx 15: left_ankle_pitch     idx 16: right_ankle_pitch
            #   idx 17: left_elbow           idx 18: right_elbow
            #   idx 19: left_ankle_roll      idx 20: right_ankle_roll
            # usd2urdf[isaac_idx] = mujoco_idx
            usd2urdf = [0, 6, 12, 1, 7, 13, 17, 2, 8, 14, 18, 3, 9, 15, 19, 4, 10, 16, 20, 5, 11]

            num_actions    = 21
            action_scale   = 0.25
            frame_stack    = 10    # history_length = 10 (from BeyondMimicEnvCfg.PolicyCfg)
            # num_single_obs = 111  (42 cmd + 3 grav + 3 ang_vel + 21 jpos + 21 jvel + 21 act)
            # policy input   = 1110 (111 × 10)

    policy = ort.InferenceSession(policy_model)
    run_mujoco(policy, Sim2simCfg(), args.headless, args.loop, motion_file, args.foxshow)
