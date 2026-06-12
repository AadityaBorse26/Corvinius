import os
import mujoco
import numpy as np

# load model xml directly
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def solve_ik(model, data, target_pos, site_name="eef", max_steps=30, step_size=0.15):
    # ik solver using damped least squares
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    joint_names = [f"joint{i}" for i in range(1, 7)]
    joint_qpos_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
    dof_ids = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]

    # end-effector oriented pointing down
    R_target = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])

    for _ in range(max_steps):
        mujoco.mj_fwdPosition(model, data)
        site_xpos = data.site_xpos[site_id]
        site_xmat = data.site_xmat[site_id].reshape(3, 3)
        
        # positional and rotational error
        err_pos = target_pos - site_xpos
        err_rot = 0.5 * (
            np.cross(site_xmat[:, 0], R_target[:, 0]) +
            np.cross(site_xmat[:, 1], R_target[:, 1]) +
            np.cross(site_xmat[:, 2], R_target[:, 2])
        )
        
        err = np.hstack([err_pos, err_rot])
        if np.linalg.norm(err) < 1e-4:
            break
            
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        
        for idx, q_adr in enumerate(joint_qpos_adr):
            data.qpos[q_adr] += dq[idx] * step_size
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[idx])
            j_range = model.jnt_range[j_id]
            data.qpos[q_adr] = np.clip(data.qpos[q_adr], j_range[0], j_range[1])

    return np.linalg.norm(err[:3]) < 1e-3

# reset and let dynamics settle
mujoco.mj_resetData(model, data)
for _ in range(200):
    mujoco.mj_step(model, data)

box1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box1_joint")
box2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box2_joint")

# set up box start positions
qpos_box1_adr = model.jnt_qposadr[box1_id]
data.qpos[qpos_box1_adr : qpos_box1_adr + 7] = [-0.2, 0.6, 0.825, 1, 0, 0, 0]
qpos_box2_adr = model.jnt_qposadr[box2_id]
data.qpos[qpos_box2_adr : qpos_box2_adr + 7] = [-0.4, 0.6, 0.825, 1, 0, 0, 0]

# reset fingers
l_finger_j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint")
r_finger_j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger_joint")
data.qpos[model.jnt_qposadr[l_finger_j_id]] = 0.0
data.qpos[model.jnt_qposadr[r_finger_j_id]] = 0.0

# preset elbow-up joint state to avoid singularity
seed = [1.9890, 1.0226, 0.8796, 1.2394, 0.0000, -1.1526]
for i in range(6):
    joint_name = f"joint{i+1}"
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    q_adr = model.jnt_qposadr[j_id]
    data.qpos[q_adr] = seed[i]

# solve ik for starting end-effector pos
target_start_pos = np.array([-0.2, 0.6, 0.98])
solve_ik(model, data, target_start_pos, site_name="eef")

# sync actuator targets
for i in range(6):
    joint_name = f"joint{i+1}"
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    data.ctrl[i] = data.qpos[model.jnt_qposadr[j_id]]
data.ctrl[6] = 0.0
data.ctrl[7] = 0.0
data.qvel[:] = 0.0

# let it settle
for _ in range(150):
    mujoco.mj_step(model, data)

ik_data = mujoco.MjData(model)
eef_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef")

# waypoint runner
def move_to_pose_diag(phase_name, target_pos, gripper_val, duration_steps=100):
    start_pos = np.copy(data.site_xpos[eef_site_id])
    start_gripper_l = data.ctrl[6]
    for step in range(duration_steps):
        t = step / duration_steps
        alpha = 3 * t**2 - 2 * t**3
        curr_pos = (1.0 - alpha) * start_pos + alpha * target_pos
        curr_gripper = (1.0 - alpha) * start_gripper_l + alpha * gripper_val
        
        ik_data.qpos[:] = data.qpos[:]
        solve_ik(model, ik_data, curr_pos, site_name="eef")
        for i in range(6):
            joint_name = f"joint{i+1}"
            data.ctrl[i] = ik_data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]]
        data.ctrl[6] = curr_gripper
        data.ctrl[7] = curr_gripper
        mujoco.mj_step(model, data)
        
        eef_pos = data.site_xpos[eef_site_id]
        box_pos = data.body("box1").xpos
        l_finger_pos = data.qpos[model.jnt_qposadr[l_finger_j_id]]
        r_finger_pos = data.qpos[model.jnt_qposadr[r_finger_j_id]]
        
        # print state diagnostics every 20 steps
        if step % 20 == 0 or step == duration_steps - 1:
            contacts_info = []
            for c_idx in range(data.ncon):
                con = data.contact[c_idx]
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
                c_force = np.zeros(6)
                mujoco.mj_contactForce(model, data, c_idx, c_force)
                force_norm = np.linalg.norm(c_force[:3])
                contacts_info.append(f"({g1} <-> {g2}: {force_norm:.2f}N)")
            print(f"[{phase_name} Step {step}] EEF: {eef_pos.round(3)}, Box: {box_pos.round(3)}, Contacts: {', '.join(contacts_info)}, Fingers: L={l_finger_pos:.4f} R={r_finger_pos:.4f}")

print("--- Start Diagnostic Run ---")
move_to_pose_diag("Approach", np.array([-0.2, 0.6, 0.98]), 0.0, 100)
move_to_pose_diag("Lower", np.array([-0.2, 0.6, 0.855]), 0.0, 100)
move_to_pose_diag("Grasp", np.array([-0.2, 0.6, 0.855]), 0.024, 100)
move_to_pose_diag("Lift", np.array([-0.2, 0.6, 1.00]), 0.024, 150)
move_to_pose_diag("Forward", np.array([-0.2, 0.42, 1.18]), 0.024, 300)
move_to_pose_diag("Traverse", np.array([0.2, 0.42, 1.18]), 0.024, 400)
move_to_pose_diag("MoveBack", np.array([0.2, 0.6, 1.00]), 0.024, 200)
move_to_pose_diag("LowerPlace", np.array([0.2, 0.6, 0.855]), 0.024, 150)
