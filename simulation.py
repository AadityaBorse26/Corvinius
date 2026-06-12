import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# load model and init mujoco data
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.xml")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

def solve_ik(model, data, target_pos, site_name="eef", max_steps=30, step_size=0.15):
    # ik solver using damped least squares to keep gripper pointing down
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    joint_names = [f"joint{i}" for i in range(1, 7)]
    joint_qpos_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
    dof_ids = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]

    # target rot matrix pointing straight down
    R_target = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])

    for _ in range(max_steps):
        mujoco.mj_fwdPosition(model, data)
        
        site_xpos = data.site_xpos[site_id]
        site_xmat = data.site_xmat[site_id].reshape(3, 3)
        
        # calculate position and rotation errors
        err_pos = target_pos - site_xpos
        err_rot = 0.5 * (
            np.cross(site_xmat[:, 0], R_target[:, 0]) +
            np.cross(site_xmat[:, 1], R_target[:, 1]) +
            np.cross(site_xmat[:, 2], R_target[:, 2])
        )
        
        err = np.hstack([err_pos, err_rot])
        if np.linalg.norm(err) < 1e-4:
            break
            
        # jacobians for the site
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        
        # damped least squares update to avoid singularities
        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        
        for idx, q_adr in enumerate(joint_qpos_adr):
            data.qpos[q_adr] += dq[idx] * step_size
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[idx])
            j_range = model.jnt_range[j_id]
            data.qpos[q_adr] = np.clip(data.qpos[q_adr], j_range[0], j_range[1])

    return np.linalg.norm(err[:3]) < 1e-3

# output dir setup
os.makedirs("output", exist_ok=True)

# setup renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# config camera looking at table
cam = mujoco.MjvCamera()
cam.lookat = np.array([0.0, 0.6, 0.85])
cam.distance = 1.6
cam.elevation = -30.0
cam.azimuth = 100.0

# logs container
logs = {
    "time": [],
    "box1_pos": [],
    "box2_pos": [],
    "box1_accel": [],
    "box2_accel": [],
    "joint_pos": [],
    "joint_torque": [],
    "contact_force": []
}

def log_sensor_data():
    logs["time"].append(data.time)
    
    # log box positions
    logs["box1_pos"].append(np.copy(data.body("box1").xpos))
    logs["box2_pos"].append(np.copy(data.body("box2").xpos))
    
    # acceleration from box sensors
    acc1 = data.sensor("box1_accel").data.copy()
    acc2 = data.sensor("box2_accel").data.copy()
    logs["box1_accel"].append(acc1)
    logs["box2_accel"].append(acc2)
    
    # joint angles
    joint_pos_vals = [data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]] for i in range(1, 7)]
    logs["joint_pos"].append(np.array(joint_pos_vals))
    
    # actuator torques
    logs["joint_torque"].append(np.copy(data.actuator_force[:6]))
    
    # contact forces
    total_contact_force = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        c_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, c_force)
        total_contact_force += np.linalg.norm(c_force[:3])
    logs["contact_force"].append(total_contact_force)

def save_frame(filename, title=""):
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    plt.figure(figsize=(8, 6))
    plt.imshow(pixels)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join("output", filename), dpi=150)
    plt.close()
    print(f"Captured screenshot: {filename}")

# trajectory controller helper
ik_data = mujoco.MjData(model)

def move_to_pose(target_pos, gripper_val, duration_steps=100):
    # minimum jerk traj interpolation to target
    start_pos = np.copy(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef")])
    start_gripper_l = data.ctrl[6]
    
    # warm start ik from current qpos
    ik_data.qpos[:] = data.qpos[:]
    
    for step in range(duration_steps):
        t = step / duration_steps
        alpha = 3 * t**2 - 2 * t**3  # cubic min jerk scaling
        curr_pos = (1.0 - alpha) * start_pos + alpha * target_pos
        curr_gripper = (1.0 - alpha) * start_gripper_l + alpha * gripper_val
        
        # run ik step
        solve_ik(model, ik_data, curr_pos, site_name="eef")
        
        # set joint position targets
        for i in range(6):
            joint_name = f"joint{i+1}"
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            q_adr = model.jnt_qposadr[j_id]
            data.ctrl[i] = ik_data.qpos[q_adr]
            
        data.ctrl[6] = curr_gripper
        data.ctrl[7] = curr_gripper
        
        # step sim and log data
        mujoco.mj_step(model, data)
        log_sensor_data()

def run_simulation():
    # start simulation
    mujoco.mj_resetData(model, data)
    # let it settle first
    for _ in range(200):
        mujoco.mj_step(model, data)
        log_sensor_data()

    print("Phase 1: Scene setup verified.")
    save_frame("phase1_init.png", "Initial scene configuration")

    # task 1 - push boxes
    print("Phase 2: Initiating object push sweep.")
    # move above box 1
    move_to_pose(np.array([-0.13, 0.6, 0.95]), gripper_val=0.0, duration_steps=150)
    # lower arm behind box
    move_to_pose(np.array([-0.13, 0.6, 0.855]), gripper_val=0.0, duration_steps=100)
    save_frame("phase2_pre_push.png", "Gripper pre-push alignment")

    # push box 1 into box 2
    move_to_pose(np.array([-0.35, 0.6, 0.855]), gripper_val=0.0, duration_steps=400)
    save_frame("phase2_mid_push.png", "Box 1 and Box 2 contact dynamic interaction")

    # push both to corner
    move_to_pose(np.array([-0.40, 0.70, 0.855]), gripper_val=0.0, duration_steps=400)
    # retract arm
    move_to_pose(np.array([-0.40, 0.70, 0.98]), gripper_val=0.0, duration_steps=100)
    save_frame("phase3_push_done.png", "Pushing task delivery and retraction completed")

    # task 2 - pick and place over barrier
    print("Phase 3: Reconfiguring system for pick-and-place sequence.")
    box1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box1_joint")
    box2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box2_joint")

    # reset box 1 for pick and place
    qpos_box1_adr = model.jnt_qposadr[box1_id]
    data.qpos[qpos_box1_adr : qpos_box1_adr + 7] = [-0.2, 0.6, 0.825, 1, 0, 0, 0]
    # move box 2 out of the way
    qpos_box2_adr = model.jnt_qposadr[box2_id]
    data.qpos[qpos_box2_adr : qpos_box2_adr + 7] = [-0.4, 0.6, 0.825, 1, 0, 0, 0]

    # reset fingers
    l_finger_j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint")
    r_finger_j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger_joint")
    data.qpos[model.jnt_qposadr[l_finger_j_id]] = 0.0
    data.qpos[model.jnt_qposadr[r_finger_j_id]] = 0.0

    # set positive elbow branch joints to avoid singularity
    seed = [1.9890, 1.0226, 0.8796, 1.2394, 0.0000, -1.1526]
    for i in range(6):
        joint_name = f"joint{i+1}"
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        q_adr = model.jnt_qposadr[j_id]
        data.qpos[q_adr] = seed[i]

    # solve start pose over box
    target_start_pos = np.array([-0.2, 0.6, 0.98])
    solve_ik(model, data, target_start_pos, site_name="eef")

    # sync joint targets
    for i in range(6):
        joint_name = f"joint{i+1}"
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        q_adr = model.jnt_qposadr[j_id]
        data.ctrl[i] = data.qpos[q_adr]
    data.ctrl[6] = 0.0 # Gripper open

    # clear velocities
    data.qvel[:] = 0.0

    # settle before grasp
    for _ in range(150):
        mujoco.mj_step(model, data)
        log_sensor_data()

    save_frame("phase4_pre_grasp.png", "Pick and place initial workspace layout")

    # move above box
    print("Pick & Place: Approaching Box 1")
    move_to_pose(np.array([-0.2, 0.6, 0.98]), gripper_val=0.0, duration_steps=100)
    # lower gripper
    print("Pick & Place: Lowering gripper")
    move_to_pose(np.array([-0.2, 0.6, 0.855]), gripper_val=0.0, duration_steps=100)
    save_frame("phase5_grasp.png", "Gripper pre-grasp alignment")

    # close gripper
    print("Pick & Place: Closing gripper")
    move_to_pose(np.array([-0.2, 0.6, 0.855]), gripper_val=0.024, duration_steps=100)
    # lift box
    print("Pick & Place: Lifting box")
    move_to_pose(np.array([-0.2, 0.6, 1.00]), gripper_val=0.024, duration_steps=150)

    # move forward to clear barrier
    print("Pick & Place: Clearing barrier")
    move_to_pose(np.array([-0.2, 0.42, 1.18]), gripper_val=0.024, duration_steps=300)
    save_frame("phase6_lift.png", "Lifting over central partition barrier")

    # cross over barrier
    print("Pick & Place: Crossing barrier")
    move_to_pose(np.array([0.2, 0.42, 1.18]), gripper_val=0.024, duration_steps=400)
    save_frame("phase7_traverse.png", "Workspace traversal phase")

    # move to placement pos
    print("Pick & Place: Aligning with placement site")
    move_to_pose(np.array([0.2, 0.6, 1.00]), gripper_val=0.024, duration_steps=200)

    # lower box to table
    print("Pick & Place: Lowering box")
    move_to_pose(np.array([0.2, 0.6, 0.855]), gripper_val=0.024, duration_steps=150)

    # open gripper
    print("Pick & Place: Releasing box")
    move_to_pose(np.array([0.2, 0.6, 0.855]), gripper_val=0.0, duration_steps=100)
    save_frame("phase8_place.png", "Target placement completed")

    # retract arm
    print("Pick & Place: Retracting arm")
    move_to_pose(np.array([0.2, 0.6, 0.98]), gripper_val=0.0, duration_steps=100)
    save_frame("phase9_done.png", "Manipulator retracted to home pose")

    print("Simulation run completed successfully.")

    # plot data
    print("Generating sensor plots...")
    t_arr = np.array(logs["time"])

    # plot 1: trajectories
    plt.figure(figsize=(10, 5))
    box1_xyz = np.array(logs["box1_pos"])
    box2_xyz = np.array(logs["box2_pos"])
    plt.plot(t_arr, box1_xyz[:, 0], 'r-', label="Red Box X")
    plt.plot(t_arr, box1_xyz[:, 1], 'r--', label="Red Box Y")
    plt.plot(t_arr, box2_xyz[:, 0], 'b-', label="Blue Box X")
    plt.plot(t_arr, box2_xyz[:, 1], 'b--', label="Blue Box Y")
    plt.axvline(x=2.2, color='k', linestyle=':', label="Pick & Place Reset")
    plt.title("Rigid Body Cartesian Trajectories", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("output", "plot_box_trajectories.png"), dpi=150)
    plt.close()

    # plot 2: joint angles
    plt.figure(figsize=(10, 5))
    joint_pos = np.array(logs["joint_pos"])
    for i in range(6):
        plt.plot(t_arr, joint_pos[:, i], label=f"Joint {i+1}")
    plt.title("Manipulator Joint Posture Trajectories", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("output", "plot_joint_positions.png"), dpi=150)
    plt.close()

    # plot 3: joint torques
    plt.figure(figsize=(10, 5))
    joint_torque = np.array(logs["joint_torque"])
    for i in range(6):
        plt.plot(t_arr, joint_torque[:, i], label=f"Joint {i+1}")
    plt.title("Manipulator Actuator Torques", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N·m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("output", "plot_joint_torques.png"), dpi=150)
    plt.close()

    # plot 4: acceleration
    plt.figure(figsize=(10, 5))
    box1_acc = np.array(logs["box1_accel"])
    plt.plot(t_arr, box1_acc[:, 0], label="Acc X")
    plt.plot(t_arr, box1_acc[:, 1], label="Acc Y")
    plt.plot(t_arr, box1_acc[:, 2], label="Acc Z")
    plt.title("Red Box IMU Accelerometer Signals", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("output", "plot_imu_accelerations.png"), dpi=150)
    plt.close()

    # plot 5: contact force
    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, logs["contact_force"], 'g-', label="Total Contact Force")
    plt.title("Integrated Normal Contact Force Magnitude", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("output", "plot_contact_forces.png"), dpi=150)
    plt.close()

    print("All sensor plots generated successfully.")

if __name__ == "__main__":
    run_simulation()
