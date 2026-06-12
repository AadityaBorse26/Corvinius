import os
import mujoco
import numpy as np
import importlib.util

# load model xml directly
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# let things settle
for _ in range(200):
    mujoco.mj_step(model, data)

target_pos = np.array([-0.1, 0.6, 0.95])

# import solve_ik dynamically from simulation.py
spec = importlib.util.spec_from_file_location("simulation", os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation.py"))
sim_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_mod)
solve_ik = sim_mod.solve_ik

# move arm towards target position
start_pos = np.copy(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef")])
for step in range(80):
    alpha = step / 80
    curr_pos = (1.0 - alpha) * start_pos + alpha * target_pos
    solve_ik(model, data, curr_pos, site_name="eef")
    
    # set controls to joints
    for i in range(6):
        joint_name = f"joint{i+1}"
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.ctrl[i] = data.qpos[model.jnt_qposadr[j_id]]
    data.ctrl[6] = 0
    data.ctrl[7] = 0
    mujoco.mj_step(model, data)

# print current contact info
print(f"Time: {data.time:.4f}")
print(f"Number of contacts: {data.ncon}")
for i in range(data.ncon):
    con = data.contact[i]
    c_force = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, c_force)
    geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
    geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
    force_norm = np.linalg.norm(c_force[:3])
    print(f"Contact {i}: '{geom1_name}' <-> '{geom2_name}', Force: {force_norm:.2f} N, Position: {con.pos}")
