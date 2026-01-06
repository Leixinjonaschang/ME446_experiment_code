import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class PandaIKSolver:
    """Inverse Kinematics solver for Franka Emika Panda using least_squares."""
    def __init__(self, model, data, ee_name="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        # Identify hinge joints for the 7-DOF arm
        self.joint_ids = [model.joint(i).id for i in range(model.njnt) 
                          if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        self.q_min = model.jnt_range[self.joint_ids, 0]
        self.q_max = model.jnt_range[self.joint_ids, 1]

    def _get_ee_pose(self, q):
        """Computes end-effector pose for a given joint configuration."""
        self.data.qpos[self.joint_ids] = q
        mujoco.mj_kinematics(self.model, self.data)
        ee_pos = self.data.xpos[self.ee_id].copy()
        ee_quat = self.data.xquat[self.ee_id].copy()
        return ee_pos, ee_quat

    def _loss_function(self, q, target_pos, target_quat_scipy):
        """Calculates 6D pose error (position + orientation) for optimization."""
        curr_pos, curr_quat_mj = self._get_ee_pose(q)
        
        # Position error
        pos_err = curr_pos - target_pos
        
        # Orientation error using rotation vector
        # MJ quat: [w, x, y, z] -> Scipy quat: [x, y, z, w]
        curr_quat_scipy = np.array([curr_quat_mj[1], curr_quat_mj[2], curr_quat_mj[3], curr_quat_mj[0]])
        rot_curr = R.from_quat(curr_quat_scipy)
        rot_target = R.from_quat(target_quat_scipy)
        
        rot_diff = rot_target * rot_curr.inv()
        ori_err = rot_diff.as_rotvec()
        
        # Weight orientation error slightly less for stability
        return np.concatenate([pos_err, ori_err * 0.5])

    def solve(self, target_pos, target_quat_scipy, q_guess):
        """Solves IK for a target pose starting from q_guess."""
        result = least_squares(
            self._loss_function, 
            q_guess, 
            bounds=(self.q_min, self.q_max),
            args=(target_pos, target_quat_scipy), 
        )
        return result.x

def compute_joint_pd_control(data, q_target, kp, kd):
    """Computes PD control torque in joint space."""
    q_curr = data.qpos[:7]
    v_curr = data.qvel[:7]
    
    q_err = q_target - q_curr
    v_err = 0 - v_curr # Target velocity is zero
    
    torque = kp * q_err + kd * v_err
    
    # Clip to Panda's approximate torque limits
    return np.clip(torque, -87, 87), q_err

def draw_frame(viewer, pos, quat, alpha=1.0):
    """Adds a coordinate frame visualization to the MuJoCo viewer."""
    # Origin sphere
    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], 
                        pos=pos, mat=np.eye(3).flatten(), rgba=[1, 1, 1, alpha])
    viewer.user_scn.ngeom += 1
    
    # Orientation axes
    quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
    mat = R.from_quat(quat_scipy).as_matrix()
    
    axis_width, axis_scale = 0.005, 0.3
    for i in range(3):
        rgba = [0, 0, 0, alpha]
        rgba[i] = 1.0 # R, G, B for X, Y, Z
        z_axis = mat[:, i]
        
        # Simple orthographic projection for arrow direction
        if abs(z_axis[0]) < 0.9: x_axis = np.cross([1, 0, 0], z_axis)
        else: x_axis = np.cross([0, 1, 0], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        arrow_rot = np.column_stack([x_axis, y_axis, z_axis]).flatten()

        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW, size=[axis_width, axis_width, axis_scale],
            pos=pos, mat=arrow_rot, rgba=rgba)
        viewer.user_scn.ngeom += 1

def main():
    model = mujoco.MjModel.from_xml_path('lab2/asset/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)
    
    # Actuator bypass for direct torque control
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0  
        model.actuator_biasprm[i, 0] = 0.0  
        model.actuator_ctrllimited[i] = 0 

    # Define target EE pose
    target_pos = np.array([0.5, 0.3, 0.5]) 
    target_quat_scipy = R.from_euler('x', 135, degrees=True).as_quat()
    target_quat_mj = np.array([target_quat_scipy[3], target_quat_scipy[0], target_quat_scipy[1], target_quat_scipy[2]])

    # Solve IK offline
    print("Computing static IK target...")
    ik_solver = PandaIKSolver(model, data)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    
    # Add noise to start guess to avoid local minima
    q_guess = data.qpos[:7].copy() + np.random.randn(7) * 0.05
    q_target = ik_solver.solve(target_pos, target_quat_scipy, q_guess)
    print("IK target found.")

    # Controller Gains
    kp = np.array([800, 700, 600, 500, 250, 150, 50]) 
    kd = np.array([50, 50, 50, 50, 30, 10, 5])

    # Simulation setup
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    time_history, pos_err_history, ori_err_history = [], [], []
    start_time = data.time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop_start = time.time()
            
            # Visualization
            viewer.user_scn.ngeom = 0
            draw_frame(viewer, target_pos, target_quat_mj, alpha=0.3) # Target
            ee_pos = data.xpos[ik_solver.ee_id].copy()
            ee_quat = data.xquat[ik_solver.ee_id].copy()
            draw_frame(viewer, ee_pos, ee_quat, alpha=1.0) # Actual

            # Control step
            torque, _ = compute_joint_pd_control(data, q_target, kp, kd)
            data.ctrl[:7] = torque
            mujoco.mj_step(model, data)

            # Error logging
            pos_err = np.linalg.norm(target_pos - ee_pos)
            
            curr_quat_scipy = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
            rot_curr, rot_target = R.from_quat(curr_quat_scipy), R.from_quat(target_quat_scipy)
            ori_err = np.linalg.norm((rot_target * rot_curr.inv()).as_rotvec())

            time_history.append(data.time - start_time)
            pos_err_history.append(pos_err)
            ori_err_history.append(ori_err)
            
            viewer.sync()
            
            # Timing control
            elapsed = time.time() - loop_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
            
            if data.time - start_time > 5.0:
                break

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_history, pos_err_history, 'r', label='Position Error')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(time_history, ori_err_history, 'b', label='Orientation Error')
    plt.ylabel('Error (rad)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
