import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class PandaIKSolver:
    def __init__(self, model, data, ee_name="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        self.joint_ids = [model.joint(i).id for i in range(model.njnt) 
                          if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        self.q_min = model.jnt_range[self.joint_ids, 0]
        self.q_max = model.jnt_range[self.joint_ids, 1]

    def _get_ee_pose(self, q):
        # temporarily set the state to q to calculate forward kinematics
        original_qpos = self.data.qpos[self.joint_ids].copy()
        self.data.qpos[self.joint_ids] = q
        mujoco.mj_kinematics(self.model, self.data)
        ee_pos = self.data.xpos[self.ee_id].copy()
        ee_quat = self.data.xquat[self.ee_id].copy() # [w, x, y, z]
        # restore the original state
        self.data.qpos[self.joint_ids] = original_qpos
        mujoco.mj_kinematics(self.model, self.data)
        return ee_pos, ee_quat

    def _loss_function(self, q, target_pos, target_quat_scipy):
        curr_pos, curr_quat_mj = self._get_ee_pose(q)
        pos_err = curr_pos - target_pos
        curr_quat_scipy = np.array([curr_quat_mj[1], curr_quat_mj[2], curr_quat_mj[3], curr_quat_mj[0]])
        rot_curr = R.from_quat(curr_quat_scipy)
        rot_target = R.from_quat(target_quat_scipy)
        rot_diff = rot_target * rot_curr.inv()
        ori_err = rot_diff.as_rotvec()
        return np.concatenate([pos_err, ori_err * 0.5])

    def solve(self, target_pos, target_quat, q_guess):
        result = least_squares(
            self._loss_function, q_guess, bounds=(self.q_min, self.q_max),
            args=(target_pos, target_quat), ftol=1e-3, xtol=1e-3, gtol=1e-3, verbose=0
        )
        return result.x

def pd_controller(data, q_target, kp, kd):
    q_current = data.qpos[:7]
    v_current = data.qvel[:7]
    error = q_target - q_current
    error_dot = 0 - v_current # 假设目标关节速度为0
    
    # PD 控制项
    torque = kp * error + kd * error_dot
    torque = np.clip(torque, -87, 87)
    
    # 重力补偿 (Gravity Compensation)
    # data.qfrc_bias 包含重力、科里奥利力和离心力
    # torque_gravity = data.qfrc_bias[:7]
    # torque = torque + torque_gravity
    
    return torque, error

def main():
    model = mujoco.MjModel.from_xml_path('lab2/asset/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)
    
    # initialize the robot to home position
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    
    # set the actuator to torque mode
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0  
        model.actuator_biasprm[i, 0] = 0.0  
        model.actuator_biasprm[i, 1] = 0.0  
        model.actuator_biasprm[i, 2] = 0.0 
        model.actuator_ctrllimited[i] = 0 

    ik_solver = PandaIKSolver(model, data)
    
    # control parameters
    kp = np.array([1000, 1000, 800, 800, 250, 150, 50]) 
    kd = np.array([50, 50, 50, 50, 30, 10, 5])
    
    # trajectory definition: circle on the XY plane
    center = np.array([0.5, 0.0, 0.5])
    radius = 0.1
    freq = 0.5 # Hz
    # target orientation: end-effector points downward
    target_quat = R.from_euler('x', 180, degrees=True).as_quat()

    # record data
    time_history = []
    ee_error_history = []
    ee_pos_history = []  # record the actual EE trajectory points
    q_target_curr = data.qpos[:7].copy()

    # visualization tools
    def draw_points(viewer, pos, rgba, radius=0.005):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
        viewer.user_scn.ngeom += 1
        
    def draw_line(viewer, points, rgba, width=0.002):
        if len(points) < 2: return
        # simplify sampling: draw a line every 5 points to avoid too many geometries
        for i in range(0, len(points) - 1, 5):
            if i + 5 >= len(points): break
            p1 = points[i]
            p2 = points[i+5]
            
            # use cylinder to simulate line
            diff = p2 - p1
            dist = np.linalg.norm(diff)
            if dist < 1e-4: continue
            
            z_axis = diff / dist
            if abs(z_axis[0]) < 0.9:
                x_axis = np.cross([1, 0, 0], z_axis)
            else:
                x_axis = np.cross([0, 1, 0], z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            mat = np.column_stack([x_axis, y_axis, z_axis]).flatten()
            
            # draw the cylinder
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[width, width, dist/2],
                pos=(p1 + p2)/2,
                mat=mat,
                rgba=rgba
            )
            viewer.user_scn.ngeom += 1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = data.time
        while viewer.is_running():
            step_start = time.time()
            sim_time = data.time - start_time
            
            # clear visualization geometries
            viewer.user_scn.ngeom = 0
            
            # 1. task space trajectory calculation (points on the discrete circle)
            target_pos = center + np.array([
                radius * np.cos(2 * np.pi * freq * sim_time),
                radius * np.sin(2 * np.pi * freq * sim_time),
                0
            ])

            # --- visualization: draw the target circle trajectory (static dashed line) ---
            for theta in np.linspace(0, 2*np.pi, 50):
                circle_pt = center + np.array([radius*np.cos(theta), radius*np.sin(theta), 0])
                draw_points(viewer, circle_pt, rgba=[0, 1, 0, 0.3], radius=0.002) # green dashed line

            # --- visualization: draw the current target point ---
            draw_points(viewer, target_pos, rgba=[1, 0, 0, 1.0], radius=0.01) # red solid point

            # 2. numerical IK mapping (solve every certain time, or every step)
            # use the previous q_target_curr as Guess to ensure continuity
            if int(data.time * 500) % 5 == 0: # 100Hz is enough to solve IK
                q_target_curr = ik_solver.solve(target_pos, target_quat, q_target_curr)
            
            # 3. joint space feedback control (PD)
            # feedback only occurs in joint space e = q_target_curr - q_curr
            torque, _ = pd_controller(data, q_target_curr, kp, kd)
            
            # execute the control
            data.ctrl[:7] = torque
            mujoco.mj_step(model, data)
            
            # 4. record task space EE Error for analysis
            ee_pos_actual = data.xpos[ik_solver.ee_id].copy()
            ee_error = np.linalg.norm(target_pos - ee_pos_actual)
            
            ee_pos_history.append(ee_pos_actual)
            if len(ee_pos_history) > 5000: # limit the history length to prevent memory/rendering explosion
                 ee_pos_history.pop(0)

            # --- visualization: draw the actual EE trajectory (blue solid line) ---
            draw_line(viewer, ee_pos_history, rgba=[0, 0, 1, 1.0], width=0.002)

            time_history.append(data.time)
            ee_error_history.append(ee_error)
            
            viewer.sync()
            
            # control frequency compensation
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
            
            if sim_time > 10.0: break

    # plot task space error
    plt.figure(figsize=(10, 5))
    plt.plot(time_history, ee_error_history, label='End-Effector Euclidean Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Task-Space Tracking Error (Joint-Space PD + Numerical IK)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()