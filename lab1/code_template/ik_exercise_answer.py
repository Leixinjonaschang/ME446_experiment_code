import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class PandaIKSolver:
    def __init__(self, model, data, ee_name="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        
        # get the ids of the revolute joints
        self.joint_ids = [model.joint(i).id for i in range(model.njnt) 
                          if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        self.n_dof = len(self.joint_ids)
        
        # get the joint limits for the scipy optimization constraints
        self.q_min = model.jnt_range[self.joint_ids, 0]
        self.q_max = model.jnt_range[self.joint_ids, 1]

    def _get_ee_pose(self, q):
        """
        Forward Kinematics: Calculate EE pose for a given q.
        Note: We must not mess up the main simulation state, so we save/restore or just use the current state 
        strictly for calculation. Here we modify data directly but effectively 'reset' it by the solver loop.
        """
        self.data.qpos[self.joint_ids] = q
        mujoco.mj_kinematics(self.model, self.data)
        
        ee_pos = self.data.xpos[self.ee_id].copy()
        ee_quat = self.data.xquat[self.ee_id].copy() # MuJoCo format: [w, x, y, z]
        return ee_pos, ee_quat

    def _loss_function(self, q, target_pos, target_quat_scipy):
        """
        The Objective Function (The part students should focus on).
        Returns a residual vector (not scalar sum of squares).
        """

        # 1. Get current end effector pose
        curr_pos, curr_quat_mj = self._get_ee_pose(q)
        
        # 2. Position Error (Vector difference)
        pos_err = curr_pos - target_pos
        
        # 3. Orientation Error
        # Convert MuJoCo [w, x, y, z] to Scipy [x, y, z, w]
        curr_quat_scipy = np.array([curr_quat_mj[1], curr_quat_mj[2], curr_quat_mj[3], curr_quat_mj[0]])
        
        # Calculate rotation difference
        rot_curr = R.from_quat(curr_quat_scipy)
        rot_target = R.from_quat(target_quat_scipy)
        
        # Diff rotation: R_diff = R_target * R_curr^T
        # We want the magnitude of the rotation vector of R_diff to be 0
        rot_diff = rot_target * rot_curr.inv()
        ori_err = rot_diff.as_rotvec()
        
        # Weight orientation higher to ensure alignment
        ori_weight = 0.5 
        
        # Combine errors into a single vector
        return np.concatenate([pos_err, ori_err * ori_weight])

    def solve(self, target_pos, target_quat, q_guess):
        """
        Uses Scipy Least Squares to solve IK.
        target_quat: [x, y, z, w] (Scipy format)
        """
        # Scipy least_squares requires the residual function
        result = least_squares(
            self._loss_function, 
            q_guess, 
            bounds=(self.q_min, self.q_max),
            args=(target_pos, target_quat),
            ftol=1e-4, # Precision tolerance
            verbose=0
        )
        return result.x

def draw_target_pose(viewer, pos, quat, alpha=0.5):
    """
    Draws a target pose (sphere and axes) in the MuJoCo viewer using the user_scn.
    """
    # 1. Draw Sphere at target position
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0, 0],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=[1, 0, 0, alpha]
    )
    viewer.user_scn.ngeom += 1
    
    # 2. Draw coordinate axes
    mat = R.from_quat(quat).as_matrix()
    axis_width = 0.005
    axis_scale = 0.1
    for i in range(3):
        rgba = [0, 0, 0, alpha]
        rgba[i] = 1.0
        
        # Calculate arrow direction (Z-axis of the arrow geom)
        z_axis = mat[:, i]
        if abs(z_axis[0]) < 0.9:
            x_axis = np.cross([1, 0, 0], z_axis)
        else:
            x_axis = np.cross([0, 1, 0], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        arrow_rot = np.column_stack([x_axis, y_axis, z_axis]).flatten()

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[axis_width, axis_width, axis_scale],
            pos=pos,
            mat=arrow_rot,
            rgba=rgba
        )
        viewer.user_scn.ngeom += 1

def generate_trajectory(q_start, q_end, steps=100):
    """Simple Linear Interpolation in Joint Space"""
    traj = []
    for i in range(steps):
        alpha = i / (steps - 1)
        q = (1 - alpha) * q_start + alpha * q_end
        traj.append(q)
    return traj

def main():
    # 1. Load Model
    model_path = "lab1/asset/franka_emika_panda/scene.xml" # replace with your actual path
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    ik_solver = PandaIKSolver(model, data, ee_name="hand")
    
    # Initialize Simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial State
        q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]) # Panda Home
        data.qpos[ik_solver.joint_ids] = q_home
        mujoco.mj_step(model, data)
        viewer.sync()

        print("System Ready. Calculating IK...")

        # --- Define Target Pose ---
        # Let's pick a target: 0.5m forward, 0.2m up, pointing down
        target_pos = np.array([0.5, 0.0, 0.8]) 
        # Target orientation: Pointing down (x-axis forward, z-axis down)
        # Quat [x, y, z, w] for 180 deg rotation around X-axis
        target_quat = R.from_euler('x', 180, degrees=True).as_quat()

        # --- Solve IK ---
        start_time = time.time()
        q_target = ik_solver.solve(target_pos, target_quat, q_guess=q_home)
        print(f"IK Solved in {time.time() - start_time:.4f}s")
        print(f"Target Joints: {q_target}")

        # --- Generate Trajectory (Interpolation) ---
        traj = generate_trajectory(q_home, q_target, steps=300)

        # --- Execute Animation ---
        print("Executing Motion...")
        for q in traj:
            # Set joint positions directly (Kinematic Animation)
            data.qpos[ik_solver.joint_ids] = q
            
            # Forward Kinematics to update visualization
            mujoco.mj_kinematics(model, data) 
            
            # --- Visualize Target ---
            viewer.user_scn.ngeom = 0
            draw_target_pose(viewer, target_pos, target_quat)
            
            viewer.sync()
            time.sleep(0.005) # Control playback speed

        print("Motion Complete.")
        
        # Keep viewer open
        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            draw_target_pose(viewer, target_pos, target_quat)
            viewer.sync()
            time.sleep(0.1)

if __name__ == "__main__":
    main()