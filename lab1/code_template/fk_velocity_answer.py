#!/usr/bin/env python3
"""
Lab 1: Forward Velocity Kinematics and Visualization

Task:
Compute the end-effector velocity (spatial twist) given joint angles and velocities.
Visualize the resulting linear velocity vector and the end-effector trajectory.
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt


class FKVelocityExercise:
    def __init__(self, model_xml, target_link_name="link7"):
        """
        Initializes the velocity exercise class.
        """
        self.model = model_xml
        self.data = mujoco.MjData(self.model)
        
        # Get joint IDs
        self.joint_ids = [self.model.joint(i).id for i in range(self.model.njnt) if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        self.num_dof = len(self.joint_ids)
        
        # Target Body
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_link_name)
        if self.end_effector_id == -1:
            raise ValueError(f"Body {target_link_name} not found")
        
        # Extract PoE initial parameters
        self.M, self.S_list = self._extract_kinematic_params()
        
        # Trajectory history for visualization
        self.trajectory_points = []
        self.max_trajectory_len = 200
        
        
        # Frame snapshots for visualization
        self.frame_snapshots = []
        
        # Error logging
        self.log_t = []
        self.log_lin_err = []
        self.log_ang_err = []

    def _skew(self, v):
        """Computes the skew-symmetric matrix for a 3D vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _vec_to_se3(self, S):
        """Converts a 6D spatial twist (omega, v) into a 4x4 se(3) matrix."""
        se3 = np.zeros((4, 4))
        se3[:3, :3] = self._skew(S[:3])
        se3[:3, 3] = S[3:]
        return se3
    
    def _adjoint(self, T):
        """Computes the Adjoint representation of a transformation matrix T."""
        R = T[:3, :3]
        p = T[:3, 3]
        AdT = np.zeros((6, 6))
        AdT[:3, :3] = R
        AdT[3:, 3:] = R
        AdT[3:, :3] = self._skew(p) @ R
        return AdT

    def _extract_kinematic_params(self):
        """Extracts M and S_list at q=0."""
        self.data.qpos[:] = 0
        mujoco.mj_forward(self.model, self.data)
        M = np.eye(4)
        M[:3, 3] = self.data.xipos[self.end_effector_id]
        M[:3, :3] = self.data.ximat[self.end_effector_id].reshape(3, 3)
        S_list = []
        for j_id in self.joint_ids:
            omega = self.data.xaxis[j_id]
            p = self.data.xanchor[j_id]
            v = -np.cross(omega, p)
            S_list.append(np.concatenate([omega, v]))
        return M, S_list

    def calculate_jacobian(self, q):
        """
        Computes the Space Jacobian J_s given joint angles q.
        J_s = [S1, Ad_T1(S2), ... , Ad_Tn-1(Sn)]
        """
        Js = np.zeros((6, self.num_dof))
        T = np.eye(4)
        
        # The first column is just S1
        Js[:, 0] = self.S_list[0]
        
        for i in range(1, self.num_dof):
            # Update transformation up to joint i-1
            # T_{i-1} = e^[S1]q1 * ... * e^[Si-1]qi-1
            T = T @ expm(self._vec_to_se3(self.S_list[i-1]) * q[i-1])
            
            # Apply Adjoint transformation to move S_i to the current configuration
            # J_si = Ad_{T_{i-1}}(S_i)
            Js[:, i] = self._adjoint(T) @ self.S_list[i]
            
        return Js

    def compute_ee_velocity(self, q, q_dot):
        """
        Computes the end-effector velocity using Space Jacobian.
        
        The forward velocity kinematics mapping is:
        V_s = J_s(q) * q_dot
        
        where:
        - V_s = [omega_s, v_s] is the spatial twist (6D)
        - J_s is the Space Jacobian (6 x n)
        - q_dot is the joint velocities (n x 1)
        
        The linear velocity of a point p on the body is:
        v_point = v_s + omega_s × p
        
        Returns:
            linear_vel_com: 3D linear velocity vector at COM (in world/base frame)
            omega_s: 3D angular velocity vector (in world/base frame)
            p_com: Current 3D position of COM (in world/base frame)
            spatial_twist: Full 6D spatial twist [omega_s, v_s] (in world/base frame)
            T: Current transformation matrix (4x4)
        """
        # 1. Calculate Space Jacobian J_s(q)
        # J_s is constructed column by column:
        # J_s = [S1, Ad_{T1}(S2), Ad_{T2}(S3), ..., Ad_{Tn-1}(Sn)]
        Js = self.calculate_jacobian(q)
        
        # 2. Forward Velocity Kinematics Mapping: V_s = J_s * q_dot
        # This is the core equation that maps joint velocities to spatial twist
        spatial_twist = Js @ q_dot
        
        # 3. Extract angular and linear velocity components
        # V_s = [omega_s, v_s] where:
        # - omega_s: angular velocity (3D)
        # - v_s: linear velocity of a point instantaneously at the origin (3D)
        omega_s = spatial_twist[:3]
        v_s = spatial_twist[3:]
        
        # 4. Calculate current end-effector pose T for COM position
        T = np.eye(4)
        for i in range(self.num_dof):
            T = T @ expm(self._vec_to_se3(self.S_list[i]) * q[i])
        T = T @ self.M
        p_com = T[:3, 3]
        
        # 5. Compute linear velocity at the COM point
        # The spatial velocity v_s is the velocity of a point at the origin.
        # For a point p on the body, we use: v_point = v_s + omega_s × p
        linear_vel_com = v_s + np.cross(omega_s, p_com)
        
        return linear_vel_com, omega_s, p_com, spatial_twist, T

    def get_mujoco_gt_velocity(self):
        """
        Gets the ground truth velocity using MuJoCo's Jacobian function.
        
        Note: data.cvel is NOT suitable for this comparison because it represents
        the spatial velocity of the kinematic SUBTREE's center of mass (including
        child bodies like fingers), not the individual body's CoM velocity.
        
        We use mj_jacBodyCom which gives the analytical Jacobian for the specific
        body's center of mass, providing: v_com = J_p @ qvel, omega = J_r @ qvel
        
        Returns:
            linear_vel_mj: 3D linear velocity of body COM in world frame
            omega_mj: 3D angular velocity in world frame
        """
        # Create Jacobian matrices
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        
        # Compute analytical Jacobian for body CoM
        mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, self.end_effector_id)
        
        # Compute velocities: v = J @ qvel
        linear_vel_mj = jacp @ self.data.qvel
        omega_mj = jacr @ self.data.qvel
        
        return linear_vel_mj, omega_mj

    def plot_errors(self):
        """Plots the recorded velocity errors."""
        if not self.log_t:
            print("No data recorded for plotting.")
            return

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot Linear Velocity Error
        ax[0].plot(self.log_t, self.log_lin_err, label='Linear Velocity Error (m/s)', color='r')
        ax[0].set_title('End-Effector Linear Velocity Error over Time')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Error (m/s)')
        ax[0].grid(True)
        ax[0].legend()
        
        # Plot Angular Velocity Error
        ax[1].plot(self.log_t, self.log_ang_err, label='Angular Velocity Error (rad/s)', color='b')
        ax[1].set_title('End-Effector Angular Velocity Error over Time')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Error (rad/s)')
        ax[1].grid(True)
        ax[1].legend()
        
        plt.tight_layout()
        plt.show()

    def _draw_arrow(self, viewer, pos, vec, rgba=[1, 0, 0, 1], scale=1.0):
        """Draws a velocity vector arrow."""
        if np.linalg.norm(vec) < 1e-6:
            return
            
        # Arrow direction
        z_axis = vec / np.linalg.norm(vec)
        
        # Construct rotation matrix to align Z with vec
        if abs(z_axis[0]) < 0.9:
            x_axis = np.cross([1, 0, 0], z_axis)
        else:
            x_axis = np.cross([0, 1, 0], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis]).flatten()
        
        # Draw arrow
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, np.linalg.norm(vec) * scale],
            pos=pos,
            mat=rot,
            rgba=rgba
        )
        viewer.user_scn.ngeom += 1

    def _draw_frame(self, viewer, T, scale=0.2, width=0.005):
        """Draws a coordinate frame (R-G-B axes) for transformation T."""
        p = T[:3, 3]
        R = T[:3, :3]
        
        # X axis (Red)
        self._draw_arrow(viewer, p, R[:, 0], rgba=[1, 0, 0, 1], scale=scale)
        # Y axis (Green)
        self._draw_arrow(viewer, p, R[:, 1], rgba=[0, 1, 0, 1], scale=scale)
        # Z axis (Blue)
        self._draw_arrow(viewer, p, R[:, 2], rgba=[0, 0, 1, 1], scale=scale)

    def run_sim_vis(self):
        """Main visualization loop."""
        print("\nVisualization Started.")
        print("Using Space Jacobian for Forward Velocity Kinematics: V_s = J_s(q) * q_dot")
        print("\nVisualization Legend:")
        print("  Cyan Trail: End-Effector Trajectory")
        print("  Red Arrow: Computed Linear Velocity (from Space Jacobian)")
        print("  Green Arrow: Computed Angular Velocity (scaled)")

        # 1. Initialize from Home keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            q_home = self.model.key_qpos[key_id][:self.num_dof].copy()
            print(f"Loaded 'home' keyframe configuration.")
        else:
            print("Warning: 'home' keyframe not found. Using q=0.")
            q_home = np.zeros(self.num_dof)
        
        # 2. Set small constant joint velocities (rad/s)
        # Each joint rotates at a different small speed for visual variety
        q_dot_const = np.array([0.1, 0.15, 0.12, 0.18, 0.14, 0.16, 0.13])[:self.num_dof]
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            start_time = time.time()
            last_snapshot_time = -1.0
            snapshot_interval = 0.2
            total_sim_time = 10.0
            
            while viewer.is_running():
                t = time.time() - start_time
                
                # Logic: Stop after 2 seconds
                if t > total_sim_time:
                    t_effective = total_sim_time
                    q_dot_curr = np.zeros(self.num_dof)
                else:
                    t_effective = t
                    q_dot_curr = q_dot_const.copy()

                # 3. Compute current joint angles: q = q_home + q_dot * t
                q_curr = q_home + q_dot_const * t_effective
                
                # Update simulator state for visualization
                self.data.qpos[:self.num_dof] = q_curr
                self.data.qvel[:self.num_dof] = q_dot_curr
                mujoco.mj_forward(self.model, self.data)
                
                # 2. Compute Velocity using Space Jacobian
                # V_s = J_s(q) * q_dot
                linear_vel, omega_s, p_com, spatial_twist, T_curr = self.compute_ee_velocity(q_curr, q_dot_curr)
                
                # 3. Get MuJoCo Ground Truth and Validate
                linear_vel_mj, omega_mj = self.get_mujoco_gt_velocity()
                
                linear_error = np.linalg.norm(linear_vel - linear_vel_mj)
                angular_error = np.linalg.norm(omega_s - omega_mj)
                
                # Print validation results periodically
                if int(t * 10) % 50 == 0:  # Print every 5 seconds
                    print(f"\n--- Velocity Validation (t={t:.2f}s) ---")
                    print(f"Computed Linear Vel: {linear_vel.round(4)}")
                    print(f"MuJoCo Linear Vel:    {linear_vel_mj.round(4)}")
                    print(f"Linear Velocity Error: {linear_error:.6e} m/s")
                    print(f"Angular Velocity Error: {angular_error:.6e} rad/s")
                
                # Log errors
                if t <= total_sim_time:
                    self.log_t.append(t)
                    self.log_lin_err.append(linear_error)
                    self.log_ang_err.append(angular_error)

                # Snapshot logic: capture every 0.2s while moving
                if t <= total_sim_time + 0.1 and (t - last_snapshot_time >= snapshot_interval):
                     self.frame_snapshots.append(T_curr.copy())
                     last_snapshot_time = t
                
                # 3. Update Trajectory
                self.trajectory_points.append(p_com.copy())
                if len(self.trajectory_points) > self.max_trajectory_len:
                    self.trajectory_points.pop(0)
                
                # 6. Visualization
                viewer.user_scn.ngeom = 0
                
                # Draw stored frame snapshots
                for T_snap in self.frame_snapshots:
                    self._draw_frame(viewer, T_snap)
                
                # Draw Computed Linear Velocity Vector (red arrow)
                self._draw_arrow(viewer, p_com, linear_vel, rgba=[1, 0, 0, 1], scale=4.0)
                
                # Draw Computed Angular Velocity Vector (green arrow)
                if np.linalg.norm(omega_s) > 1e-6:
                    self._draw_arrow(viewer, p_com, omega_s * 0.3, rgba=[0, 1, 0, 0.8], scale=1.0)
                
                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":

    # Load the model
    xml_path = "lab1/asset/franka_emika_panda/scene.xml"
    xml_content = mujoco.MjModel.from_xml_path(xml_path)
    target_link_name = "hand"

    fk_vel_exercise = FKVelocityExercise(xml_content, target_link_name=target_link_name)
    fk_vel_exercise.run_sim_vis()
    fk_vel_exercise.plot_errors()

