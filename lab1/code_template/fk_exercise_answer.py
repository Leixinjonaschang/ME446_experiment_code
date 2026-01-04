#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import expm
import time

# 1. Load the model
xml_path = "lab1/asset/franka_emika_panda/scene.xml"
xml_content = mujoco.MjModel.from_xml_path(xml_path)

class FKExercise:
    def __init__(self, model_xml):
        """
        Initializes the FK exercise class with the MuJoCo model and data.
        Finds the target body ID and extracts initial kinematic parameters.
        """
        self.model = model_xml
        self.data = mujoco.MjData(self.model)
        
        # Get joint IDs
        self.joint_ids = [self.model.joint(i).id for i in range(self.model.njnt) if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        
        # Target Body: hand
        target_name = "hand"
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        if self.hand_id == -1:
            self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
            
        # Extract PoE initial parameters
        self.M, self.S_list = self._extract_kinematic_params()

    def _skew(self, v):
        """
        Computes the skew-symmetric matrix for a given 3D vector.
        Essential for converting angular velocity vectors into matrix form.
        """
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _vec_to_se3(self, S):
        """
        Converts a 6D spatial twist (omega, v) into a 4x4 se(3) matrix.
        Used as the input for the matrix exponential function.
        """
        se3 = np.zeros((4, 4))
        se3[:3, :3] = self._skew(S[:3])
        se3[:3, 3] = S[3:]
        return se3

    def _extract_kinematic_params(self):
        """
        Extracts the Home configuration matrix (M) and spatial screw axes (S_list)
        from the MuJoCo model at the zero configuration (q=0).
        """
        self.data.qpos[:] = 0
        mujoco.mj_forward(self.model, self.data)
        M = np.eye(4)
        M[:3, 3] = self.data.xpos[self.hand_id]
        M[:3, :3] = self.data.xmat[self.hand_id].reshape(3, 3)
        S_list = []
        for j_id in self.joint_ids:
            omega = self.data.xaxis[j_id]
            p = self.data.xanchor[j_id]
            v = -np.cross(omega, p)
            S_list.append(np.concatenate([omega, v]))
        return M, S_list

    def forward_kinematics_poe(self, q):
        """
        Computes the Forward Kinematics using the Product of Exponentials formula.
        T = e^[S1]q1 * e^[S2]q2 * ... * e^[Sn]qn * M
        """
        T = np.eye(4)
        for i in range(len(q)):
            T = T @ expm(self._vec_to_se3(self.S_list[i]) * q[i])
        return T @ self.M

    def _draw_axis(self, viewer, pos, mat, alpha, scale=0.3):
        """Internal method: draws a set of RGB coordinate axes in the scene"""
        axis_width = 0.01
        for i in range(3):
            rgba = [0, 0, 0, alpha]
            rgba[i] = 1.0
            
            # Calculate arrow direction
            target_axis = mat[:, i]
            z_axis = target_axis
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
                size=[axis_width, axis_width, scale],
                pos=pos,
                mat=arrow_rot,
                rgba=rgba
            )
            viewer.user_scn.ngeom += 1

    def visualize(self):
        """
        Launches a passive MuJoCo viewer to visualize the robot and the FK results.
        Draws coordinate frames for both Ground Truth and calculated results.
        """
        # 1. Set Home pose
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            self.data.qpos[:] = self.model.key_qpos[key_id]
        mujoco.mj_forward(self.model, self.data)

        print("\nVisualization Started.")
        print("Transparent Frame (Alpha 0.3): Real Hand (GT)")
        print("Semi-Solid Frame (Alpha 0.6): Calculated FK")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Ensure global frame display is turned off
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            
            while viewer.is_running():
                # Compute FK
                q_curr = self.data.qpos[:len(self.joint_ids)]
                T_fk = self.forward_kinematics_poe(q_curr)
                
                # Get GT pose
                pos_gt = self.data.xpos[self.hand_id]
                mat_gt = self.data.xmat[self.hand_id].reshape(3, 3)
                
                # Reset user geometries
                viewer.user_scn.ngeom = 0
                
                # 1. Draw the real Hand Link coordinate frame (GT) - high transparency
                self._draw_axis(viewer, pos_gt, mat_gt, alpha=0.3)
                
                # 2. Draw the calculated FK coordinate frame - keep original transparency
                self._draw_axis(viewer, T_fk[:3, 3], T_fk[:3, :3], alpha=0.6)
                
                # Real-time error printing
                if int(time.time() * 2) % 10 == 0:
                    err = np.linalg.norm(T_fk[:3, 3] - pos_gt)
                    print(f"FK Error: {err:.10f} m", end='\r')

                viewer.sync()
                time.sleep(0.01)
    
    def validate_fk(self):
        """
        Validates the custom FK implementation by comparing it with MuJoCo's 
        internal forward kinematics and printing the Euclidean distance error.
        """
        q_curr = self.data.qpos[:len(self.joint_ids)]
        T_fk = self.forward_kinematics_poe(q_curr)
        pos_fk = T_fk[:3, 3]
        pos_gt = self.data.xpos[self.hand_id]
        err = np.linalg.norm(pos_fk - pos_gt)
        print(f"FK Error: {err:.10f} m")

if __name__ == "__main__":
    fk_exercise = FKExercise(xml_content)
    fk_exercise.validate_fk()
    fk_exercise.visualize()
