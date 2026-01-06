import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

class TwoLinkID:
    def __init__(self):
        # 1. Load Model from file
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, "../asset/double_pendulum/double_pendulum.xml")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_dof = self.model.nv 
        
        # ground truth read from the robot description file
        self.ground_truth = {
            'm2': 2.0,
            'mx2': 2.0 * 0.25, 
            'I2': 0.167116    
        }

    def get_analytical_regressor(self, q, dq, ddq):
        """
        Build the regressor matrix Y for Joint 2.
        
        Correct dynamics equation for Joint 2 of a planar double pendulum:
        tau2 = (I2 + m2*l1*lc2*cos(q2))*ddq1 + I2*ddq2 + m2*l1*lc2*sin(q2)*dq1^2 + m2*g*lc2*cos(q1+q2)
        
        Rearranging to separate parameters pi = [I2, mx2] where mx2 = m2*lc2:
        tau2 = I2*(ddq1 + ddq2) + mx2*(l1*cos(q2)*ddq1 + l1*sin(q2)*dq1^2 + g*cos(q1+q2))
        """
        q1, q2 = q[0], q[1] # joint position
        dq1, dq2 = dq[0], dq[1] # joint velocity
        ddq1, ddq2 = ddq[0], ddq[1] # joint acceleration
        
        g = 9.81
        l1 = 0.5 # Length of Link 1 (from XML)
        
        # TODO: construct the regressor matrix Y_row for joint 2
        # Y = [ coeff_I2,  coeff_mx2 ], a 1x2 vector.
        # You can get the specific form in the tutorial.
        Y_row = None # replace None with the correct value
        
        return Y_row

    def run_identification(self):
        print("Collecting Data with Visualization...")
        n_steps = 3000
        dt = self.model.opt.timestep
        
        history_Y = []
        history_tau = []
        t_arr = []
        
        # Reset to a non-zero initial position
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.5  # Initial angle for joint1
        self.data.qpos[1] = 0.3  # Initial angle for joint2
        
        # Debug counters
        debug_interval = 500
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for i in range(n_steps):
                step_start = time.time()
                t = i * dt
                
                # Step 1: Data Collection with Excitation Trajectory (Rich enough to excite dynamics)
                q_target = np.array([
                    0.5 * np.sin(2 * t) + 0.2 * np.cos(5 * t),
                    0.8 * np.cos(3 * t) + 0.3 * np.sin(4 * t)
                ])
                dq_target = np.array([
                    1.0 * np.cos(2 * t) - 1.0 * np.sin(5 * t),
                    -2.4 * np.sin(3 * t) + 1.2 * np.cos(4 * t)
                ])
                
                # PD Control to follow trajectory
                kp, kd = 50, 10
                q = self.data.qpos[:2]
                dq = self.data.qvel[:2]
                
                tau = kp * (q_target - q) + kd * (dq_target - dq)
                self.data.ctrl[:2] = tau
                
                mujoco.mj_step(self.model, self.data)
                
                # Compute inverse dynamics to populate qfrc_inverse,
                # Equivalent to simulating the robot joint torque sensor.
                mujoco.mj_inverse(self.model, self.data)
                
                # Get acceleration
                ddq = self.data.qacc[:2]
                
                # Step 2: Construct Regressor
                Y_row = self.get_analytical_regressor(q, dq, ddq)
                current_tau_id = self.data.qfrc_inverse[1]
                
                history_Y.append(Y_row)
                history_tau.append(current_tau_id)
                t_arr.append(t)
                
                viewer.sync()
                
                # Debug output
                if i % debug_interval == 0:
                    print(f"Step {i}: q={q}, tau_id={current_tau_id:.4f}")

                # synchronization for real-time visualization
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                
                if not viewer.is_running():
                    break

        # Step 3: Least Squares Solution
        Y = np.vstack(history_Y) # (N, 2)
        Tau = np.array(history_tau) # (N, )
        
        print(f"\nSolving Linear System: {Y.shape} * pi = {Tau.shape}")
        print(f"Y range: [{Y.min():.4f}, {Y.max():.4f}]")
        print(f"Tau range: [{Tau.min():.4f}, {Tau.max():.4f}]")
        
        # Solve Y * pi = Tau
        pi_est, residuals, rank, s = lstsq(Y, Tau)
        
        print(f"Matrix rank: {rank}, Singular values: {s}")
        
        I2_est = pi_est[0]
        mx2_est = pi_est[1]
        
        # 4. Result Visualization
        print("\n=== Identification Results (Link 2) ===")
        print(f"{'Parameter':<15} | {'Estimated':<10} | {'Ground Truth':<12} | {'Error':<10}")
        print("-" * 55)
        print(f"{'Inertia (I2)':<15} | {I2_est:.5f}    | {self.ground_truth['I2']:.5f}       | {abs(I2_est - self.ground_truth['I2']):.5f}")
        print(f"{'Moment (mx2)':<15} | {mx2_est:.5f}    | {self.ground_truth['mx2']:.5f}       | {abs(mx2_est - self.ground_truth['mx2']):.5f}")
        
        # Validation Plot
        tau_pred = Y @ pi_est
        plt.figure(figsize=(10, 5))
        plt.plot(t_arr, Tau, 'k-', alpha=0.6, linewidth=2, label='MuJoCo Torque (Inverse Dynamics)')
        plt.plot(t_arr, tau_pred, 'r--', linewidth=1.5, label='Predicted Torque (Our Model)')
        plt.title("System Identification Verification (Joint 2)", fontsize=18)
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Torque (N.m)", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = TwoLinkID()
    sim.run_identification()
