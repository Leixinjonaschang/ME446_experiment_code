import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import argparse

def compute_gravity_torque_RNEA_matrix(data):
    """
    Follow the steps of the Recursive Newton-Euler Algorithm (RNEA) from the tutorial,
    using matrix operations for step-by-step derivation (Kinematic Propagation & Torque Recovery).
    """
    # 1. Extract current joint angles
    q1 = data.qpos[0]
    q2 = data.qpos[1]
    
    # 2. Model parameters extracted from double_pendulum.xml
    m1 = 1.0; l1 = 0.5; r1 = 0.25
    m2 = 2.0;             r2 = 0.25
    g = 9.81
    
    # ---------------- 1. Condition Initialization ----------------
    # "trick" the solver by setting base acceleration a0 = -g 
    # (represents a base acceleration directed opposite to gravity)
    a0 = np.array([[0], 
                   [g]])
    
    # ---------------- 2. Forward Pass: Kinematic Propagation ----------------
    # Link 1 Acceleration: a1 = R^0_1 * a0
    R_1_0 = np.array([
        [np.cos(q1),  np.sin(q1)],
        [-np.sin(q1), np.cos(q1)]
    ])
    a1 = R_1_0 @ a0
    
    # Link 2 Acceleration: a2 = R^1_2 * a1
    R_2_1 = np.array([
        [np.cos(q2),  np.sin(q2)],
        [-np.sin(q2), np.cos(q2)]
    ])
    a2 = R_2_1 @ a1
    
    # ---------------- 3. Backward Pass: Force & Torque Recovery ----------------
    # Joint 2 Torque:
    # Only depends on Link 2's mass and its local pose
    tau2 = r2 * m2 * a2[1, 0]  # a2[1, 0] is the component of upward acceleration in the torque direction
    
    # Joint 1 Torque:
    # (A) Support the weight of Link 1 itself
    tau1_self = r1 * m1 * a1[1, 0]
    
    # (B) Torque transmitted due to supporting Link 2
    # Compute local force vector for Link 2
    f2_local = m2 * a2 
    # Transform force on Link 2 back to Link 1's local frame via R_1_2 (R_1_2 is the transpose of R_2_1)
    R_1_2 = R_2_1.T
    f2_on_link1 = R_1_2 @ f2_local
    # Moment arm of force from Link 2 on Link 1 is l1
    tau1_coupling = l1 * f2_on_link1[1, 0]
    
    # tau1 = self-support torque + counter-force torque from child link + joint torque transmitted from child link
    tau1 = tau1_self + tau1_coupling + tau2
    
    return np.array([tau1, tau2])

def compute_gravity_torque_RNEA_analytical(data):
    """
    Direct calculation according to the final simplified formula from the tutorial.
    This result should be exactly consistent with the matrix version above.
    """
    q1 = data.qpos[0]
    q2 = data.qpos[1]
    
    m1 = 1.0; l1 = 0.5; r1 = 0.25
    m2 = 2.0;             r2 = 0.25
    g = 9.81
    
    # Joint 2 Torque: Only depends on Link 2's mass and pose
    tau2 = r2 * (m2 * g * np.cos(q1 + q2))
    
    # Joint 1 Torque: Must support both Link 1 and the reaction force from Link 2
    # Distinguish contribution from Link 1 and Link 2
    tau1 = m1 * g * r1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + r2 * np.cos(q1 + q2))
    
    return np.array([tau1, tau2])

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Double Pendulum Gravity Compensation Demo")
    parser.add_argument("--gravity_comp", type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help="Enable gravity compensation (default: true)")
    args = parser.parse_args()

    # Get absolute path of the XML model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "../asset/double_pendulum/double_pendulum.xml")
    
    # Load the model and corresponding data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Set an initial pose with significant gravity effect for testing
    data.qpos[:] = [ np.pi / 3 , 0]

    # Ensure actuators are in pure torque control mode (remove default gains and biases)
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biastype[i] = 0

    # Sync kinematics state
    mujoco.mj_forward(model, data)

    # ---------- Cross-validation ----------
    tau_matrix     = compute_gravity_torque_RNEA_matrix(data)
    tau_analytical = compute_gravity_torque_RNEA_analytical(data)
    
    mode_str = "[Gravity Compensation ENABLED]" if args.gravity_comp else "[Gravity Compensation DISABLED]"
    print("==========================================================")
    print(f"Starting 2-DOF RNEA Gravity Compensation Test... {mode_str}")
    print("----------------------------------------------------------")
    print(f"[Verify] Matrix Propagation Method   tau = {np.round(tau_matrix, 4)}")
    print(f"[Verify] Analytical Formula Method   tau = {np.round(tau_analytical, 4)}")
    print(f"[Verify] Max Difference              delta = {np.abs(tau_matrix - tau_analytical).max():.2e}")
    print("==========================================================")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.8
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = 0.0
        
        while viewer.is_running():
            step_start = time.time()
            
            if args.gravity_comp:
                # Calculate and apply torque when gravity compensation is enabled
                tau_gc = compute_gravity_torque_RNEA_matrix(data)
                data.ctrl[:] = tau_gc 
            else:
                # Control output is 0 when gravity compensation is disabled
                data.ctrl[:] = 0.0
            
            # Step MuJoCo physics engine
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Run simulation in real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
