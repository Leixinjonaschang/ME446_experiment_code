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
    
    # 2. Model parameters
    m1 = 1.0; l1 = 0.5; r1 = 0.25
    m2 = 2.0;             r2 = 0.25
    g = 9.81
    
    # ---------------- 1. Condition Initialization ----------------
    # "trick" the solver by setting base acceleration a0 = -g 
    a0 = np.array([[0], 
                   [g]])
    
    # ---------------- 2. Forward Pass: Kinematic Propagation ----------------
    # TODO: Define Rotation Matrix from Frame 0 to Frame 1: R_1_0
    R_1_0 = np.zeros((2, 2)) 

    # TODO: Propagate base acceleration to Link 1: a1 = R_1_0 * a0
    a1 = np.zeros((2, 1))

    # TODO: Define Rotation Matrix from Frame 1 to Frame 2: R_2_1
    R_2_1 = np.zeros((2, 2))

    # TODO: Propagate acceleration to Link 2: a2 = R_2_1 * a1
    a2 = np.zeros((2, 1))
    
    # ---------------- 3. Backward Pass: Force & Torque Recovery ----------------
    # Joint 2 Torque:
    # TODO: Compute tau2 (Depends on Link 2's mass and its local acceleration a2)
    tau2 = 0.0
    
    # Joint 1 Torque:
    # (A) Support the weight of Link 1 itself
    # TODO: Compute tau1_self
    tau1_self = 0.0
    
    # (B) Torque transmitted due to supporting Link 2
    # TODO: Compute local force vector for Link 2: f2_local = m2 * a2
    f2_local = np.zeros((2, 1))

    # TODO: Transform force on Link 2 back to Link 1's local frame via R_1_2
    # Hint: R_1_2 is the transpose of R_2_1
    f2_on_link1 = np.zeros((2, 1))

    # TODO: Compute torque on Joint 1 caused by Link 2's weight (moment arm l1)
    tau1_coupling = 0.0
    
    # Final tau1 = self-support torque + coupling torque from child link + joint torque from child link
    # TODO: Sum the components to get tau1
    tau1 = 0.0
    
    return np.array([tau1, tau2])

def compute_gravity_torque_RNEA_analytical(data):
    """
    Direct calculation according to the final simplified analytical formula.
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
    parser = argparse.ArgumentParser(description="Double Pendulum Gravity Compensation Lab Template")
    parser.add_argument("--gravity_comp", type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help="Enable gravity compensation (default: true)")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "../asset/double_pendulum/double_pendulum.xml")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    data.qpos[:] = [ np.pi / 3 , 0]

    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biastype[i] = 0

    mujoco.mj_forward(model, data)

    # ---------- Cross-validation ----------
    tau_matrix     = compute_gravity_torque_RNEA_matrix(data)
    tau_analytical = compute_gravity_torque_RNEA_analytical(data)
    
    mode_str = "[Gravity Compensation ENABLED]" if args.gravity_comp else "[Gravity Compensation DISABLED]"
    print("==========================================================")
    print(f"Starting 2-DOF RNEA Gravity Compensation Test... {mode_str}")
    print("----------------------------------------------------------")
    print(f"[Verify] Matrix Method:       tau = {np.round(tau_matrix, 4)}")
    print(f"[Verify] Analytical Method:   tau = {np.round(tau_analytical, 4)}")
    print(f"[Verify] Max Difference:      delta = {np.abs(tau_matrix - tau_analytical).max():.2e}")
    print("==========================================================")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.8
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = 0.0
        
        while viewer.is_running():
            step_start = time.time()
            if args.gravity_comp:
                tau_gc = compute_gravity_torque_RNEA_matrix(data)
                data.ctrl[:] = tau_gc 
            else:
                data.ctrl[:] = 0.0
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
