"""
Lab 3: Task-Space Control Comparison — Inverse Dynamics vs IK+PD+GC
────────────────────────────────────────────────────────────────────
Runs both controllers on the same circular trajectory and plots
tracking error / torque / XY trajectory comparisons.

  (A) Inverse Dynamics  — imported from task_inv_dyna_control.py
  (B) Task-space IK     — q̇_cmd = J†(ẋ_d + K(x_d−x)) → PD + g(q)
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt

# ── reuse everything from the inverse-dynamics module ──
from task_inv_dyna_control_template import (
    TORQUE_LIMIT, ARM_DOF,
    desired_trajectory,
    get_ee_position, get_jacobian,
    task_inv_dyna_controller, # needs to be implemented
    draw_sphere, draw_circle_markers,
    plot_tracking_error, plot_trajectory_xy, plot_joint_torques,
)


# ─────────────────── gravity-only RNEA (for IK controller) ────────
def compute_gravity_torque_rnea(model, data):
    """
    Compute g(q) using RNEA with qvel=0, flg_acc=0.
    Saves/restores state so the main simulation is not disturbed.
    """
    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    qacc_save = data.qacc.copy()
    ctrl_save = data.ctrl.copy()
    qfrc_save = data.qfrc_applied.copy()
    xfrc_save = data.xfrc_applied.copy()
    act_save  = data.act.copy() if model.na > 0 else None

    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    if model.na > 0:
        data.act[:] = 0.0

    mujoco.mj_forward(model, data)
    result = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_rne(model, data, 0, result)
    g_q = result[:ARM_DOF].copy()

    data.qpos[:] = qpos_save
    data.qvel[:] = qvel_save
    data.qacc[:] = qacc_save
    data.ctrl[:] = ctrl_save
    data.qfrc_applied[:] = qfrc_save
    data.xfrc_applied[:] = xfrc_save
    if act_save is not None:
        data.act[:] = act_save
    mujoco.mj_forward(model, data)
    return g_q


# ──────────────── Controller B: Task-space IK + PD + GC ───────────
def task_ik_controller(model, data, ee_body_id,
                      x_d, x_dot_d,
                      K_task, q_target,
                      kp_joint, kd_joint, dt):
    """
    Task-space IK controller:
        1. Resolved-rate IK:  q̇_cmd = J†(ẋ_d + K(x_d − x))
        2. Integrate:         q_target += q̇_cmd · dt
        3. Joint PD + GC:     τ = Kp(q_target − q) + Kd(q̇_cmd − q̇) + g(q)

    Args:
        K_task   : task-space proportional gain  (3×3)
        q_target : mutable array — updated in place  (ARM_DOF,)
        kp_joint : joint-space proportional gains     (ARM_DOF,)
        kd_joint : joint-space derivative gains       (ARM_DOF,)

    Returns:
        tau_cmd  : clipped torque   (ARM_DOF,)
        x_error  : position error   (3,)
        tau_raw  : unclipped torque  (ARM_DOF,)
    """
    # ── current state ──
    x     = get_ee_position(data, ee_body_id)
    J     = get_jacobian(model, data, ee_body_id)
    q     = data.qpos[:ARM_DOF]
    q_dot = data.qvel[:ARM_DOF]

    # ── resolved-rate IK ──
    x_error = x_d - x
    x_dot_cmd = x_dot_d + K_task @ x_error
    J_pinv = np.linalg.pinv(J)
    q_dot_cmd = J_pinv @ x_dot_cmd

    # ── integrate to joint target ──
    q_target += q_dot_cmd * dt

    # ── joint PD + gravity compensation ──
    tau_g = compute_gravity_torque_rnea(model, data)
    tau_pd = kp_joint * (q_target - q) + kd_joint * (q_dot_cmd - q_dot)
    tau_raw = tau_pd + tau_g

    tau_cmd = np.clip(tau_raw, -TORQUE_LIMIT, TORQUE_LIMIT)
    return tau_cmd, x_error, tau_raw


# ─────────────────────── experiment runner ────────────────────────
def run_experiment(model, controller_mode="inv_dyna", sim_time=20.0):
    """
    controller_mode:
        "inv_dyna" – task-space inverse dynamics  (imported)
        "ik"       – task-space IK + joint PD + gravity compensation
    """
    data = mujoco.MjData(model)

    # Reset to home configuration
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    # Switch actuators to pure torque mode
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = 0.0
        model.actuator_biasprm[i, 2] = 0.0
        model.actuator_ctrllimited[i] = 0

    # End-effector body
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # Compute initial EE position → use as circle centre
    mujoco.mj_forward(model, data)
    ee_init = data.xpos[ee_body_id].copy()
    xc, yc, zc = ee_init
    mode_label = "Inv-Dyna" if controller_mode == "inv_dyna" else "IK+PD+GC"
    print(f"[{mode_label}] Initial EE position (circle centre): "
          f"x={xc:.4f}  y={yc:.4f}  z={zc:.4f}")

    # ── trajectory parameters ──
    r     = 0.1               # radius  [m]
    omega = 2 * np.pi         # angular velocity  [rad/s]

    dt = model.opt.timestep

    # ── controller-specific state ──
    if controller_mode == "inv_dyna":
        Kp = np.diag([400.0, 400.0, 400.0])
        Kd = np.diag([ 40.0,  40.0,  40.0])
        J_prev = None
    else:  # "ik"
        K_task   = np.diag([10.0, 10.0, 10.0])   # task-space IK gain
        kp_joint = np.array([600, 600, 400, 400, 100, 100, 100], dtype=np.float64)
        kd_joint = np.array([ 50,  50,  50,  50,  20,  20,  20], dtype=np.float64)
        q_target = data.qpos[:ARM_DOF].copy()     # initialise to home

    # ── logging ──
    time_log    = []
    x_err_log   = []
    x_des_log   = []
    x_act_log   = []
    tau_log     = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 2.5
        viewer.cam.azimuth   = 150.0
        viewer.cam.elevation = -30.0

        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # Desired trajectory at current time
            x_d, x_dot_d, x_ddot_d = desired_trajectory(t, xc, yc, zc, r, omega)

            # Controller
            if controller_mode == "inv_dyna":
                tau_cmd, x_error, J_prev, tau_raw = task_inv_dyna_controller(
                    model, data, ee_body_id,
                    x_d, x_dot_d, x_ddot_d,
                    Kp, Kd, J_prev, dt,
                )
            else:
                tau_cmd, x_error, tau_raw = task_ik_controller(
                    model, data, ee_body_id,
                    x_d, x_dot_d,
                    K_task, q_target,
                    kp_joint, kd_joint, dt,
                )

            # Log
            time_log.append(t)
            x_err_log.append(x_error.copy())
            x_des_log.append(x_d.copy())
            x_act_log.append(data.xpos[ee_body_id].copy())
            tau_log.append(tau_raw.copy())

            # Apply torque & step
            data.ctrl[:ARM_DOF] = tau_cmd
            mujoco.mj_step(model, data)

            # ── visualisation: desired circle + current target ──
            viewer.user_scn.ngeom = 0
            draw_circle_markers(viewer, xc, yc, zc, r)
            draw_sphere(viewer, x_d, size=0.02, rgba=(1, 0.2, 0.2, 1))

            viewer.sync()

            remain = model.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)

            if t > sim_time:
                break

    return {
        "time":    np.array(time_log),
        "x_error": np.array(x_err_log),
        "x_des":   np.array(x_des_log),
        "x_act":   np.array(x_act_log),
        "tau":     np.array(tau_log),
        "circle":  {"xc": xc, "yc": yc, "zc": zc, "r": r},
        "mode":    controller_mode,
    }


# ───────────────────── comparison plots ───────────────────────────
def plot_comparison(res_inv, res_ik):
    """Side-by-side comparison of the two controllers."""
    # ── tracking error norm ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for res, label, color in [
        (res_inv, "Inverse Dynamics", "tab:blue"),
        (res_ik,  "IK + PD + GC",    "tab:orange"),
    ]:
        err_norm = np.linalg.norm(res["x_error"], axis=1)
        axes[0].plot(res["time"], err_norm, label=label, linewidth=1.2, color=color)
    axes[0].set_ylabel("‖position error‖ (m)", fontsize=14)
    axes[0].legend(fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Tracking Error Norm Comparison", fontsize=16)

    # ── torque norm ──
    for res, label, color in [
        (res_inv, "Inverse Dynamics", "tab:blue"),
        (res_ik,  "IK + PD + GC",    "tab:orange"),
    ]:
        tau_norm = np.linalg.norm(res["tau"], axis=1)
        axes[1].plot(res["time"], tau_norm, label=label, linewidth=1.0, color=color)
    axes[1].set_ylabel("‖τ‖ (Nm)", fontsize=14)
    axes[1].set_xlabel("Time (s)", fontsize=14)
    axes[1].legend(fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Torque Norm Comparison", fontsize=16)

    plt.tight_layout()
    plt.show()

    # ── XY trajectory overlay ──
    plt.figure(figsize=(8, 8))
    plt.plot(res_inv["x_des"][:, 0], res_inv["x_des"][:, 1],
             "k--", label="Desired", linewidth=1.5)
    plt.plot(res_inv["x_act"][:, 0], res_inv["x_act"][:, 1],
             label="Inv-Dyna", linewidth=1.2, color="tab:blue")
    plt.plot(res_ik["x_act"][:, 0], res_ik["x_act"][:, 1],
             label="IK+PD+GC", linewidth=1.2, color="tab:orange")
    plt.xlabel("X (m)", fontsize=14)
    plt.ylabel("Y (m)", fontsize=14)
    plt.title("XY Trajectory Comparison", fontsize=18)
    plt.legend(fontsize=13)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ──────────────────────────── main ────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Task-Space Control: Inverse Dynamics vs IK+PD+GC")
    parser.add_argument("--sim_time", type=float, default=5.0)
    args = parser.parse_args()

    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../asset/franka_emika_panda/scene.xml",
    )
    model = mujoco.MjModel.from_xml_path(xml_path)

    print("="*60)
    print("  Running Inverse Dynamics controller ...")
    print("="*60)
    res_inv = run_experiment(model, "inv_dyna", args.sim_time)

    print("\n" + "="*60)
    print("  Running IK + PD + GC controller ...")
    print("="*60)
    res_ik = run_experiment(model, "ik", args.sim_time)

    plot_comparison(res_inv, res_ik)



if __name__ == "__main__":
    main()
