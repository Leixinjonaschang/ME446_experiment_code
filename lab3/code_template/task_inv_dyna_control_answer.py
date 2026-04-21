"""
Lab 3: Task-Space Inverse Dynamics Control
──────────────────────────────────────────
Circular trajectory tracking using task-space inverse dynamics.

Algorithm:
  1. ẍ_cmd = ẍ_d + Kp(x_d − x) + Kd(ẋ_d − ẋ)
  2. q̈_cmd = J†(ẍ_cmd − J̇ q̇)
  3. τ = M(q) q̈_cmd + C(q,q̇)q̇ + g(q)
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import matplotlib.pyplot as plt

TORQUE_LIMIT = 87
ARM_DOF = 7


# ─────────────────────────── trajectory ───────────────────────────
def desired_trajectory(t, xc, yc, zc, r, omega):
    """
    Circular trajectory in the x-y plane.

    Returns:
        x_d      : desired position  (3,)
        x_dot_d  : desired velocity  (3,)
        x_ddot_d : desired acceleration (3,)
    """
    x_d = np.array([
        xc + r * np.cos(omega * t),
        yc + r * np.sin(omega * t),
        zc,
    ])
    x_dot_d = np.array([
        -r * omega * np.sin(omega * t),
         r * omega * np.cos(omega * t),
         0.0,
    ])
    x_ddot_d = np.array([
        -r * omega**2 * np.cos(omega * t),
        -r * omega**2 * np.sin(omega * t),
         0.0,
    ])
    return x_d, x_dot_d, x_ddot_d


def get_ee_position(data, ee_body_id):
    """Current end-effector Cartesian position."""
    return data.xpos[ee_body_id].copy()


def get_jacobian(model, data, ee_body_id):
    """
    Positional Jacobian J_p (3 × ARM_DOF) for the body origin of
    the end-effector link (not COM).
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
    return jacp[:, :ARM_DOF]


def compute_dynamics(model, data):
    """
    Returns:
        M   : mass matrix               M(q)                    (ARM_DOF × ARM_DOF)
        c_g : Coriolis + gravity term    C(q,q̇)q̇ + g(q)       (ARM_DOF,)
    """
    # ---- mass matrix M(q) ----
    M_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    M = M_full[:ARM_DOF, :ARM_DOF]

    # ---- C(q,q̇)q̇ + g(q) via RNEA with flg_acc=0 ----
    rnea_result = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_rne(model, data, 0, rnea_result)
    c_g = rnea_result[:ARM_DOF].copy()

    return M, c_g


def task_inv_dyna_controller(model, data, ee_body_id,
                             x_d, x_dot_d, x_ddot_d,
                             Kp, Kd,
                             J_prev, dt):
    """
    Task-space inverse dynamics controller (position-only, 3-DOF task).

    Returns:
        tau_cmd  : clipped torque sent to actuators     (ARM_DOF,)
        x_error  : task-space position error            (3,)
        J        : current Jacobian (cached for J̇)     (3 × ARM_DOF)
        tau_raw  : unclipped torque                     (ARM_DOF,)
    """
    # ── current state ──
    x     = get_ee_position(data, ee_body_id)
    J     = get_jacobian(model, data, ee_body_id)
    q_dot = data.qvel[:ARM_DOF]
    x_dot = J @ q_dot                          # ẋ = J q̇

    # ── commanded task-space acceleration ──
    x_error     = x_d - x
    x_dot_error = x_dot_d - x_dot
    x_ddot_cmd  = x_ddot_d + Kp @ x_error + Kd @ x_dot_error

    # ── J̇ q̇  via finite difference of J ──
    if J_prev is not None:
        J_dot = (J - J_prev) / dt
    else:
        J_dot = np.zeros_like(J)
    J_dot_q_dot = J_dot @ q_dot

    # ── commanded joint acceleration ──
    # q̈_cmd = J† (ẍ_cmd − J̇ q̇)
    J_pinv     = np.linalg.pinv(J)
    q_ddot_cmd = J_pinv @ (x_ddot_cmd - J_dot_q_dot)

    # ── inverse dynamics: τ = M q̈_cmd + C q̇ + g ──
    M, c_g = compute_dynamics(model, data)
    tau_raw = M @ q_ddot_cmd + c_g

    tau_cmd = np.clip(tau_raw, -TORQUE_LIMIT, TORQUE_LIMIT)
    return tau_cmd, x_error, J, tau_raw


# ───────────────────── visualisation  ──────────────────────
def draw_sphere(viewer, pos, size=0.015, rgba=(1, 0, 0, 0.8)):
    """Draw a small sphere marker in the viewer."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[size, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def draw_circle_markers(viewer, xc, yc, zc, r, n_pts=60,
                         size=0.005, rgba=(0, 0.8, 0.2, 0.4)):
    """Draw small spheres along the desired circle."""
    for i in range(n_pts):
        theta = 2.0 * np.pi * i / n_pts
        pos = [xc + r * np.cos(theta), yc + r * np.sin(theta), zc]
        draw_sphere(viewer, pos, size=size, rgba=rgba)


# ─────────────────────── experiment ───────────────────────────────
def run_experiment(model, sim_time=20.0):
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
    print(f"[Inv-Dyna] Initial EE position (circle centre): "
          f"x={xc:.4f}  y={yc:.4f}  z={zc:.4f}")

    # ── trajectory parameters ──
    r     = 0.1               # radius  [m]
    omega = 2 * np.pi         # angular velocity  [rad/s]

    # ── task-space PD gains ──
    Kp = np.diag([400.0, 400.0, 400.0])
    Kd = np.diag([ 40.0,  40.0,  40.0])

    dt     = model.opt.timestep
    J_prev = None

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
            tau_cmd, x_error, J_prev, tau_raw = task_inv_dyna_controller(
                model, data, ee_body_id,
                x_d, x_dot_d, x_ddot_d,
                Kp, Kd, J_prev, dt,
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
        "mode":    "inv_dyna",
    }


# ───────────────────────── plotting ───────────────────────────────
def plot_tracking_error(result, title_suffix=""):
    t   = result["time"]
    err = result["x_error"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, err[:, i], linewidth=1.2)
        ax.set_ylabel(f"{label} error (m)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
    axes[-1].set_xlabel("Time (s)", fontsize=14)
    fig.suptitle(f"End-Effector Tracking Error{title_suffix}", fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_trajectory_xy(result, title_suffix=""):
    x_des = result["x_des"]
    x_act = result["x_act"]

    plt.figure(figsize=(8, 8))
    plt.plot(x_des[:, 0], x_des[:, 1], "r--", label="Desired", linewidth=1.5)
    plt.plot(x_act[:, 0], x_act[:, 1], "b-",  label="Actual",  linewidth=1.2)
    plt.xlabel("X (m)", fontsize=14)
    plt.ylabel("Y (m)", fontsize=14)
    plt.title(f"End-Effector Trajectory (XY Plane){title_suffix}", fontsize=18)
    plt.legend(fontsize=13)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_joint_torques(result, title_suffix=""):
    t   = result["time"]
    tau = result["tau"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()
    for i in range(ARM_DOF):
        axes[i].plot(t, tau[:, i], linewidth=1.0)
        axes[i].set_ylabel(f"τ_{i+1} (Nm)", fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=10)
    if ARM_DOF < len(axes):
        axes[-1].axis("off")
    axes[-2].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle(f"Joint Torques{title_suffix}", fontsize=18)
    plt.tight_layout()
    plt.show()


# ──────────────────────────── main ────────────────────────────────
def main():
    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../asset/franka_emika_panda/scene.xml",
    )
    model = mujoco.MjModel.from_xml_path(xml_path)

    result = run_experiment(model, sim_time=20.0)

    plot_tracking_error(result, " — Inverse Dynamics")
    plot_trajectory_xy(result, " — Inverse Dynamics")


if __name__ == "__main__":
    main()
