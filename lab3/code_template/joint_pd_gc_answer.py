import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt


TORQUE_LIMIT =87
ARM_DOF = 7


def pd_controller(model, data, q_target, kp, kd):
    """
    Joint-space PD controller.

    Returns:
        tau_cmd: clipped torque actually sent to the robot
        error: joint position error
        tau_pd_raw: unclipped PD torque, useful for analysis/plotting
    """
    q_current = data.qpos[:ARM_DOF]
    v_current = data.qvel[:ARM_DOF]

    error = q_target - q_current
    error_dot = -v_current

    tau_pd_raw = kp * error + kd * error_dot
    tau_cmd = np.clip(tau_pd_raw, -TORQUE_LIMIT, TORQUE_LIMIT)

    return tau_cmd, error, tau_pd_raw


def compute_gravity_torque(model, data):
    """
    Compute the joint-space gravity torque g(q) via the mass matrix.

    In free fall with zero control and qvel=0, the equation of motion reduces to:
        M(q) * qacc = -g(q)
    Therefore:
        g(q) = -M(q) * qacc_freefall

    We measure qacc_freefall by stepping the simulation with all controls and
    external forces disabled, then multiply by the analytically-computed mass
    matrix M(q).

    Args:
        model: MuJoCo model
        data:  MuJoCo data (qpos must already be set to the query configuration)

    Returns:
        g_q: gravity torque for the first ARM_DOF joints
    """
    # ---- save state ----
    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    qacc_save = data.qacc.copy()
    ctrl_save = data.ctrl.copy()
    qfrc_applied_save = data.qfrc_applied.copy()
    xfrc_applied_save = data.xfrc_applied.copy()
    act_save = data.act.copy() if model.na > 0 else None

    # ---- zero velocity and all external forces ----
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    if model.na > 0:
        data.act[:] = 0.0

    # ---- free-fall step: measure gravity-induced acceleration ----
    mujoco.mj_step1(model, data)
    mujoco.mj_step2(model, data)
    qacc_free = data.qacc[:ARM_DOF].copy()

    # ---- compute M(q) ----
    M_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    M = M_full[:ARM_DOF, :ARM_DOF]

    # ---- g(q) = -M * qacc_free ----
    g_q = -M @ qacc_free

    # ---- restore ----
    data.qpos[:] = qpos_save
    data.qvel[:] = qvel_save
    data.qacc[:] = qacc_save
    data.ctrl[:] = ctrl_save
    data.qfrc_applied[:] = qfrc_applied_save
    data.xfrc_applied[:] = xfrc_applied_save
    if act_save is not None:
        data.act[:] = act_save
    mujoco.mj_forward(model, data)

    return g_q

def compute_gravity_torque_rnea(model, data):
    """
    Compute the joint-space gravity torque g(q) using MuJoCo's internal RNEA.

    MuJoCo's mj_rne computes:
        result = M(q) * qacc + C(q, qvel)

    If flg_acc = 0, the inertial term M(q) * qacc is removed, so:
        result = C(q, qvel)

    Under the static condition qvel = 0:
        result = C(q, 0) = g(q)

    Therefore, by setting qvel = 0 and calling mj_rne(..., flg_acc=0),
    we obtain the gravity term directly.

    Args:
        model: MuJoCo model
        data:  MuJoCo data

    Returns:
        g_q: gravity torque for the first ARM_DOF joints
    """
    # ---- save state ----
    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    qacc_save = data.qacc.copy()
    ctrl_save = data.ctrl.copy()
    qfrc_applied_save = data.qfrc_applied.copy()
    xfrc_applied_save = data.xfrc_applied.copy()
    act_save = data.act.copy() if model.na > 0 else None

    # ---- static condition for gravity-only evaluation ----
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    if model.na > 0:
        data.act[:] = 0.0

    # Recompute model-dependent intermediate quantities at current q, qvel=0
    mujoco.mj_forward(model, data)

    # ---- run RNEA without inertial term ----
    rnea_result = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_rne(model, data, 0, rnea_result)

    g_q = rnea_result[:ARM_DOF].copy()

    # ---- restore original state ----
    data.qpos[:] = qpos_save
    data.qvel[:] = qvel_save
    data.qacc[:] = qacc_save
    data.ctrl[:] = ctrl_save
    data.qfrc_applied[:] = qfrc_applied_save
    data.xfrc_applied[:] = xfrc_applied_save
    if act_save is not None:
        data.act[:] = act_save

    mujoco.mj_forward(model, data)

    return g_q

def pd_gc_controller(model, data, q_target, kp, kd):
    """
    PD + gravity compensation.

    tau = Kp * (qd - q) + Kd * (0 - qdot) + g(q)

    Returns:
        tau_cmd: clipped torque actually sent to the robot
        error: joint position error
        tau_pd_raw: unclipped PD torque
        tau_g: gravity torque
        tau_total_raw: unclipped total torque
    """
    _, error, tau_pd_raw = pd_controller(model, data, q_target, kp, kd)
    # tau_g = compute_gravity_torque(model, data)
    tau_g = compute_gravity_torque_rnea(model, data)

    tau_total_raw = tau_pd_raw + tau_g
    tau_cmd = np.clip(tau_total_raw, -TORQUE_LIMIT, TORQUE_LIMIT)

    return tau_cmd, error, tau_pd_raw, tau_g, tau_total_raw


def run_experiment(model, controller_mode="pd", sim_time=15.0):
    """
    Run one experiment and return logged data.

    controller_mode:
        "pd"    -> pure PD
        "pd_gc" -> PD + gravity compensation
    """
    data = mujoco.MjData(model)

    # Reset to home
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    # Torque control mode
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = 0.0
        model.actuator_biasprm[i, 2] = 0.0
        model.actuator_ctrllimited[i] = 0

    kp = np.array([300, 300, 200, 200, 60, 60, 60], dtype=np.float64)
    kd = np.array([50, 50, 50, 50, 50, 50, 50], dtype=np.float64)

    # Gravity-challenging pose
    q_target = np.array([0.8, 0.3, 0.0, -1.2, 0.0, 1.0, 0.5], dtype=np.float64)

    time_history = []
    error_history = []
    tau_pd_history = []
    tau_g_history = []
    tau_total_history = []

    start_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0

        while viewer.is_running():
            step_start = time.time()

            if controller_mode == "pd_gc":
                tau_cmd, error, tau_pd_raw, tau_g, tau_total_raw = pd_gc_controller(
                    model, data, q_target, kp, kd
                )
            else:
                tau_cmd, error, tau_pd_raw = pd_controller(
                    model, data, q_target, kp, kd
                )
                tau_g = np.zeros(ARM_DOF)
                tau_total_raw = tau_pd_raw

            current_time = time.time() - start_time

            time_history.append(current_time)
            error_history.append(error.copy())
            tau_pd_history.append(tau_pd_raw.copy())
            tau_g_history.append(tau_g.copy())
            tau_total_history.append(tau_total_raw.copy())

            data.ctrl[:ARM_DOF] = tau_cmd
            mujoco.mj_step(model, data)
            viewer.sync()

            remain = model.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)

            if current_time > sim_time:
                break

    return {
        "time": np.array(time_history),
        "error": np.array(error_history),
        "tau_pd": np.array(tau_pd_history),
        "tau_g": np.array(tau_g_history),
        "tau_total": np.array(tau_total_history),
        "controller_mode": controller_mode,
    }


def plot_error(result):
    time_array = result["time"]
    error_array = result["error"]
    controller_label = "PD + Gravity Compensation" if result["controller_mode"] == "pd_gc" else "PD Only"

    plt.figure(figsize=(10, 6))
    for i in range(ARM_DOF):
        plt.plot(time_array, error_array[:, i], label=f"Joint {i+1}")
    plt.xlabel("Real Time (s)", fontsize=18)
    plt.ylabel("Position Error (rad)", fontsize=18)
    plt.title(f"Position Error vs Time — {controller_label}", fontsize=20)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    plt.tight_layout()
    plt.show()


def plot_torque_decomposition(result):
    time_array = result["time"]
    tau_pd = result["tau_pd"]
    tau_g = result["tau_g"]
    tau_total = result["tau_total"]
    controller_label = "PD + Gravity Compensation" if result["controller_mode"] == "pd_gc" else "PD Only"

    selected_joints = [0, 3]  # joint 1 and joint 4

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for idx, joint_idx in enumerate(selected_joints):
        ax = axes[idx]
        ax.plot(time_array, tau_pd[:, joint_idx], label="PD torque", linewidth=1.5)
        ax.plot(time_array, tau_g[:, joint_idx], label="Gravity comp torque", linewidth=1.5)
        ax.plot(
            time_array,
            tau_total[:, joint_idx],
            label="Total torque",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_xlabel("Real Time (s)", fontsize=16)
        ax.set_ylabel("Torque (Nm)", fontsize=16)
        ax.set_title(f"Joint {joint_idx+1} Torque Decomposition", fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=12)

    plt.suptitle(f"Torque Decomposition — {controller_label}", fontsize=20)
    plt.tight_layout()
    plt.show()


def main():
    model = mujoco.MjModel.from_xml_path(
        "/home/phi/Downloads/489_lab_materials/ME446_experiment_code/lab3/asset/franka_emika_panda/scene.xml"
    )

    # choose "pd" or "pd_gc"
    # controller_mode = "pd_gc"
    controller_mode = "pd"

    result = run_experiment(model, controller_mode=controller_mode, sim_time=7.0)
    plot_error(result)
    if controller_mode == "pd_gc":
        plot_torque_decomposition(result)


if __name__ == "__main__":
    main()