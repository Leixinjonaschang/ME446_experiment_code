import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def compute_gravity_torque_RNEA_matrix(data):
    """
    按照教程中的 Recursive Newton-Euler Algorithm (RNEA) 步骤，
    使用矩阵运算一步步推导(Kinematic Propagation & Torque Recovery)。
    """
    # 1. 提取当前关节角度
    q1 = data.qpos[0]
    q2 = data.qpos[1]
    
    # 2. 从 double_pendulum.xml 提取的模型参数
    m1 = 1.0; l1 = 0.5; r1 = 0.25
    m2 = 2.0;             r2 = 0.25
    g = 9.81
    
    # ---------------- 1. Condition 初始化 ----------------
    # “trick” the solver by setting base acceleration a0 = -g (基于推导图，表示给基座一个反向于重力方向的加速度)
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
    # 只取决于 Link 2 自己的质量与其局部姿态
    tau2 = r2 * m2 * a2[1, 0]  # a2[1, 0] 为向上的加速度在力矩方向（垂直于重心的方向）的分量
    
    # Joint 1 Torque:
    # (A) 支撑自己 Link 1 的质量影响
    tau1_self = r1 * m1 * a1[1, 0]
    
    # (B) 支撑 Link 2 造成的传递影响力矩
    # 计算 Link 2 局部的受力向量
    f2_local = m2 * a2 
    # 通过 R_1_2 将连杆 2 的受力转回连杆 1 的局部坐标系 (R_1_2 是 R_2_1 的转置)
    R_1_2 = R_2_1.T
    f2_on_link1 = R_1_2 @ f2_local
    # 连杆 2 对连杆 1 产生的力臂长度为 l1
    tau1_coupling = l1 * f2_on_link1[1, 0]
    
    # τ1 = 自身支撑力矩 + 子连杆反作用力矩 + 本身传递的子连杆关节转矩
    tau1 = tau1_self + tau1_coupling + tau2
    
    return np.array([tau1, tau2])

def compute_gravity_torque_RNEA_analytical(data):
    """
    根据教程中的最终化简公式直接计算。
    此结果应该与上方矩阵版本的结果完全一致。
    """
    q1 = data.qpos[0]
    q2 = data.qpos[1]
    
    m1 = 1.0; l1 = 0.5; r1 = 0.25
    m2 = 2.0;             r2 = 0.25
    g = 9.81
    
    # Joint 2 Torque: Only depends on Link 2's mass and pose
    tau2 = r2 * (m2 * g * np.cos(q1 + q2))
    
    # Joint 1 Torque: Must support both Link 1 and the reaction force from Link 2
    # 分辨出 Link 1 contribution 和 Link 2 contribution
    tau1 = m1 * g * r1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + r2 * np.cos(q1 + q2))
    
    return np.array([tau1, tau2])

def main():
    # 获取 XML 模型文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "../asset/double_pendulum/double_pendulum.xml")
    
    # 载入模型和对应的数据集
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 设置一个受重力影响较大的初始偏转姿态进行测试
    data.qpos[:] = [np.pi / 4, np.pi / 4]

    # 确保驱动器为纯力矩控制模式 (消除默认增益和偏置带来影响)
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biastype[i] = 0

    # 同步运动学状态（在第一次控制步之前确保 body 位置/COM 等已更新）
    mujoco.mj_forward(model, data)

    # ---------- 交叉验证：矩阵版 vs 解析公式版 ----------
    tau_matrix     = compute_gravity_torque_RNEA_matrix(data)
    tau_analytical = compute_gravity_torque_RNEA_analytical(data)
    print("==========================================================")
    print("开始双连杆 RNEA 重力补偿测试...")
    print("没有使用 mj_rne，完全使用自定义基于公式与步骤的实现。")
    print("如果计算正确，由于前馈重力补偿，机器人应在此初始姿态下保持悬空静止不变。")
    print("----------------------------------------------------------")
    print(f"[验证] 矩阵传递法   τ = {np.round(tau_matrix, 4)}")
    print(f"[验证] 解析公式法   τ = {np.round(tau_analytical, 4)}")
    print(f"[验证] 最大差值     Δ = {np.abs(tau_matrix - tau_analytical).max():.2e}  (应 ≈ 0)")
    print("==========================================================")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -15.0
        
        while viewer.is_running():
            step_start = time.time()
            
            # 使用手动的 RNEA 矩阵传递函数计算重力补偿力矩 (与解析版计算结果等价)
            tau_gc = compute_gravity_torque_RNEA_matrix(data)
            
            # 将计算得到的所需补偿力矩传给关节驱动器
            data.ctrl[:] = tau_gc
            
            # 步进 Mujoco 物理引擎
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 按实时进行仿真控制
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
