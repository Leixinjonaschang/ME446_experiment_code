import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

def pd_controller(model, data, q_target, kp, kd):
    """
    PD control function
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        q_target: target joint positions
        kp: proportional gains
        kd: derivative gains
    """
    # get current state
    q_current = data.qpos[:7]
    v_current = data.qvel[:7]
    
    # TODO: calculate error
    error = None # replace None with the correct error calculation
    error_dot = None # replace None with the correct error derivative
    
    # TODO: PD control law
    torque = None # replace None with the correct PD control law

    torque = np.clip(torque, -87, 87) 
    
    return torque, error

def pid_controller(model, data, q_target, kp, kd, ki, error_integral, dt):
    """
    PID control function
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        q_target: target joint positions
        kp: proportional gains
        kd: derivative gains
        ki: integral gains
        error_integral: accumulated error integral (will be updated in-place)
        dt: time step for integral calculation
    
    Returns:
        error: current position error
    """
    # get current state
    q_current = data.qpos[:7]
    v_current = data.qvel[:7]
    
    # TODO: calculate error
    error = None # replace None with the correct error calculation
    error_dot = None # replace None with the correct error derivative
    
    # update integral term (accumulate error over time)
    error_integral += error * dt
    
    # TODO: PID control law: P + I + D
    torque = None # replace None with the correct PID control law
    
    # set torque saturation (simulate motor limits)
    torque = np.clip(torque, -87, 87) 
    
    return torque, error, error_integral

def main():
    # load 7 DOF panda robot model
    model = mujoco.MjModel.from_xml_path('lab2/asset/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)
    
    # initialize the robot in home position
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    
    # set the actuator to torque control mode
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0  
        model.actuator_biasprm[i, 0] = 0.0  
        model.actuator_biasprm[i, 1] = 0.0  
        model.actuator_biasprm[i, 2] = 0.0 
        model.actuator_ctrllimited[i] = 0 
    
    # set KP and KD for each joint
    kp = np.array([300, 300, 200, 200, 60, 60, 60]) 
    kd = np.array([50, 50, 50, 50, 50, 50, 50])        
    ki = np.array([50, 50, 50, 50, 50, 50, 50])
    error_integral = np.zeros(7, dtype=np.float64)  # initialize as float array
    dt = model.opt.timestep  # use model's timestep 
    
    # target position (in radians)
    q_target = np.array([0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5]) 
    
    # nitialize data recording for plotting
    time_history = []
    error_history = []  # shape: (n_steps, 7)
    start_time = time.time()
    
    # launch simulation environment
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        while viewer.is_running():
            step_start = time.time()
            
            # execute PD control 
            torque, error = pd_controller(model, data, q_target, kp, kd)

            # execute PID control 
            # torque, error, error_integral = pid_controller(model, data, q_target, kp, kd, ki, error_integral, dt)
            
            # record data for plotting
            current_time = time.time() - start_time
            time_history.append(current_time)
            error_history.append(error.copy())
            
            # physical step
            data.ctrl[:7] = torque
            mujoco.mj_step(model, data)
            
            # refresh rendering
            viewer.sync()
            
            # maintain simulation frequency
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            if current_time > 15:
                break
                
    # plot position error vs real time
    if len(time_history) > 0:
        time_array = np.array(time_history)
        error_array = np.array(error_history)  # shape: (n_steps, 7)
        
        plt.figure(figsize=(10, 6))
        for i in range(7):
            plt.plot(time_array, error_array[:, i], label=f'Joint {i+1}')
        plt.xlabel('Real Time (s)', fontsize=18)
        plt.ylabel('Position Error (rad)', fontsize=18)
        plt.title('Position Error vs Real Time for All Joints', fontsize=20)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)

        plt.show()

if __name__ == "__main__":
    main()