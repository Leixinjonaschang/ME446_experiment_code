#!/usr/bin/env python3
"""
MuJoCo Exercise: Basic Robot Simulation

This script is a template for:
1. Loading a robot model from an MJCF file
2. Accessing robot state (joint positions and velocities)
3. Reading and setting control inputs (actuators)
4. Running a basic simulation loop with visualization
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os


def main():
    # --- 1. Import Robot Model ---
    # TODO: Define the path to your MJCF file
    MJCF_PATH = "lab0/asset/franka_emika_panda/scene.xml" 
    
    # TODO: Load the model from the path
    # Hint: model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    model = None # EXERCISE: Replace with proper loading code
    # Reference Answer:
    # model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    
    # Check if model loaded successfully
    if model is None:
        print("Model not loaded! Please fill in the code in Section 1.")
        return

    # Create data structure for simulation state
    data = mujoco.MjData(model)

    # --- 2. Visualization Setup ---
    # This will open a window showing the simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Starting simulation... Close the window to exit.")
        
        start_time = time.time()
        
        # --- 3. Simulation Loop ---
        while viewer.is_running():
            step_start = time.time()

            # Step the simulation
            # mj_step computes the physics for the next time step
            mujoco.mj_step(model, data) # core function of simulation: s_{t+1} = f(s_{t}, a_{t})

            # --- 4. Read Robot State ---
            # TODO: Read generalized positions (qpos), velocities (qvel), and control (ctrl)
            # Hint: 
            # qpos = data.qpos
            # qvel = data.qvel
            # ctrl = data.ctrl
            
            # Reference Answer:
            # qpos = data.qpos
            # qvel = data.qvel
            # ctrl = data.ctrl
            
            
            # Print state every 0.5 seconds
            if int((time.time() - start_time) * 2) > int((time.time() - start_time - 0.01) * 2):
                pass


            # --- 5. Read and Set Robot Control Input ---
            # TODO: Set the control input for the actuators
            if model.nu > 0:
                # Example: Apply a simple oscillating control to the first actuator
                # Hint: data.ctrl[0] = 1.0 * np.sin(data.time * 2.0 * np.pi)
                
                # EXERCISE: Set data.ctrl[0] here
                pass
                # Reference Answer:
                # data.ctrl[0] = 1.0 * np.sin(data.time * 2.0 * np.pi)

            # Update the viewer with the latest state
            viewer.sync()

            # Maintain real-time simulation speed (optional)
            # time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

