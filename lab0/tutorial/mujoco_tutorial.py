#!/usr/bin/env python3
"""
MuJoCo Tutorial: Basic Robot Simulation

This script demonstrates:
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
    # TODO: Replace with your actual MJCF file path
    # Example: MJCF_PATH = "path/to/your/robot.xml"
    MJCF_PATH = "lab0/asset/franka_emika_panda/scene.xml" 
    
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)

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
            # qpos: generalized positions (joint angles, base position, etc.)
            # qvel: generalized velocities
            # ctrl: actuator control signals
            qpos = data.qpos
            qvel = data.qvel
            ctrl = data.ctrl
            
            # Print state every 0.5 seconds
            if int((time.time() - start_time) * 2) > int((time.time() - start_time - 0.01) * 2):
                # Format numerical output with fewer decimals for brevity
                qpos_str = np.array2string(qpos, precision=3, separator=', ')
                qvel_str = np.array2string(qvel, precision=3, separator=', ')
                ctrl_str = np.array2string(ctrl, precision=3, separator=', ')
                print(f"Time: {data.time:.2f}s")
                print(f"  Robot Positions (qpos): {qpos_str}")
                print(f"  Robot Velocities (qvel): {qvel_str}")
                print(f"  Robot Control (ctrl): {ctrl_str}")
            # --- 5. Read and Set Robot Control Input ---
            # ctrl: actuator control signals
            # Depending on your model, this could be motor torques, position targets, etc.
            
            # Example: Apply a simple oscillating control to the first actuator if it exists
            if model.nu > 0:
                # Read current control (though typically you set it)
                current_ctrl = data.ctrl
                
                # Set new control input (e.g., a sine wave)
                data.ctrl[0] = 1.0 * np.sin(data.time * 2.0 * np.pi)
                pass

            # Update the viewer with the latest state
            viewer.sync()

            # Maintain real-time simulation speed
            # time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

