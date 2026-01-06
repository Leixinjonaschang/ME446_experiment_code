import mujoco
import mujoco.viewer
import time
import os
import numpy as np

def main():
    # 1. Load Model
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, "../asset/double_pendulum/double_pendulum.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Launch Passive Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo Viewer started. Double pendulum is swinging under gravity.")
        print("Press Ctrl+C in terminal to exit.")
        
        # Initial position (start from a non-equilibrium state)
        data.qpos[0] = 1.0
        data.qpos[1] = 0.5
        
        # 3. Simulation Loop
        while viewer.is_running():
            step_start = time.time()

            # Physics step
            mujoco.mj_step(model, data)

            # Sync rendering
            viewer.sync()

            # Maintain simulation frequency
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()

