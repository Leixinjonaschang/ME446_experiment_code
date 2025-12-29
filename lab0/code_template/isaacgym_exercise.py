#!/usr/bin/env python3
"""
Isaac Gym Exercise: Multi-Robot Attractor Control

Task:
1. Load the Franka Panda URDF asset.
2. Create a 2x2 grid of environments.
3. Configure DOF properties for position control.
4. Update the attractor target in the simulation loop to make the robot 
   perform a circular motion.
"""

import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

def main():
    # --- 1. Initialize Gym & Sim ---
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="Isaac Gym Exercise")

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        return

    # Add ground
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # --- 2. Load Asset ---
    asset_root = "lab0/asset/franka_description/robots" 
    asset_file = "franka_panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01

    print(f"Loading asset '{asset_file}' from '{asset_root}'")
    # TODO: EXERCISE: Load the asset
    # franka_asset = ...
    franka_asset = None # Replace me
    # Reference Answer:
    # franka_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # --- 3. Create Environments ---
    num_envs = 4
    num_per_row = 2
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    envs = []
    franka_handles = []
    attractor_handles = []

    # TODO: EXERCISE: Initialize trajectory storage for each environment
    # trajectories = [[] for _ in range(num_envs)]
    trajectories = [] # Replace me
    # Reference Answer:
    # trajectories = [[] for _ in range(num_envs)]

    # Attractor Template
    attractor_props = gymapi.AttractorProperties()
    attractor_props.stiffness = 5e5
    attractor_props.damping = 5e3
    attractor_props.axes = gymapi.AXIS_ALL

    print(f"Creating {num_envs} environments...")
    for i in range(num_envs):
        # TODO: EXERCISE: Create environment
        # env = ...
        env = None # Replace me
        envs.append(env)
        # Reference Answer:
        # env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        # envs[i] = env

        # TODO: EXERCISE: Create franka actor
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(0, 0, 0)
        # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        # handle = ...
        handle = None # Replace me
        franka_handles.append(handle)
        # Reference Answer:
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(0, 0, 0)
        # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        # handle = gym.create_actor(env, franka_asset, pose, "franka", i, 2)
        # franka_handles[i] = handle

        # TODO: EXERCISE: Set up the attractor for the 'panda_hand'
        # 1. Find the hand body handle
        # 2. Set attractor_props.rigid_handle
        # 3. Create the attractor: gym.create_rigid_body_attractor(env, attractor_props)
        pass
        # Reference Answer:
        # body_dict = gym.get_actor_rigid_body_dict(env, handle)
        # props = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)
        # hand_handle = gym.find_actor_rigid_body_handle(env, handle, "panda_hand")
        # attractor_props.rigid_handle = hand_handle
        # attractor_props.target = props['pose'][:][body_dict["panda_hand"]]
        # attractor_handle = gym.create_rigid_body_attractor(env, attractor_props)
        # attractor_handles.append(attractor_handle)

    # --- 4. Configure DOF Properties ---
    if len(envs) > 0:
        # TODO: EXERCISE: Get DOF properties from the first env/actor
        # props = gym.get_actor_dof_properties(...)
        
        # TODO: EXERCISE: Set stiffness, damping and driveMode
        # props['stiffness'].fill(...)
        # props['damping'].fill(...)
        # props['driveMode'].fill(gymapi.DOF_MODE_POS)
        
        # TODO: EXERCISE: Apply properties to ALL environments in a loop
        pass
        # Reference Answer:
        # props = gym.get_actor_dof_properties(envs[0], franka_handles[0])
        # props['stiffness'].fill(1000.0)
        # props['damping'].fill(1000.0)
        # props['driveMode'].fill(gymapi.DOF_MODE_POS)
        # for i in range(num_envs):
        #     gym.set_actor_dof_properties(envs[i], franka_handles[i], props)

    # --- 5. Viewer Setup ---
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(4.0, 1.5, 3.0), gymapi.Vec3(0, 0, 1))

    # --- 6. Simulation Loop ---
    while not gym.query_viewer_has_closed(viewer):
        t = gym.get_sim_time(sim)
        
        # Clear previous lines
        gym.clear_lines(viewer)

        # TODO: EXERCISE: Update attractor targets to move in a circle
        # For each environment:
        # 1. Get current attractor properties
        # 2. Update target.p.x and target.p.z using math.sin(t) and math.cos(t)
        # 3. Set the new target
        # 4. Store and draw the trajectory (similar to tutorial)
        
        # Reference Answer:
        # for i in range(num_envs):
        #     # Update Attractor
        #     attractor_props = gym.get_attractor_properties(envs[i], attractor_handles[i])
        #     pose = attractor_props.target
        #     pose.p.x = 0.2 * math.sin(t)
        #     pose.p.z = 0.5 + 0.2 * math.cos(t)
        #     gym.set_attractor_target(envs[i], attractor_handles[i], pose)
        #
        #     # Update and Draw Trajectory
        #     trajectories[i].append([pose.p.x, pose.p.y, pose.p.z])
        #     if len(trajectories[i]) > 100: trajectories[i].pop(0)
        #
        #     if len(trajectories[i]) > 1:
        #         verts = []
        #         for j in range(len(trajectories[i]) - 1):
        #             verts.append(trajectories[i][j])
        #             verts.append(trajectories[i][j+1])
        #         colors = [[1, 1, 0]] * (len(verts) // 2)
        #         gym.add_lines(viewer, envs[i], len(colors), verts, colors)
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()

