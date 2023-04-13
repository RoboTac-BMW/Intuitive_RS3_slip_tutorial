import time
import pybullet as p
import numpy as np
from robotac_sim.robotac_env import RobotacSimEnv
import logging
import robotac_sim.utils as utils
import pickle
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO)
# A logger for this file
log = logging.getLogger('main')


def get_grasp_trajectory(traj_index, object, object_model, slip=True, location='front'):
    forces = []
    slipping = []
    poses = []
    slip_occurred = False
    force_applied = False
    sim_freq = 240
    lift_speed = 0.2  # 0.1 m/sec
    lift_distance = 0.275
    lift_time = lift_distance / lift_speed

    object_pos = [0, 0.425, 0.1]
    object_orn = [0, 0, 0]

    # Define the RobotacSimEnv
    env = RobotacSimEnv(object_model, object_pos, object_orn, show_gui=False)

    # Reset the environment
    env.reset()
    assert env.robot.initialized

    # Object specific
    if 'Banana' in object:
        force_threshold = 5
        y_max = 0.44
        y_min = 0.4
    else:
        force_threshold = 8
        y_max = 0.45
        y_min = 9.38


    grasp_point = None
    if not slip:
        # Grasping around COM
        apply_external_force = False
        grasp_point = 0.425 + np.random.uniform(-0.02, 0.02)
    else:
        apply_external_force = True
        if location == 'front':
            # Clockwise Rotational Slip
            grasp_point = y_max + np.random.uniform(-0.02, 0.02)
            force_location = -0.1 + np.random.uniform(-0.01, 0.01)
        elif location == 'back':
            # Anticlockwise Rotational Slip
            grasp_point = y_min + np.random.uniform(-0.02, 0.02)
            force_location = 0.1 + np.random.uniform(-0.01, 0.01)
        else:
            # Linear slip
            grasp_point = 0.425
            force_location = 0.425

    # Obtain object pose and orientation
    obj_pos, obj_orn, _, _ = env.object.get_observation()

    #  Assumption - objects are of similar height - 0.025
    tool_o_m = np.dot(utils.euler2rotm(obj_orn), np.array(p.getMatrixFromQuaternion([0.7, 0.7, 0, 0])).reshape(3, 3))
    tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

    # Start grasping procedure
    # log.info("Moving to pick-up location")
    env.robot.move_ee(position=[obj_pos[0], grasp_point, 0.15], orientation=tool_o_q, blocking=True, speed=0.01,
                      update_gripper=False)
    env.robot.actuate_gripper(action='open', speed=0.7)
    env.robot.move_ee(position=[obj_pos[0], grasp_point, 0.025], orientation=tool_o_q, blocking=True, speed=0.01,
                      update_gripper=True)
    normal_force = 0
    env.robot.actuate_gripper(action='close', speed=0.1, force=100, timer_out=1)
    time_out = 0

    # Grasp till force threshold reached
    while abs(normal_force) < force_threshold and time_out <= 3000:
        env.robot.update_gripper()
        env.robot.p.stepSimulation()
        force_x, force_y, force_z = env.tactile_sensor.get_observation(env.object.object_id)
        normal_force = force_y
        time_out += 1
    # Keep constant force while grasping
    # env.robot.actuate_gripper(action='close', speed=0.01, force=100, timer_out=1)

    # trajectory = np.linspace(0.025, 0.3, int(num_samples))
    trajectory = np.arange(0.025, lift_distance + 0.025, lift_speed / sim_freq)
    for traj in trajectory:
        # Execute action (position based joint control) - open loop/close loop
        env.robot.move_ee(position=[obj_pos[0], grasp_point, traj], orientation=tool_o_q, blocking=False)
        env.robot.update_gripper()
        env.robot.p.stepSimulation()
        time.sleep(1/sim_freq)
        # Obtain Visual observations of the scene
        # rgb_img, depth_img, seg_mask = env.vision_sensor.get_observation(visual_locations[1])

        # Obtain G.T Object position and orientation
        pos, orn, lin_vel, ang_vel = env.object.get_observation()

        # Obtain the Tactile observations with the object in contact (here only 1)
        force_x, force_y, force_z = env.tactile_sensor.get_observation(env.object.object_id)

        if apply_external_force:
            sampled_height = np.random.uniform(0.1, 0.2)
            if sampled_height < traj < (sampled_height + 0.01):
                p.applyExternalForce(env.object.object_id, -1, [0, 0, -2], [0, force_location, 0], flags=p.LINK_FRAME)
                force_applied = True
        # Start recording the data
        if traj > 0.05:
            if lin_vel[2] < -0.01 or abs(orn[0]) > 0.02 or force_applied:
                slipping.append(1)
                slip_occurred = True
            else:
                if not slip_occurred:
                    slipping.append(0)
                else:
                    slipping.append(1)
        else:
            slipping.append(0)

        forces.append(np.array([force_x, force_y, force_z]))
        poses.append(np.array([pos[2], lin_vel[2], orn[0], ang_vel[0]]))


    # env.close()

    # Generate the dump file for sanity
    forces_array = np.array(forces)
    poses_array = np.array(poses)
    slipping_array = np.array(slipping)

    fig, ax = plt.subplots(2, 4)
    ax[0, 0].plot(forces_array[:, 0], '.b')
    ax[0, 1].plot(forces_array[:, 1], 'Xb')
    ax[0, 2].plot(forces_array[:, 2], '*b')
    ax[0, 3].plot(slipping_array, '.r')
    ax[1, 0].plot(poses_array[:, 0], '.g')
    ax[1, 1].plot(poses_array[:, 1], '*g')
    ax[1, 2].plot(poses_array[:, 2], '.k')
    ax[1, 3].plot(poses_array[:, 3], '*k')
    fig_name = 'dump/' + object + '_' + str(traj_index) + '_grasp_trajectory.png'
    plt.savefig(fig_name)
    plt.close()

    return forces, slipping, slip_occurred


# #################################  Solution code for database collection #############################################
if __name__ == '__main__':
    num_trajectories = 120
    slip_indices = np.arange(0, num_trajectories, num_trajectories / 6)
    object_models_path = os.path.join('robotac_sim', 'descriptions', 'ycb_objects')
    object_files = sorted(os.listdir(object_models_path))
    slip_label = []
    object_label = []
    db_forces = []
    db_slipping = []
    for object in object_files:
        object_model = os.path.join(object_models_path, object, 'model.urdf')
        for traj_index in range(0, num_trajectories):
            print('Collecting ', traj_index, ' out of ', num_trajectories, ' for ', object)
            if traj_index < slip_indices[3]:
                slip = False
                location = None
            elif slip_indices[3] <= traj_index < slip_indices[4]:
                slip = True
                location = 'front'
            elif slip_indices[4] <= traj_index < slip_indices[5]:
                slip = True
                location = 'back'
            elif slip_indices[5] <= traj_index < num_trajectories:
                slip = True
                location = 'center'

            forces, slipping, slipped = get_grasp_trajectory(traj_index, object, object_model, slip=slip, location=location)
            slip_label.append(int(slipped))
            object_label.append(object)

        db = {'forces': db_forces, 'slipping': db_slipping, 'slip_label': slip_label, 'object_label':object_label}
        with open('slip_database.pickle', 'wb') as f:
            pickle.dump(db, f)
