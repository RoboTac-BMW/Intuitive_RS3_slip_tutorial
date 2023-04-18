import os
import time
import pybullet as p
from robotac_sim.robotac_env import RobotacSimEnv
from robotac_sim.object_state_visualizer import ObjectStateVisualiser
from robotac_sim.utils import interactive_camera_placement, euler2rotm, rotm2euler, compute_object_rotation, plane_seg
import logging
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

logging.basicConfig(level=logging.INFO)
# A logger for this file
log = logging.getLogger('main')

# ############################################### Part I - Data Collection #############################################
# ############################################## I.1 Load object and robots ############################################
object_models_path = os.path.join('robotac_sim', 'descriptions', 'ycb_objects')
object_files = os.listdir(object_models_path)
# Select a random object to be placed in the scene
# object = np.random.choice(object_files) or
# TODO: Fill in the object and the variation
object = 'YcbMustardBottle'  # Fill among YcbMustardBottle, YcbBanana, YcbHammer, YcbChipsCan
variation = '1'  # Fill among 1, 2, 3

object_model = os.path.join(object_models_path, '_'.join([object, variation]), 'model.urdf')
env = RobotacSimEnv(object_model, show_gui=True)
# Reset the environment
env.reset()
assert env.robot.initialized
rgb_img, depth_img, seg_mask = env.vision_sensor.get_observation()

time.sleep(1)
exit()  # Comment once done with this section

# ####################### I. 2 Find best camera locations to get shape and pose information  ###########################
interactive_camera_placement(env.vision_sensor.camera_id) # Comment this line when done this section
# TODO: Select 3-4 visual locations [x,y,z]
# visual_locations = [[x_1, y_1, z_1], [x_2, y_2, z_2], [x_3, y_3, z_3], [x_4, y_4, z_4]]
visual_locations = []

exit() # Comment once done with this section

# ############ I. 3 Combine the RGB-D information as pointclouds from multiple visual locations  #######################
def crop_pc(point_cloud):
    # TODO: crop the points outside the table (HINT: remember the dimension of the table and the location)
    extent = []
    center = []

    R = np.identity(3)
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    return point_cloud.crop(obb)


point_cloud_merged = o3d.geometry.PointCloud()
# Obtain point cloud from the selected locations
for loc in visual_locations:
    log.info("Resetting Camera positions to: " + str(loc))
    rgb_img, depth_img, seg_mask = env.vision_sensor.get_observation(loc)
    bullet_xyz = env.vision_sensor.get_point_cloud()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(bullet_xyz)
    cropped_point_cloud = point_cloud
    # TODO: crop the pointcloud
    # cropped_point_cloud = crop_pc(point_cloud)

    o3d.visualization.draw_geometries([cropped_point_cloud])
    point_cloud_merged.points.extend(cropped_point_cloud.points)
    time.sleep(1)

exit() # Comment out this when done with this section

rgb_img, depth_img, seg_mask = env.vision_sensor.get_observation(visual_locations[1])
# ##################### I. 4 Remove the ground plane & find object position & orientation ##############################
# TODO: segment the plane, Hint - pass the merged and cropped point cloud
plane_pc, object_pc = plane_seg()

obb = o3d.geometry.OrientedBoundingBox.create_from_points(object_pc.points)
o3d.visualization.draw([object_pc, obb])

object_pos = obb.center
object_rotm = obb.R
object_extent = obb.extent

object_orn = compute_object_rotation(object_rotm)
object_orn_euler = euler2rotm(object_orn)
exit()

# Attach Object State Visualizer
viz = ObjectStateVisualiser(env)
tool_o_m = np.dot(object_orn_euler, np.array(p.getMatrixFromQuaternion([0.7, 0.7, 0, 0])).reshape(3, 3))
tool_o_q = p.getQuaternionFromEuler(rotm2euler(tool_o_m).tolist())

# ######################################### I. 5 Sample Grasp Point ####################################################
object_length = np.max(object_extent)
# TODO: sample a point in the local frame of the object
# grasp_point_of = np.array([[x], [y]])
grasp_point_of = np.array([[], [], [1]])
object_trans_mat = np.array(
    [[object_orn_euler[0, 0], object_orn_euler[0, 1], object_pos[0]],
     [object_orn_euler[1, 0], object_orn_euler[1, 1], object_pos[1]],
     [0, 0, 1]])
grasp_point_wf = np.matmul(object_trans_mat, grasp_point_of)

# ###################################### I. 6 Tune Grasp Force Value ###################################################
log.info("Moving to pick-up location")
env.robot.move_ee(position=[grasp_point_wf[0], grasp_point_wf[1], 0.15], orientation=tool_o_q, blocking=True, speed=0.01,
                  update_gripper=False)
env.robot.actuate_gripper(action='open', speed=0.7)
env.robot.move_ee(position=[grasp_point_wf[0], grasp_point_wf[1], 0.025], orientation=tool_o_q, blocking=True, speed=0.01,
                  update_gripper=True)
normal_force = 0
# TODO: Fill in appropriate force threshold value
force_threshold = 1
env.robot.actuate_gripper(action='close', speed=0.1, force=100, timer_out=1)
# Grasp till force threshold reached
while abs(normal_force) < force_threshold:
    env.robot.update_gripper()
    env.robot.p.stepSimulation()
    force_x, force_y, force_z = env.tactile_sensor.get_observation(env.object.object_id)
    normal_force = force_y

time.sleep(1)

# ########################### I. 6 Perform Object Grasping and Force Analysis ##########################################
lift_speed = 0.2  # 0.1 m/sec
lift_distance = 0.275
lift_time = lift_distance/lift_speed
wait_time = 1
num_samples = lift_time*240

trajectory = np.arange(0.025, lift_distance+0.025, lift_speed/240)
for traj in trajectory:
    # Execute action (position based joint control) - open loop/close loop
    env.robot.move_ee(position=[grasp_point_wf[0], grasp_point_wf[1], traj], orientation=tool_o_q, blocking=False)
    env.robot.update_gripper()
    env.robot.p.stepSimulation()
    # Obtain Visual observations of the scene
    # rgb_img, depth_img, seg_mask = env.vision_sensor.get_observation(visual_locations[1])

    # Obtain G.T Object position and orientation
    pos, orn, lin_vel, ang_vel = env.object.get_observation()

    # Obtain the Tactile observations with the object in contact (here only 1)
    force_x, force_y, force_z = env.tactile_sensor.get_observation(env.object.object_id)

    # TODO: check the force_x, force_y and force_y plots, can try a analytical slip detection approach
    slip = []  # Analytical approach for slip detection
    env.slip_user = slip

    if traj > 0.05:
        viz.start_detecting = True
    viz.update_plot()

# Check for stability of the grasp
for wait_out in range(0, wait_time*240):
    env.p.stepSimulation()
    time.sleep(1/240)
    # rgb_img, depth_img, seg_mask = env.vision_sensor.get_observation(visual_locations[1])

    # Obtain G.T Object position and orientation
    pos, orn, lin_vel, ang_vel = env.object.get_observation()

    # Obtain the Tactile observations with the object in contact (here only 1)
    force_x, force_y, force_z = env.tactile_sensor.get_observation(env.object.object_id)
    viz.update_plot()