import logging
logging.basicConfig(level=logging.INFO)
import sys
import pybullet as p
sys.path.append('..')
import time
import numpy as np
from robotac_sim.utils import link_to_idx, joint_to_idx
import threading
import pybullet_data
import os


# A logger for this file
log = logging.getLogger(__name__)

POS_CTRL = 'pos'
VEL_CTRL = 'vel'


class simUR5:
    """
    # Setup UR5 with Robotiq Gripper robot in simulation
    """
    def __init__(self, physics_id, cid):
        #log.info("Initialising UR5")
        self.p = physics_id
        self.cid = cid
        self.use_nullspace = False
        self.base_position = [0, 0, 0]
        self.base_orientation = self.p.getQuaternionFromEuler([0, 0, 0])
        self.robot_model = 'robotac_sim/descriptions/ur5/ur5_robotiq_2f.urdf'
        self.joint_epsilon = 1e-2
        # self.ur5_initial_joint_positions = np.array([0.252238382702895, -2.6382404165678888, 1.9867945050793192, -0.919691244323651, -1.5734818198621676, 0.0])
        self.ur5_initial_joint_positions = np.array([1.5937507392055053, -0.26221807874604586, -2.2430546509043614, -2.205582499640513, 1.5766058323807324, -1.5478008990415741])
        self.robot_id = None
        self.ur5_joint_indices = None
        self.gripper_joint_indices = self.gripper_passive_joint_indices = None
        self.robot_joint_indices = None
        self.robot_jointLowerLimit = None
        self.robot_jointUpperLimit = None
        self.robot_jointRanges = None
        self.vision_sensor = None
        self.tactile_sensor = None
        self.ll = self.ul = self.jr = self.rp = None
        self.mixed_ik = None
        self.euler_obs = True

        # Done with initialization
        self.initialized = False

    def load(self):
        # Add UR5 robot to simulation environment
        self.robot_id = self.p.loadURDF(self.robot_model, self.base_position, self.base_orientation)
        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [self.p.getJointInfo(self.robot_id, x) for x in range(self.p.getNumJoints(self.robot_id))]
        self.ur5_joint_indices = [x[0] for x in robot_joint_info if x[2] == self.p.JOINT_REVOLUTE and x[0] < 10]
        # 10th joint is the attachment of the gripper to the ur5 base
        self.gripper_joint_indices = [x[0] for x in robot_joint_info if
                                      x[2] == self.p.JOINT_REVOLUTE and x[0] > 10]  # Only inner fingers should be actuated
        self.gripper_passive_joint_indices = self.gripper_joint_indices[1:]
        self.robot_joint_indices = self.ur5_joint_indices + self.gripper_joint_indices

        self.robot_jointLowerLimit = [self.p.getJointInfo(self.robot_id, x)[8] for x in self.robot_joint_indices]
        self.robot_jointUpperLimit = [self.p.getJointInfo(self.robot_id, x)[9] for x in self.robot_joint_indices]
        self.robot_jointRanges = [self.robot_jointUpperLimit[x] - self.robot_jointLowerLimit[x] for x in range(len(self.robot_jointLowerLimit))]

        # link name to link index mappings
        self.lnk2Idx_robot = link_to_idx(self.robot_id)
        # self.lnk2Idx_gripper = link_to_idx(self.gripper_body_id)

        # joint name to joint index mapping
        self.jn2Idx_robot = joint_to_idx(self.robot_id)
        # self.jn2Idx_gripper = joint_to_idx(self.gripper_body_id)

        self.tcp_link_id = self.jn2Idx_robot['tool_fixed_joint']  # For planning purposes

        # Set friction coefficients for gripper fingers
        self.p.changeDynamics(self.robot_id,
                                self.lnk2Idx_robot['robotiq_2f_85_right_follower'],
                                lateralFriction=0.8,
                                spinningFriction=0.05,
                                rollingFriction=0.05)
        self.p.changeDynamics(self.robot_id,
                                self.lnk2Idx_robot['robotiq_2f_85_left_follower'],
                                lateralFriction=0.8,
                                spinningFriction=0.05,
                                rollingFriction=0.05)
        self.p.changeDynamics(self.robot_id,
                                self.lnk2Idx_robot['robotiq_2f_85_right_pad'],
                                lateralFriction=0.8,
                                spinningFriction=0.05,
                                rollingFriction=0.05)
        self.p.changeDynamics(self.robot_id,
                                self.lnk2Idx_robot['robotiq_2f_85_left_pad'],
                                lateralFriction=0.8,
                                spinningFriction=0.05,
                                rollingFriction=0.05)

    def reset(self):
        self.move_joints(self.ur5_initial_joint_positions, blocking=True, speed=1.0)
        self.actuate_gripper(action='close')
        #log.info('Closed Gripper')
        self.p.stepSimulation()
        time.sleep(1./240)
        self.actuate_gripper(action='open')
        #log.info('Open Gripper')
        self.p.stepSimulation()
        time.sleep(1./240)
        self.initialized = True

    def get_observation(self):
        raise NotImplementedError

    # --------------------------------------------------------  Extension methods ---------------------------------------------------------------------
    def update_gripper(self):
        # Use position control to enforce hard constraints on gripper behavior as pybullet does not capture the mimic tag in urdf
        gripper_joint_positions = np.array([self.p.getJointState(self.robot_id, i)[0] for i in self.gripper_joint_indices])
        self.p.setJointMotorControlArray(self.robot_id, self.gripper_passive_joint_indices, self.p.POSITION_CONTROL,
                                    [-gripper_joint_positions[0], gripper_joint_positions[0],
                                     gripper_joint_positions[0], -gripper_joint_positions[0],
                                     gripper_joint_positions[0]],
                                    positionGains=0.02*np.ones(len(self.gripper_passive_joint_indices)))

    def actuate_gripper(self, blocking=True, action='open', speed=0.5, force=1, timer_out=300):
        # Gripper control via veloctiy control
        if action == 'open':
            targetVelocity = -speed
            gripper_target_config = [-0.001, 0.001, -0.001, 0.001, -0.001]
        elif action == 'close':
            targetVelocity = speed
            gripper_target_config = [-0.6, 0.6, 0.6, -0.6, 0.6]
        self.p.setJointMotorControl2(self.robot_id, self.gripper_joint_indices[0], self.p.VELOCITY_CONTROL, targetVelocity=targetVelocity, force=force)

        # Block call until gripper joints move to target configuration, time_out is critical for compliant movement
        if blocking:
            time_out = 0
            reach_target = False
            while time_out < timer_out and not reach_target:
                self.update_gripper()
                self.p.stepSimulation()
                # time.sleep(1. / 240)
                actual_joint_config = [self.p.getJointState(self.robot_id, x)[0] for x in self.gripper_passive_joint_indices]
                reach_target = all([np.abs(actual_joint_config[i] - gripper_target_config[i]) < self.joint_epsilon for i in range(len(self.gripper_passive_joint_indices))])
                time_out = time_out + 1

    # Move robot arm to specified joint configuration
    def move_joints(self, target_joint_config, blocking=False, speed=0.03, update_gripper=True, timer_out=300):
        # Move joints via position control
        #self.p.setJointMotorControlArray(self.robot_id, self.ur5_joint_indices, self.p.POSITION_CONTROL, target_joint_config, positionGains=speed * np.ones(len(self.ur5_joint_indices)))
        self.p.setJointMotorControlArray(self.robot_id, self.ur5_joint_indices, self.p.POSITION_CONTROL, target_joint_config)
        # Block call until joints move to target configuration
        if blocking:
            actual_joint_config = [self.p.getJointState(self.robot_id, x)[0] for x in self.ur5_joint_indices]
            time_out = 0
            while time_out < timer_out or not all([np.abs(actual_joint_config[i] - target_joint_config[i]) < self.joint_epsilon for i in range(6)]):
                if update_gripper:
                    self.update_gripper()
                self.p.stepSimulation()
                time.sleep(1. / 240)
                actual_joint_config = [self.p.getJointState(self.robot_id, x)[0] for x in self.ur5_joint_indices]
                time_out += 1

    def move_ee(self, position, orientation, blocking=True, speed=0.03, update_gripper=False, timer_out=300):
        # Use IK to compute target joint configuration
        target_joint_config = self.p.calculateInverseKinematics(self.robot_id, self.tcp_link_id, position,
                                                           orientation,
                                                           lowerLimits=self.robot_jointLowerLimit,
                                                           upperLimits=self.robot_jointUpperLimit,
                                                           jointRanges=self.robot_jointRanges,
                                                           maxNumIterations=100,
                                                           residualThreshold=1e-3)
        # Move joints
        self.move_joints(target_joint_config[:6], blocking, speed, update_gripper, timer_out)

    '''
    # -----------------High Level primitives---------------------------------
    def primitive_gohome(self, speed=0.03):
        self.move_tool(position=self.center + [self.prepare_height],
                       orientation=[-1.0, 1.0, 0.0, 0.0], blocking=True, speed=0.1)
        self.move_tool(position=[0, 0.1, 0.3], orientation=[-1.0, 1.0, 0.0, 0.0], blocking=True, speed=0.1)

    # modified by xzj:
    def primitive_push_tilt(self, position, rotation_angle, speed=0.01):
        push_orientation = [1.0, 0.0]
        # tool_rotation_angle = rotation_angle/2
        # tool_orientation = np.asarray([push_orientation[0]*np.cos(tool_rotation_angle) - push_orientation[1]*np.sin(tool_rotation_angle), push_orientation[0]*np.sin(tool_rotation_angle) + push_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
        # tool_orientation_an gle = np.linalg.norm(tool_orientation)
        # tool_orientation_axis = tool_orientation/tool_orientation_angle
        # tmp = np.array([tool_orientation_angle, tool_orientation_axis[0],tool_orientation_axis[1],tool_orientation_axis[2]])
        # tool_orientation_rotm = utils.angle2rotm(tmp)

        # tool_rotation_angle = rotation_angle + np.pi / 2
        tool_rotation_angle = rotation_angle
        tool_o_m = np.dot(utils.euler2rotm([0, 0, tool_rotation_angle]),
                          np.array(p.getMatrixFromQuaternion([-1, 1, 0, 0])).reshape(3, 3))
        tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

        # Compute push direction and endpoint
        push_direction = np.asarray(
            [push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
             push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle), 0.0])
        distance = 0.1
        target_x = position[0] + push_direction[0] * distance
        target_y = position[1] + push_direction[1] * distance
        position_end = np.asarray([target_x, target_y, position[2]])
        push_direction.shape = (3, 1)

        # Compute tilted tool orientation during push
        tilt_axis = np.dot(utils.euler2rotm(np.asarray([0, 0, np.pi / 2]))[:3, :3], push_direction)
        tilt_rotm = utils.angle2rotm([-np.pi / 8, tilt_axis[0], tilt_axis[1], tilt_axis[2]])
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_o_m)
        tool_t_q = p.getQuaternionFromEuler(utils.rotm2euler(tilted_tool_orientation_rotm).tolist())

        # Attempt push
        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q, blocking=True, speed=0.1)

        self.move_tool(position, orientation=tool_o_q, blocking=True, speed=0.1)
        self.move_tool(position_end, orientation=tool_t_q, blocking=True, speed=speed)

        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q, blocking=True, speed=0.1)

    # modified by xzj
    def primitive_topdown_grasp(self, position, rotation_angle):

        tool_o_m = np.dot(utils.euler2rotm([0, 0, rotation_angle]), np.array(p.getMatrixFromQuaternion([-1, 1, 0, 0])).reshape(3, 3))
        tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

        # Approach target location +0.3
        self.move_tool(position=self.center + [self.prepare_height],
                       orientation=tool_o_q, blocking=True, speed=0.1)
        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q, blocking=True, speed=0.1)

        self.open_gripper(blocking=True)
        self.move_tool(position, orientation=tool_o_q, blocking=True)
        self.close_gripper(blocking=True)

        # Lift object up 10 cm and check if something has been grasped
        self.move_tool(position=[position[0], position[1], position[2] + 0.1],
                       orientation=tool_o_q, blocking=True)
        tmp_grasp_success = True  # self.check_grasp()
        return tmp_grasp_success

    def primitive_place(self, position, rotation_angle, speed=0.01):
        tool_o_m = np.dot(utils.euler2rotm([0, 0, rotation_angle]),
                          np.array(p.getMatrixFromQuaternion([-1, 1, 0, 0])).reshape(3, 3))
        tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q,
                       speed=speed, blocking=True)

        self.move_tool(position, tool_o_q, blocking=True, speed=speed)
        self.open_gripper(blocking=True)

    def primitive_touch(self, position, rotation_angle):
        tool_o_m = np.dot(utils.euler2rotm([0, 0, rotation_angle]),
                          np.array(p.getMatrixFromQuaternion([-1, 1, 0, 0])).reshape(3, 3))
        tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q, blocking=True)
        self.close_gripper(blocking=True)
        self.move_tool(position, tool_o_q, blocking=True)
        self.open_gripper(blocking=True)
        self.close_gripper(blocking=True)

    ############################################################################################
    '''