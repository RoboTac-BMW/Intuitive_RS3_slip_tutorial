import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import gym
from gym.utils import seeding
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
from robotac_sim.simRobot import simUR5
from robotac_sim.simSensors import simCam, simTactile
from robotac_sim.simObjects import simMovableObject
import os

# A logger for this file
log = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class FpsController:
    def __init__(self, freq):
        self.loop_time = (1.0 / freq) * 10 ** 9
        self.prev_time = time.time_ns()

    def step(self):
        current_time = time.time_ns()
        delta_t = current_time - self.prev_time
        if delta_t < self.loop_time:
            time.sleep(self.loop_time - delta_t)
        self.prev_time = time.time_ns()


class RobotacSimEnv(gym.Env):
    """
    Superclass for PyBullet-based gym environments for Sim-RoboTac environment.
    """
    def __init__(self, action_primitive, cam_position, object_model, object_pos, object_orn, freq=240, show_gui=False):

        self.bullet_time_step = freq
        self.dt = 1/self.bullet_time_step
        # self.fps_controller = FpsController(self.bullet_time_step)
        control_freq = 30
        self.action_primitive = action_primitive
        self.action_repeat = int(self.bullet_time_step // control_freq)
        self.np_random = None
        self.cid = -1
        # self.seed()

        self.show_gui = show_gui
        self.initialize_bullet()

        self.robot = simUR5(self.p, self.cid)
        self.object = simMovableObject(self.p, self.cid, object_model, initial_pos=object_pos, initial_orn=object_orn)
        self.vision_sensor = simCam(self.p, self.cid, cam_position)
        self.tactile_sensor = simTactile(self.p, self.cid)

        self.load()
        # log.info("Environment successfully loaded")

    # Env methods
    # ------------------------------------------------------------------------------------------------------------------
    def initialize_bullet(self):
        if self.cid < 0:
            self.ownsPhysicsClient = True
            if self.show_gui:
                self.p = bc.BulletClient(connection_mode=p.GUI)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to connect to GUI.")
            else:
                self.p = bc.BulletClient(connection_mode=p.DIRECT)
                cid = self.p._client
                if cid < 0:
                   log.error("Failed to start DIRECT bullet mode.")
            #log.info(f"Connected to server with id: {cid}")

            self.cid = cid
            self.p.resetSimulation(physicsClientId=self.cid)
            # self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)
            # self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            # self.p.setTimeStep(0.01, physicsClientId=self.cid)
            return cid

    def load(self):
        #log.info("Resetting simulation")
        self.p.resetSimulation(physicsClientId=self.cid)
        #log.info("Setting gravity")

        #log.info("Loading the ground plane")
        # load PyBullet default plane
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        # load the table
        self.p.loadURDF('robotac_sim/descriptions/scene_setup/robotac_table.urdf', basePosition=[0.0, 0.4, 0.0], useFixedBase=True)

        # load the robot
        self.robot.load()

        # load the object
        self.object.load()

        self.p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        # load the tactile sensor
        if self.action_primitive == 'grasp':
            sensor_links = ['robotiq_2f_85_right_follower', 'robotiq_2f_85_left_follower']
        elif self.action_primitive == 'push':
            sensor_links = ['robotiq_2f_85_right_pad']

        self.tactile_sensor.attach_sensor(self.robot.robot_id, self.robot.lnk2Idx_robot, sensor_links)
        self.robot.vision_sensor = self.vision_sensor
        self.robot.tactile_sensor = self.tactile_sensor

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if np.any(np.isnan(action)):
            print(f"action has NaNs: {action}")
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self._set_action(action)

        self._step_sim()
        self._step_callback()

        self.current_pos, self.current_vel = self._get_joint_states()

        obs = self._get_obs()
        reward = self._compute_reward()
        done = False
        info = {
            'is_success': self._is_success(),  # used by the HER algorithm
        }
        return obs, reward, done, info

    def close(self):
        """
        Cleanup sim.
        """
        if self.ownsPhysicsClient:
            print("disconnecting id %d from server" % self.cid)
            if self.cid >= 0 and self.p is not None:
                try:
                    self.p.disconnect(physicsClientId=self.cid)
                except TypeError:
                    pass
        else:
            print("does not own physics client id")

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self._reset_callback()
        obs = self._get_obs()
        return obs

    def _reset_sim(self):
        """
        Resets a simulation and indicates whether or not it was successful.
        """
        self.robot.reset()
        self.object.reset()
        self.p.stepSimulation(physicsClientId=self.cid)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=228.4, cameraPitch=-19.4, cameraTargetPosition=[0.25, 0.08, 0.02])

        # Let the simulation settle up
        count = 500
        while count > 0:
            self.p.stepSimulation(physicsClientId=self.cid)
            count -= 1

        return True

    def _env_setup(self, initial_state):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        self._reset_state(initial_state)

    def _step_sim(self):
        """Steps one timestep.
        """
        p.stepSimulation()
        # time.sleep(self.dt)

    def _set_action(self, action):
        """
        Applies the given action to the simulation.
        """
        raise NotImplementedError

    def _get_obs(self):
        """Collect camera, robot and object observations."""
        # rgb_obs, depth_obs = self.camera.get_observation()
        obs = True
        #robot_obs, robot_info = self.robot.get_observation()
        #object_obs = self.object.get_observation()
        #obs = {"robot_obs": robot_obs, "object_obs": object_obs, "rgb_obs": rgb_obs, "depth_obs": depth_obs}

        return obs

    def _compute_reward(self):
        """
        Returns the reward for this timestep.
        """
        # raise NotImplementedError()
        return 1

    def _is_success(self):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        # raise NotImplementedError()
        return 1

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _reset_callback(self):
        """A custom callback that is called after resetting the simulation. Can be used
        to randomize certain environment properties.
        """
        pass

    # PyBullet Wrapper
    # ----------------------------
    def _reset_state(self, state):
        raise NotImplementedError

    def _set_joint_pos(self, joint_idx, joint_pos):
        if self.robotId:
            p.resetJointState(self.robotId, joint_idx, joint_pos)

    '''
    def _lock_joints(self, joint_name):
        joint_idx = self.jn2Idx[joint_name]
        current_pos = self.current_pos[self.joints.index(joint_name)]
        p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                jointIndex=joint_idx,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=current_pos,
                                force=500)
        print('Locking Joint successful')
    '''

    def _set_desired_q(self, joint_idx, des_q):
        p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                jointIndex=joint_idx,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=des_q)

    def get_contact_info(self, bodyA, bodyB, linkA, linkB):
        if self.robotId:
            cps = p.getContactPoints(bodyA=bodyA,
                                     bodyB=bodyB,
                                     linkIndexA=linkA,
                                     linkIndexB=linkB)

            return cps
        else:
            return None

    def get_link_state(self, link_name):
        ls = p.getLinkState(self.robotId, self.lnk2Idx[link_name])
        pos = ls[4]
        orient = ls[5]
        return pos, orient

    def _get_joint_states(self):
        pos = []
        vel = []

        for js in p.getJointStates(self.robotId, [self.jn2Idx[jn] for jn in self.joints]):
            pos.append(js[0])
            vel.append(js[1])
        return pos, vel

    def create_desired_state(self, des_qs):
        """Creates a complete desired joint state from a partial one.
        """
        ds = self.desired_pos.copy()

        for jn, des_q in des_qs.items():
            if jn not in self.joints:
                print(f"Unknown joint {jn}")
                exit(-1)
            ds[self.joints.index(jn)] = des_q
        return ds

    def get_state_dicts(self):
        return dict(zip(self.joints, self.current_pos)), dict(zip(self.joints, self.current_vel))

    def get_desired_q_dict(self):
        return dict(zip(self.joints, self.desired_pos))

