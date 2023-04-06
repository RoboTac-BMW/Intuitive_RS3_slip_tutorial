import numpy as np
import pybullet as p
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class simMovableObject(object):
    def __init__(self, p, cid, filename, initial_pos=None, initial_orn=None):
        self.p = p
        self.cid = cid
        self.name = 'rect1'
        self.object_file = filename
        self.initial_pos = initial_pos or np.array([0, 0, 0])
        self.initial_orn = p.getQuaternionFromEuler(initial_orn) or np.array(p.getQuaternionFromEuler([0, 0, 0]))
        self.euler_obs = True

    def load(self):
        self.object_id = self.p.loadURDF(self.object_file, self.initial_pos, self.initial_orn, globalScaling=1, physicsClientId=self.cid, flags=self.p.URDF_USE_INERTIA_FROM_FILE)

    def reset(self, state=None):
        if state is None:
            initial_pos = self.initial_pos
            initial_orn = self.initial_orn
        else:
            initial_pos, initial_orn = np.split(state, [3])
            if len(initial_orn) == 3:
                initial_orn = self.p.getQuaternionFromEuler(initial_orn)
        self.p.resetBasePositionAndOrientation(
            self.object_id,
            initial_pos,
            initial_orn,
            physicsClientId=self.cid,
        )

    def get_observation(self):
        pos, orn = self.p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.cid)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.object_id, physicsClientId=self.cid)
        if self.euler_obs:
            orn = self.p.getEulerFromQuaternion(orn)
        return pos, orn, lin_vel, ang_vel

    def get_info(self):
        pos, orn = self.p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.cid)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.object_id, physicsClientId=self.cid)
        obj_info = {
            "current_pos": pos,
            "current_orn": orn,
            "current_lin_vel": lin_vel,
            "current_ang_vel": ang_vel,
            "contacts": self.p.getContactPoints(bodyA=self.object_id, physicsClientId=self.cid),
            "uid": self.object_id,
        }
        return obj_info

    def set_physical_properties(self, obj_num, obj_id_list, cfg):
        """
        :param obj_num:
        :param obj_id_list:
        :param cfg:
        :return:
        """
        friction_discrete, mass_discrete = [], []
        for i in range(obj_num):
            friction_discrete.append(np.random.choice(range(cfg.Friction_categories)))
            mass_discrete.append(np.random.choice(range(cfg.Mass_categories)))

        friction_list = [cfg.Friction_min + x / (cfg.Friction_categories - 1) * (cfg.Friction_max - cfg.Friction_min)
                         for x
                         in friction_discrete]
        mass_list = [cfg.Mass_min + x / (cfg.Mass_categories - 1) * (cfg.Mass_max - cfg.Mass_min) for x in
                     mass_discrete]

        # auxiliary
        friction_list.append(cfg.Friction_auxiliary)
        mass_list.append(cfg.Mass_auxiliary)

        for index, (friction, mass) in enumerate(zip(friction_list, mass_list)):
            self.p.changeDynamics(obj_id_list[index], -1,
                             lateralFriction=friction,
                             mass=mass,
                             spinningFriction=0.5,
                             restitution=1.0)

        # auxiliary
        p.changeDynamics(obj_id_list[-1], -1, spinningFriction=0.5, restitution=0.8)

        return friction_discrete, mass_discrete