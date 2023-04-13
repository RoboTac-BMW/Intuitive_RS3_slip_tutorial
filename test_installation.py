# Check all imports
import time
import numpy as np
from robotac_sim.robotac_env import RobotacSimEnv
from robotac_sim.object_state_visualizer import ObjectStateVisualiser
import logging
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('main')

if __name__ == '__main__':
    # Create the sim_env
    log.info('Setting up RoboTac Sim')
    try:
        env = RobotacSimEnv('robotac_sim/descriptions/ycb_objects/YcbBanana_1/model.urdf', show_gui=False)
        env.reset()
        viz = ObjectStateVisualiser(env)
        log.info('Setting up Vizualizer')
        assert env.robot.initialized
        env.p.stepSimulation()
        # time.sleep(2)
        log.info('Sim Setup Done')
        env.close()
        print('Open3D version: ', o3d.__version__)
        random_noise = np.random.uniform(0, 5, 100)
        plt.plot(random_noise, '.r')
        # time.sleep(1)
        plt.close()
        print('Matplotlib version: ', matplotlib.__version__)
        print('Tensorflow version installed: ', tf.__version__)
        print('GPU enabled: ', tf.test.is_gpu_available())
        print("All test passed, Installation successful")
    except:
        log.error('Installation Failed')



