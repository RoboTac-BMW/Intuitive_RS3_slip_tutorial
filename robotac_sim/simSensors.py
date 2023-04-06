import pybullet as p
import numpy as np
from collections import deque
from robotac_sim.utils import changeFrame


class simCam(object):
    def __init__(self, physics_id, cid, camera_pos):
        """
        Simulates the synthetic RGB-D camera
        """
        self.nearval = 0.01
        self.farval = 2
        self.fov = 90
        self.aspect = 1
        self.width = 480
        self.height = 480
        self.cid = cid
        self.p = physics_id
        self.name = 'azure_kinect'
        self.camera_home_pos = camera_pos

    def set_position_from_gui(self):
        info = self.p.getDebugVisualizerCamera(physicsClientId=self.cid)
        look_at = np.array(info[-1])
        dist = info[-2]
        forward = np.array(info[5])
        look_from = look_at - dist * forward
        self.viewMatrix = self.p.computeViewMatrix(cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.cameraUpVector)
        look_from = [float(x) for x in look_from]
        look_at = [float(x) for x in look_at]
        return look_from, look_at

    def get_observation(self, cameraEyePosition=None):
        self.cameraEyePosition = cameraEyePosition or self.camera_home_pos
        self.cameraTargetPosition = [0, 0.3, 0]  # center of the RoboTac table
        self.cameraUpVector = [0, -1, 0]
        self.viewMatrix = self.p.computeViewMatrix(cameraEyePosition=self.cameraEyePosition, cameraTargetPosition=self.cameraTargetPosition, cameraUpVector=self.cameraUpVector)
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov=self.fov, aspect=self.aspect, nearVal=self.nearval, farVal=self.farval)

        image = self.p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            physicsClientId=self.cid,
        )
        rgb_img, depth_img, seg_mask = self.process_image(image, self.nearval, self.farval)
        return rgb_img, depth_img, seg_mask

    def z_buffer_to_real_distance(self, z_buffer, far, near):
        """Function to transform depth buffer values to distances in camera space"""
        return 1.0 * far * near / (far - (far - near) * z_buffer)

    def process_image(self, obs, nearval, farval):
        (width, height, rgbPixels, depthPixels, segmentationMaskBuffer) = obs
        self.depth_opengl = depthPixels
        rgb = np.reshape(rgbPixels, (height, width, 4))
        rgb_img = rgb[:, :, :3]
        depth_buffer = np.reshape(depthPixels, [height, width])
        depth = self.z_buffer_to_real_distance(z_buffer=depth_buffer, far=farval, near=nearval)
        seg_mask = np.reshape(segmentationMaskBuffer, [width, height])
        return rgb_img, depth, seg_mask

    # Reference: world2pixel
    # https://github.com/bulletphysics/bullet3/issues/1952
    def project(self, point):
        """
        Projects a world point in homogeneous coordinates to pixel coordinates
        Args
            point: np.array of len 4; indicates the desired point to project
        Output
            (x, y): tuple (u, v); pixel coordinates of the projected point
        """

        # reshape to get homogeneus transform
        persp_m = np.array(self.projectionMatrix).reshape((4, 4)).T
        view_m = np.array(self.viewMatrix).reshape((4, 4)).T

        # Perspective proj matrix
        world_pix_tran = persp_m @ view_m @ point
        world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
        world_pix_tran[:3] = (world_pix_tran[:3] + 1) / 2
        x, y = world_pix_tran[0] * self.width, (1 - world_pix_tran[1]) * self.height
        x, y = np.floor(x).astype(int), np.floor(y).astype(int)
        return (x, y)

    def deproject(self, point, depth_img, homogeneous=False):
        """
        Deprojects a pixel point to 3D coordinates
        Args
            point: tuple (u, v); pixel coordinates of point to deproject
            depth_img: np.array; depth image used as reference to generate 3D coordinates
            homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                         else returns the world coordinates (x, y, z) position
        Output
            (x, y): np.array; world coordinates of the deprojected point
        """
        T_world_cam = np.linalg.inv(np.array(self.viewMatrix).reshape((4, 4)).T)

        u, v = point
        z = depth_img[v, u]
        foc = self.height / (2 * np.tan(np.deg2rad(self.fov) / 2))
        x = (u - self.width // 2) * z / foc
        y = -(v - self.height // 2) * z / foc
        z = -z
        world_pos = T_world_cam @ np.array([x, y, z, 1])
        if not homogeneous:
            world_pos = world_pos[:3]
        return world_pos

    def get_point_cloud(self):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        # get a depth image
        # "infinite" depths will have a value close to self.farVal

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projectionMatrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.viewMatrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.height, -1:1:2 / self.width]
        y *= -1.
        x, y, z = np.array(x).reshape(-1), np.array(y).reshape(-1), np.array(self.depth_opengl).reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < (self.farval-0.05)]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3:4]
        points = points[:, :3]

        del self.depth_opengl
        del self.projectionMatrix
        del self.viewMatrix

        return points


class simTactile(object):
    """
    Simulates synthetic tactile sensor providing average normal and shear force between the robot and the object
    """
    def __init__(self, physics_id, cid):
        self.cid = cid
        self.p = physics_id

        self.force_x = []
        self.force_y = []
        self.force_z = []
        self.force_smoothing = 1
        self.force_noise_mu = 0.0
        self.force_noise_sigma = 0.001
        self.force_threshold = 10

        self.force_x_buffer = deque(maxlen=self.force_smoothing)
        self.force_y_buffer = deque(maxlen=self.force_smoothing)
        self.force_z_buffer = deque(maxlen=self.force_smoothing)

    def attach_sensor(self, robot_id, lnk2Idx_robot, sensor_links):
        self.robot_id = robot_id
        self.lnk2Idx_robot = lnk2Idx_robot
        self.sensor_link_ids = sensor_links

    def get_sensor_pose(self, sensor_link):

        position, orientation = self.p.getLinkState(self.robot_id, self.lnk2Idx_robot[sensor_link], computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=self.cid)[:2]
        orientation = self.p.getEulerFromQuaternion(orientation, physicsClientId=self.cid)

        return position, orientation

    def get_observation(self, object_id):
        # Pybullet provides forces in world frame, this will cancel out the net force value
        # TODO: Generalize the force computation
        total_force_x = 0  # X
        total_force_y = 0  # Y
        total_force_z = 0  # Z
        # contact_locations = []
        # contact_velocities = []

        for sensor in self.sensor_link_ids:
            # Obtain the sensor pose in the world frame
            position, orientation = self.get_sensor_pose(sensor)
            pts = p.getContactPoints(bodyA=object_id, bodyB=self.robot_id, linkIndexA=-1, linkIndexB=self.lnk2Idx_robot[sensor], physicsClientId=self.cid)
            if len(pts) > 0:
                for pt in pts:
                    normal_direction = changeFrame(pt[7], orientation)
                    shear_direction_1 = changeFrame(pt[11], orientation)
                    shear_direction_2 = changeFrame(pt[13], orientation)
                    total_force_x += normal_direction[0]*pt[9] + shear_direction_1[0]*pt[10] + shear_direction_2[0]*pt[12]
                    total_force_y += normal_direction[1]*pt[9] + shear_direction_1[1]*pt[10] + shear_direction_2[1]*pt[12]
                    total_force_z += normal_direction[2]*pt[9] + shear_direction_1[2]*pt[10] + shear_direction_2[2]*pt[12]

                total_force_x = float(total_force_x/len(pts))
                total_force_y = float(total_force_y/len(pts))
                total_force_z = float(total_force_z/len(pts))

        self.force_x_buffer.append(total_force_x/len(self.sensor_link_ids))
        self.force_y_buffer.append(total_force_y/len(self.sensor_link_ids))
        self.force_z_buffer.append(total_force_z/len(self.sensor_link_ids))

        self.force_x = np.mean(self.force_x_buffer)
        self.force_y = np.mean(self.force_y_buffer)
        self.force_z = np.mean(self.force_z_buffer)

        return self.force_x, self.force_y, self.force_z

