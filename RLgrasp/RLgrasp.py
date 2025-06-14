import cv2
import gc
import gymnasium as gym
from io import BytesIO
import mujoco
import matplotlib.pyplot as plt
import numpy as np
import os
import time
# 从上一级目录导入DexHandEnv类
from dexhand.dexhand import DexHandEnv
import metric.labels as labels
import metric.metrics as metrics
import RLgrasp.utils as utils

class RLGraspEnv(DexHandEnv):
    def __init__(self):
        """
        RLGraspEnv is an implementation of the DexHandEnv + RL API + multiobject scene engineed by Mujoco, with API formulated based on Gym.
        RLGraspEnv rewrites the following important methods:
        - step(): Conduct an action in the form of (translation, rotation).
        - get_observation(): Get the hand obsrvation and the visual observation.
        - reset(): Resample the object posture and grasping scene.
        """
        super().__init__(model_dir="RLgrasp", render_mode="human")
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)  # Action space is a 7D vector
        
    def step(self, action):
        """
        The action is in the form of a 6D vector:
        1. r, beta (0 ~ 2 * pi): grasping point in the manifold of unit disk
        2. depth factor: the grasping point depth is among the min depth and the max depth of the object depth image.
        3. theta (0 ~ pi / 2), phi (0 ~ 2 * pi), alpha (0 ~ 2 * pi): the rotation angles of the approach vector
        4. grasp force (0 ~ 3): the force applied to the object during grasping.
        
        The action is conducted by three steps to avoiding the hand from being stuck in the object:
        1. Set the target approach position and target rotation.
        2. Move to the target grasp position.
        3. Apply the grasping force to the object.
        4. Lift the object to a certain height.
        5. If the contact flag is True, then drop and release the object.
        
        The DexHandEnv is controlled by relative motion, and the RLGraspEnv is controlled by absolute target grasping position.

        :return: A 5-item tuple containing the observation, reward, done flag, truncated flag and info dictionary.
        """
        hand_offsest = 0.035
        approach_offset = 0.4  # The offset distance from the grasp point to the approach position
        lift_height = 0.03

        depth_image = self.get_observation()["depth"]
        approach_pos, target_rot, target_pos, target_force = utils.transform_action(action, depth_image, hand_offsest, approach_offset)
        
        # Step 1: Set the target approach position and target rotation
        self.mj_data.qpos[0:3] = approach_pos
        self.mj_data.qpos[3:6] = target_rot
        self.mj_data.qvel[:] = 0.0
        super().step(np.zeros(7), sleep=True)  # wait for the object to drop on the floor
        # Step 2: Move to the target grasp position
        super().step(np.concatenate([target_pos - approach_pos, np.zeros(3), np.zeros(1)]))
        # Step 3: Apply the grasping force to the object
        super().step(np.concatenate([np.zeros(3), np.zeros(3), np.array([target_force])]))
        # Step 4: Lift the object to a certain height
        super().step(np.concatenate([np.array([0, 0, lift_height]), np.zeros(3), np.array([target_force])]))

        # calculate the relative feedback
        observation = self.get_observation()
        reward = self.compute_reward()
        done = self.is_done()
        truncated = False
        info = {}

        # # Step 5: If the contact flag is True, then drop and release the object
        # if info["contact"] == True:
        #     super().step(np.concatenate([np.array([0, 0, -lift_height]), np.zeros(3), np.array([target_force])]))
        #     super().step(np.concatenate([np.zeros(3), np.zeros(3), np.array([-1])]))
        
        # # Collect the garbage to avoid memory leak
        # gc.collect()

        return observation, reward, done, truncated, info
        





