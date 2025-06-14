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
import RL.utils as utils

class RLGraspEnv(DexHandEnv):
    def __init__(self):
        """
        RLGraspEnv is an implementation of the DexHandEnv + RL API + multiobject scene engineed by Mujoco, with API formulated based on Gym.
        RLGraspEnv rewrites the following important methods:
        - step(): Conduct an action in the form of (translation, rotation).
        - get_observation(): Get the hand obsrvation and the visual observation.
        - reset(): Resample the object posture and grasping scene.
        """
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  # Action space is a 6D vector
        
    def step(self, action):
        """
        The action is in the form of a 6D vector:
        1. r, beta (0 ~ 2 * pi): grasping point in the manifold of unit disk
        2. depth factor: the grasping point depth is among the min depth and the max depth of the object depth image.
        3. theta (0 ~ pi / 2), phi (0 ~ 2 * pi): the rotation angles of the approach vector
        4. grasp force (0 ~ 3): the force applied to the object during grasping.
        
        The action is conducted by three steps to avoiding the hand from being stuck in the object:
        1. Move to the target approach position.
        2. Rotate the hand to the target rotation.
        3. Move to the target grasp position.
        4. Apply the grasping force to the object.
        5. Lift the object to a certain height.
        
        The DexHandEnv is controlled by relative motion, and the RLGraspEnv is controlled by absolute target grasping position.

        :return: A 5-item tuple containing the observation, reward, done flag, truncated flag and info dictionary.
        """
        approach_offset = 0.2  # The offset distance from the grasp point to the approach position
        lift_height = 0.03
        current_pos = self.mj_data.qpos[:3].copy()
        current_rot = self.mj_data.qpos[3:7].copy()

        depth_image = self.get_observation()["depth"]
        approach_pos, target_rot, target_pos, target_force = utils.transform_action(action, depth_image, approach_offset)
        
        # Step 1: Move to the target approach position
        super().step(np.concatenate([approach_pos - current_pos, np.zeros(3), np.zeros(1)]))
        # Step 2: Rotate the hand to the target rotation
        target_rot = np.array([0, 0, 0, 1])
        super().step(np.concatenate([np.zeros(3), target_rot - current_rot, np.zeros(1)]))
        # Step 3: Move to the target grasp position
        super().step(np.concatenate([target_pos - approach_pos, np.zeros(3), np.zeros(1)]))
        # Step 4: Apply the grasping force to the object
        super().step(np.concatenate([np.zeros(3), np.zeros(3), np.array([target_force])]))
        # Step 5: Lift the object to a certain height
        super().step(np.concatenate([np.array([0, 0, lift_height]), np.zeros(3), np.array([target_force])]))

        observation = self.get_observation()
        reward = self.compute_reward(observation)
        done = self.is_done(observation)
        truncated = False
        info = {}





