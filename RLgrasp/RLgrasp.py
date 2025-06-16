import cv2
import gc
import gymnasium as gym
from gymnasium import spaces
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
        - compute_reward(): Calculate the reward based on the contact state of the object.
        - reset(): Resample the object posture and grasping scene.
        """
        super().__init__(model_dir="RLgrasp", render_mode="human")
        self.observation_space = spaces.Dict({
            "history_observation": spaces.Dict({
                "depth": spaces.Box(low=0, high=1, shape=(640, 480, 3), dtype=np.float32),
                "action": spaces.Box(low=-1, high=1, shape=(7, ), dtype=np.float32),
            }),
            "current_observation": spaces.Box(low=0, high=1, shape=(640, 480,), dtype=np.float32)
        })
        self.action_buffer = []  # Buffer to store the action history
        self.max_attempts = 10  # Maximum number of attempts to grasp the object
        
    def reset(self):
        _ = super().reset()
        self.action_buffer = []  # Clear the action history buffer
        super().step(np.zeros(7), sleep=True, add_frame=True)  # wait for the object to drop on the floor

        return self.get_observation()
    
    def step(self, action):
        """
        The action is in the form of a 6D vector:
        1. r, beta (0 ~ 2 * pi): grasping point in the manifold of unit disk
        2. depth factor (0 ~ 1): the grasping point depth is among the min depth and the max depth of the object depth image.
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
        hand_offsest = 0.165 + 0.01  # 0.01 for more stable grasping
        approach_offset = 0.4  # The offset distance from the grasp point to the approach position
        lift_height = 0.03

        # Step 0: get the depth_image with object segmentation mask.
        depth_image = self.episode_buffer["depth"][-1]
        segmentation_mask = self.episode_buffer["segmentation"][-1][..., 0]
        approach_pos, target_rot, target_pos, target_force = utils.transform_action(action, depth_image, segmentation_mask, hand_offsest, approach_offset)
        
        # Step 1: Set the target approach position and target rotation
        self.mj_data.qpos[0:3] = approach_pos
        self.mj_data.qpos[3:6] = target_rot
        self.mj_data.qvel[:] = 0.0
        # Step 2: Move to the target grasp position
        super().step(np.concatenate([target_pos - approach_pos, np.zeros(3), np.zeros(1)]))
        # Step 3: Apply the grasping force to the object
        super().step(np.concatenate([np.zeros(3), np.zeros(3), np.array([target_force])]))
        # Step 4: Lift the object to a certain height
        super().step(np.concatenate([np.array([0, 0, lift_height]), np.zeros(3), np.array([target_force])]), add_frame=True)

        # calculate the relative feedback
        reward, done, truncated = self.compute_reward()
        info = {}

        # Step 5: If the contact flag is True, then drop the object
        if reward >= -0.5:
            super().step(np.concatenate([np.array([0, 0, -lift_height]), np.zeros(3), np.array([target_force])]))
        
        # Step 6: Set the hand to the initial qpos and get the next observation (image).
        self.mj_data.qpos[0:6] = 0  # Reset the joint positions to zero
        self.mj_data.qpos[2] = 0.5  # Set the hand to a certain height
        self.mj_data.qvel[0:6] = 0  # Reset the joint velocities to zero
        super().step(np.concatenate([np.zeros(3), np.zeros(3), np.array([-10])]))
        super().step(np.concatenate([np.zeros(3), np.zeros(3), np.zeros(1)]), add_frame=True)
        observation = self.get_observation()
        
        return observation, reward, done, truncated, info
        
    def get_observation(self):
        """
        The observation is the accumulated version of the episode buffer + action history.
        The structure of the episode buffer and action is:
        [0, visual, 0, visual, 0, ...], [0, 0, tactile, 0, tactile, ...], [action, action, ...]
        After the Nth attempt, the length of the episode buffer is 2N + 1, and the length of the action history is N.
        :return: A dictionary with the length of N + 1. N represent the history information, and 1 is the current visual information.
        """
        history_observation = {"depth":self.episode_buffer["depth"][1:-1:2], 
                               "action":self.action_buffer}
        current_observation = self.episode_buffer["depth"][-1]

        return {"history_observation": history_observation, 
                "current_observation": current_observation}


    def compute_reward(self):
        """
        Multistage reward function:
        1. if the object is in not in contact with the hand, then the reward is -1.0.
        2. if the object is in contact with the hand and the floor, then the reward is -0.5.
        3. if the object is in contact with the hand and not the floor, then the reward is 0.0.
        """
        contact_hand, contact_floor = labels.contact_labels(self)
        if not contact_hand:
            reward = -1.0
        elif contact_hand and contact_floor:
            reward = -0.5
        else:
            reward = 0.0
        
        truncated = len(self.episode_buffer["rgb"]) >= 2 * self.max_attempts
        done = (reward == 0.0) or truncated
        
        return reward, done, truncated






