import gc
from gymnasium import spaces
import mujoco
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

# 从上一级目录导入DexHandEnv类
from dexhand.dexhand import DexHandEnv
import metric.interactions as interactions
import metric.labels as labels
import metric.metrics as metrics
import Memory_Grasp.utils as utils

class MemoryGraspEnv(DexHandEnv):
    def __init__(self, render_mode="rgb_array", grasp_mode="free", scene_range=range(50), scene_id=5):
        """
        RLGraspEnv is an implementation of the DexHandEnv + RL API + multiobject scene engineed by Mujoco, with API formulated based on Gym.
        RLGraspEnv rewrites the following important methods:
        - step(): Conduct an action in the form of (translation, rotation).
        - get_observation(): Get the hand obsrvation and the visual observation.
        - compute_reward(): Calculate the reward based on the contact state of the object.
        - reset(): Resample the object posture and grasping scene.
        - params:
        render_mode (str): The rendering mode, can be "human" or "rgb_array". Default is "rgb_array".
        grasp_mode (str): The grasp mode, can be "free" (0-3N) or "fixed_force" (3N). Default is "free".
        scene_range (list): The scene.xml range that is being randomized. e.g.: 0-50 for training and 50-88 for evaluation.
        """
        super().__init__(model_path=f"RLgrasp/scenes/{scene_id:03d}.xml", render_mode=render_mode, resolution=(480, 640))
        # self.observation_space = spaces.Dict({
        #     "history_depth": spaces.Box(low=0, high=1, shape=(1, 512, 512), dtype=np.float32),
        #     "history_action": spaces.Box(low=-1, high=1, shape=(7, ), dtype=np.float32),
        #     "current_depth": spaces.Box(low=0, high=1, shape=(1, 512, 512), dtype=np.float32)})

        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 1), dtype=np.uint8)
        self.action_buffer = []  # Buffer to store the action history
        self.max_attempts = 100  # Maximum number of attempts to grasp the object
        self.grasp_mode = grasp_mode  # Grasp mode, can be "fixed_force" or "variable_force"
        self.scene_xml_list = [f"RLgrasp/scenes/{i:03d}.xml" for i in scene_range if i not in []]
        print("RLgrasp env initialized.")
        
    def reset(self, seed=None, options=None):
        """
        The reset method of the son class will reload the model.
        """
        self.model_path = self.scene_xml_list[5]#random.choice(self.scene_xml_list)
        # self.model_path = self.scene_xml_list[1]
        # print(f"RLgrasp env reset: {self.model_path}")
        # self._release_model()  # Release the current model to avoid memory leak
        # self._load_model(self.model_path, resolution=(480, 640))  # Load a new model from the scene XML file
        super().reset()
        self.action_buffer = []  # Clear the action history buffer

        self.mj_data.qpos[8:10] = np.random.uniform(-0.2, 0.2, size=2)  # Randomly set the object position
        random_xyzw = R.random().as_quat()  # Randomly set the object orientation
        random_wxyz = np.array([random_xyzw[3], random_xyzw[0], random_xyzw[1], random_xyzw[2]])
        self.mj_data.qpos[11:15] = random_wxyz

        super().step(np.zeros(7), sleep=True, add_frame=True)  # wait for the object to drop on the floor

        return self.get_observation(), {}
    
    def step(self, action):
        """
        The action is in the form of a 8D vector:
        1. grasp_pos (3D): the target grasp position in 3D space.
        2. approach vector (3D): the target approach vector in 3D space.
        3. alpha (1D): the rotation angle around the approach vector.
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
        hand_offsest = 0.175 - 0.0001  # 0.01 for more stable grasping
        approach_offset = 0.4  # The offset distance from the grasp point to the approach position
        lift_height = 0.03

        # Step 0: get the depth_image with object segmentation mask.
        self.action_buffer.append(action)
        
        approach_pos, target_rot, target_pos, target_force = utils.encode_action(action, hand_offsest, approach_offset)
        if approach_pos is None:
            print(f"From {self.model_path}: Error in transforming action")
            return self.get_observation(), -1.0 * self.max_attempts, True, True, {}
        
        if self.grasp_mode == "fixed_force":
            target_force = 5.0
        
        # Step 1: Set the target approach position and target rotation
        self.mj_data.qpos[0:3] = approach_pos
        self.mj_data.qpos[2] -= 0.5  # The origin of the hand has a height of 0.5, so we need to lower the hand to the ground level
        self.mj_data.qpos[3:6] = target_rot
        self.mj_data.qvel[:] = 0.0
        # Step 2: Move to the target grasp position
        super().step(np.concatenate([target_pos - approach_pos, np.zeros(3), np.zeros(1)]))
        # Step 3: Apply the grasping force to the object
        super().step(np.concatenate([np.zeros(3), np.zeros(3), np.array([target_force])]))
        # Step 4: Lift the object to a certain height
        super().step(np.concatenate([np.array([0, 0, lift_height]), np.zeros(3), np.array([target_force])]), add_frame=False)

        # calculate the relative feedback
        reward, done, truncated = self.compute_reward()
        info = {}

        # # Step 5: If the contact flag is True, then drop the object
        # if reward > -1.0:
        #     super().step(np.concatenate([np.array([0, 0, -lift_height]), np.zeros(3), np.array([target_force])]))  # drop the hand
            # super().step(np.concatenate([np.array([0, 0, -lift_height]), np.zeros(3), np.array([-target_force])]))  # release the hand
            # super().step(np.concatenate([approach_pos - target_pos, np.zeros(3), np.zeros(1)]))  # return to the approach pos
        
        # Step 6: Set the hand to the initial qpos and get the next observation (image).
        self.mj_data.qpos[0:8] = 0  # Reset the joint positions to zero, including the finger drivers
        self.mj_data.qvel[0:8] = 0  # Reset the joint velocities to zero, including the finger drivers
        # super().step(np.concatenate([np.zeros(3), np.zeros(3), np.array([-10])]))
        super().step(np.concatenate([np.zeros(3), np.zeros(3), np.zeros(1)]), add_frame=True)
        observation = self.get_observation()

        # print(f"reward: {reward}, done: {done}, truncated: {truncated}")
        # full_size = asizeof.asizeof(self)  # 包含所有嵌套对象
        # print(f"完整内存占用: {full_size / 1024:.2f} KB")  # 如CartPole约20KB，Atari可达50MB
        gc.collect()
        
        return observation, reward, done, truncated, info

    def close(self):
        self._release_model()
        self.episode_buffer, self.joint_dict, self.actuator_dict, self.action_buffer, self.scene_xml_list = None, None, None, None, None
        gc.collect()

    def get_observation(self):
        """
        The observation is the accumulated version of the episode buffer + action history.
        The structure of the episode buffer and action is:
        [0, visual, 0, visual, 0, ...], [0, 0, tactile, 0, tactile, ...], [action, action, ...]
        After the Nth attempt, the length of the episode buffer is 2N + 1, and the length of the action history is N.
        :return: A dictionary with the length of N + 1. N represent the history information, and 1 is the current visual information.
        """
        # observation = {"history_depth":np.array(self.episode_buffer["depth"][1:-1:2]), "history_action":np.array(self.action_buffer), "current_depth":np.array(self.episode_buffer["depth"][-1])}

        return np.array(self.episode_buffer["depth"][-1])#observation["current_depth"]


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
        # elif contact_hand and contact_floor:
        #     reward = -0.5
        # else:
        #     reward = 0.0
        else:
            measurement = interactions.measure(self)
            our_metric, Fv = metrics.calculate_our_metric(measurement)
            reward = -np.mean(our_metric)
        
        truncated = len(self.episode_buffer["depth"]) >= self.max_attempts + 2
        done = (contact_hand and (not contact_floor)) or truncated
        
        return reward, done, truncated






