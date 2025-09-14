import cv2
import matplotlib.pyplot as plt
from Memory_Grasp.memory_grasp_env import MemoryGraspEnv
from Memory_Grasp.server import send_image_to_server
from Memory_Grasp.utils import camera2world, generate_candidate_actions, transform_action
import numpy as np
T_CAMERA2WORLD = camera2world()

import os
from scipy.spatial.transform import Rotation as R
SERVER_IP = "183.173.80.82"


class MemoryGraspAgent:
    """
    The GraspMemory contains the scene register, the database search, and the data storage.
    coordinate frames definition:
        - anchor frame: the frame of the anchor mesh model
        - camera frame: the camera frame of the RGB-D sensor
        - base frame: the robot base frame
    """
    def __init__(self):
        self.env = MemoryGraspEnv(render_mode="human", scene_id=5)
        self.env.reset(switch_scene=False)
        self.anchor_candidate_actions = generate_candidate_actions(num_samples=300)
        self.reward, self.done = -1.0, False
        pass

    def load_memory(self, memory_path):
        try:
            memory = np.load(memory_path, allow_pickle=True)
        except:
            print(f"Error in loading memory from {memory_path}")
            memory = None
        
        return memory
    
    def save_memory(self, memory, anchor_action, reward, memory_path):
        if memory is None:  # no memory yet
            os.makedirs(os.path.dirname(memory_path), exist_ok=True)
            memory = np.array([np.hstack((anchor_action, reward))])
        elif anchor_action is None:  # just save the existing memory
            pass
        else:  # already have memory
            memory = np.vstack((memory, np.hstack((anchor_action, reward))))
        np.save(memory_path, memory)
        pass
        
    def choose_action_from_memory(self, memory, top_k=5):
        # Choose the action with the highest reward from memory with memory[:, 5] > 0.97
        memory_mat = R.from_quat(memory[:, 3:7]).as_matrix()
        valid_idx = np.where(memory_mat[:, 2, 2] > 0.97)[0]
        top_k = min(top_k, len(valid_idx))
        top_k_idx = np.argsort(memory[valid_idx, -1])[-top_k:]
        # The reward value is not very well distributed, so we randomly choose one from the top_k
        action_id = valid_idx[top_k_idx[np.random.randint(0, top_k)]]

        return action_id
    
    def choose_action_from_random(self, candidate_actions):
        # choose a random action_id from candidate_actions with candidate_actions[:, 5] > 0.97
        candidate_actions_mat = R.from_quat(candidate_actions[:, 3:7]).as_matrix()
        valid_idx = np.where(candidate_actions_mat[:, 2, 2] > 0.97)[0]
        if len(valid_idx) == 0:
            print("No valid candidate actions found. Choosing a random action from all candidates.")
            action_id = np.random.randint(0, len(candidate_actions))
        else:
            action_id = valid_idx[np.random.randint(0, len(valid_idx))]
        
        return action_id
    
    def get_imgs(self):
        self.env.mj_renderer_rgb.update_scene(self.env.mj_data, camera="main")
        self.env.mj_renderer_depth.update_scene(self.env.mj_data, camera="main")
        color = self.env.mj_renderer_rgb.render().astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = (self.env.mj_renderer_depth.render() * 1000).astype(np.uint16)
        
        return color, depth

    def get_object_name(self):
        # Use CLIP to recognize the object
        color, depth = self.get_imgs()
        success, encoded_image = cv2.imencode('.png', color)
        if success:
            color_bytes = encoded_image.tobytes()
        else:
            color_bytes = None
        object_name = send_image_to_server(color_bytes, None, None, None, SERVER_IP, server_name="clip")

        return object_name
    
    def set_anchor(self, object_name):
        # Use Any6D to set the anchor frame
        color, depth = self.get_imgs()
        # save the images
        cv2.imwrite("./Memory_Grasp/results/debug/color.png", color)
        cv2.imwrite("./Memory_Grasp/results/debug/depth.png", depth)
        with open("./Memory_Grasp/results/debug/color.png", "rb") as f:
            color_bytes = f.read()
        with open("./Memory_Grasp/results/debug/depth.png", "rb") as f:
            depth_bytes = f.read()
        
        # Use Any6D to set the anchor frame
        task = "anchor"
        anchor2camera = send_image_to_server(color_bytes, depth_bytes, object_name.encode('utf-8'), task.encode('utf-8'), SERVER_IP, "any6d")
        anchor2base = T_CAMERA2WORLD @ anchor2camera

        return anchor2base

    def get_anchor2base(self, object_name):
        # # get ground truth from simulator
        # translation, rotation_wxyz = self.env.mj_data.qpos[8:11], self.env.mj_data.qpos[11:15]
        # rotation_matrix = R.from_quat(np.array([rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]])).as_matrix()
        # anchor2base = np.eye(4)
        # anchor2base[0:3, 0:3] = rotation_matrix
        # anchor2base[0:3, 3] = translation

        # Use Any6D to get the object pose
        color, depth = self.get_imgs()
        # save the images
        cv2.imwrite("./Memory_Grasp/results/debug/color.png", color)
        cv2.imwrite("./Memory_Grasp/results/debug/depth.png", depth)
        with open("./Memory_Grasp/results/debug/color.png", "rb") as f:
            color_bytes = f.read()
        with open("./Memory_Grasp/results/debug/depth.png", "rb") as f:
            depth_bytes = f.read()
        object_name = "banana"#send_image_to_server(color_bytes, None, None, None, SERVER_IP, server_name="clip")
        
        # Use Any6D to set the anchor frame
        task = "query"
        anchor2camera = send_image_to_server(color_bytes, depth_bytes, object_name.encode('utf-8'), task.encode('utf-8'), SERVER_IP, "any6d")
        anchor2base = T_CAMERA2WORLD @ anchor2camera

        return anchor2base

    def get_action_from(self, object_name, anchor2base, reward, done):      
        memory_path = os.path.join("Memory_Grasp/memory", f"{object_name}_memory.npy")
        self.anchor_memory = self.load_memory(memory_path)
        if self.anchor_memory is None or self.anchor_memory.shape[0] == 0:  # no memory yet
            print(f"No memory found for {object_name}. Setting anchor frame and initializing memory.")
            # if reward > -1.0 and (not done):
            #     return "recovery"
            return "random"
        
        memory = self.anchor_memory.copy()
        memory[:, :-1] = transform_action(self.anchor_memory[:, :-1].copy(), anchor2base)  # transform memory to current object pose
        memory_mat = R.from_quat(memory[:, 3:7]).as_matrix()
        valid_memory_idx = np.where(memory_mat[:, 2, 2] > 0.97)[0]
        if np.random.random() < 1.3 + np.max(memory[:, -1]) and len(valid_memory_idx) > 0:  # use max reward to adjust the greedy threshold
            return "memory"
        # if reward > -1.0 and (not done):
        #     return "recovery"   
        return "random"

    def run_one_turn(self):
        """
        A decision-interaction step:
        1. recognize the object_name
        2. if no memory, set anchor frame and initialize memory
        3. load memory
        4. choose action
        5. execute action and get reward
        6. save memory
        """
        object_name = "test_object"  # TODO: self.tracker.recognize_object()
        # object_name = self.get_object_name()
        print(f"Recognized object: {object_name}.")
        memory_path = os.path.join("Memory_Grasp/memory", f"{object_name}_memory.npy")

        anchor2base = self.get_anchor2base(object_name)  # get the object pose
        action_from = self.get_action_from(object_name, anchor2base, self.reward, self.done)
        print(f"Choosing action from: {action_from}.")
        
        if action_from == "random":
            candidate_actions = transform_action(self.anchor_candidate_actions.copy(), anchor2base)  # transform candidate actions to current object pose
            action_id = self.choose_action_from_random(candidate_actions)
            action = candidate_actions[action_id]
        
        elif action_from == "memory":
            memory = self.anchor_memory.copy()
            memory[:, :-1] = transform_action(self.anchor_memory[:, :-1].copy(), anchor2base)  # transform memory to current object pose
            action_id = self.choose_action_from_memory(memory, top_k=5)
            action = memory[action_id]
        
        elif action_from == "recovery":
            tactile = self.observation.reshape(3, 20, 20)  # Reshape 1200 to 3*20*20
            X, Y = np.meshgrid(np.arange(20), np.arange(20))
            Fx, Fy, Fz = tactile[1, ...], tactile[2, ...], tactile[0, ...]
            torque_z = np.sum(X.flatten() * Fy.flatten() - Y.flatten() * Fx.flatten())
            grasp_mat = R.from_quat(self.action[3:7]).as_matrix()
            action = self.action.copy()
            action[:3] = action[:3] + grasp_mat[:, 1] * np.sign(torque_z) * 0.01  # move along the lateral direction of the gripper
            action_id = -1  # no action_id for recovery action

        observation, reward, done, truncated, info = self.env.step(action)
        self.action, self.observation, self.reward, self.done = action, observation, reward, done
        print(f"Executed {action}, received reward: {reward}, done: {done}.")
        
        if done:
            if action_from == "random" or action_from == "recovery":
                anchor2base = self.get_anchor2base(object_name)  # get the updated object pose
                anchor_action = transform_action(action[np.newaxis, :].copy(), np.linalg.inv(anchor2base))[0]  # transform action to anchor frame
                print(f"Saved updated memory for {object_name} with reward: {reward}.")
                self.save_memory(self.anchor_memory, anchor_action, reward, memory_path)
            print(f"Successful grasp! Resetting the environment.")
            self.env.reset(switch_scene=False)
        else:
            if action_from == "memory":
                print(f"Warning: action from memory resulted in failure. Delete this memory.")
                self.anchor_memory = np.delete(self.anchor_memory, action_id, axis=0)
                self.save_memory(self.anchor_memory, None, reward, memory_path)
        
        return action_from, done, self.anchor_memory.shape[0] if self.anchor_memory is not None else 0


if __name__ == "__main__":
    agent = MemoryGraspAgent()
    # action_from_list, done_list, memory_size_list = [], [], []
    # for i in range(500):
    #     print(f"--- Running turn {i+1} ---")
    #     action_from, done, memory_size = agent.run_one_turn()
    #     action_from_list.append(action_from)
    #     done_list.append(done)
    #     memory_size_list.append(memory_size)
    
    # action_from_map = {"random":0.0, "recovery":1.0, "memory":2.0}
    # action_from_list = [action_from_map[a] for a in action_from_list]
    # data = np.array([action_from_list, done_list, memory_size_list])
    # np.savetxt("./Memory_Grasp/results/debug/log_any6d.txt", data.T)


    data = np.loadtxt("./Memory_Grasp/results/debug/log_any6d.txt")
    random_flag, recovery_flag, memory_flag = (data[:, 0] == 0.0), (data[:, 0] == 1.0), (data[:, 0] == 2.0)
    interval = 50
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot([np.mean(random_flag[interval * i : interval * (i + 1)]) for i in range(data.shape[0] // interval)], label="Random rate")
    # plt.plot([np.mean(recovery_flag[interval * i : interval * (i + 1)]) for i in range(data.shape[0] // interval)], label="Recovery rate")
    plt.plot([np.mean(memory_flag[interval * i : interval * (i + 1)]) for i in range(data.shape[0] // interval)], label="Memory rate")
    plt.ylabel("Action source rate")
    plt.legend()
    plt.subplot(1, 3, 2)
    rate_1 = np.sum(data[:, 1][random_flag]) / np.sum(random_flag)
    rate_2 = np.sum(data[:, 1][recovery_flag]) / np.sum(recovery_flag)
    rate_3 = np.sum(data[:, 1][memory_flag]) / np.sum(memory_flag)
    plt.plot([np.nonzero(data[interval * i : interval * (i + 1), 1])[0].shape[0] / interval for i in range(data.shape[0] // interval)])
    plt.axhline(y=rate_1, color='r', linestyle='--', label=f"Random success rate: {rate_1:.2f}")
    # plt.axhline(y=rate_2, color='b', linestyle='--', label=f"Recovery success rate: {rate_2:.2f}")
    plt.axhline(y=rate_3, color='g', linestyle='--', label=f"Memory success rate: {rate_3:.2f}")
    plt.legend()
    plt.ylabel("Success rate")
    plt.subplot(1, 3, 3)
    plt.plot(np.mean(data[:interval * (data.shape[0] // interval), 2].reshape(-1, interval), axis=1))
    plt.ylabel("Memory size")
    plt.tight_layout()
    plt.savefig("./Memory_Grasp/results/debug/log_any6d.png")
    plt.show()
        
    # color, depth = agent.get_imgs()
    # print(f"Color image shape: {color.shape}, Depth image shape: {depth.shape}.")
    # cv2.imwrite("./Memory_Grasp/results/debug/color.png", color)
    # cv2.imwrite("./Memory_Grasp/results/debug/depth.png", depth)

