from cad.grasp_sampling import sample_grasp_point, sample_grasp_quat, sample_grasp_depth, sample_grasp_collision, initialize_gripper, visualize_grasp
import cv2
import matplotlib.pyplot as plt
from Memory_Grasp.memory_grasp_env import MemoryGraspEnv
from Memory_Grasp.server import send_image_to_server
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
SERVER_IP = "183.173.80.82"
OBJECT_SCALES = {'005': 1.0, '010': 0.7}


class MemoryGraspAgent:
    """
    The GraspMemory contains the scene register, the database search, and the data storage.
    coordinate frames definition:
        - anchor frame: the frame of the anchor mesh model
        - camera frame: the camera frame of the RGB-D sensor
        - base frame: the robot base frame
    """
    def __init__(self):
        self.camera2base = np.eye(4)  # camera2base transformation matrix
        self.env = MemoryGraspEnv(render_mode="human", scene_id=5)
        self.env.reset(switch_scene=False)
        self.anchor_candidate_actions = generate_candidate_actions(num_samples=300)
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
            os.mkdir(os.path.dirname(memory_path))
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
        if len(valid_idx) == 0:
            print("No valid memory found. Choosing a best-reward action from all memory.")
            action_id = np.argmax(memory[:, -1])
        else:
            top_k = min(top_k, len(valid_idx))
            top_k_idx = np.argsort(memory[valid_idx, -1])[-top_k:]
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
        depth = (self.env.mj_renderer_depth.render() * 1000).astype(np.float32)
        # TODO: adjust the relative params.
        
        return color, depth

    def get_object_name(self):
        # Use CLIP to recognize the object
        color, depth = self.get_imgs()
        object_name = send_image_to_server(color, None, None, None, SERVER_IP, server_name="clip")
        if object_name is None:
            print("Error in CLIP client.")
            return None

        return object_name
    
    def set_anchor(self, object_name):
        # Use Any6D to set the anchor frame
        color, depth = self.get_imgs()
        anchor2camera = send_image_to_server(color, depth, object_name, "anchor", SERVER_IP, "any6d")
        if anchor2camera is None:
            print("Error in Any6D client.")
            return None
        pass

    def get_anchor2base(self, object_name):
        # get ground truth from simulator
        translation, rotation_wxyz = self.env.mj_data.qpos[8:11], self.env.mj_data.qpos[11:15]
        rotation_matrix = R.from_quat(np.array([rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]])).as_matrix()
        anchor2camera = np.eye(4)
        anchor2camera[0:3, 0:3] = rotation_matrix
        anchor2camera[0:3, 3] = translation

        # # Use Any6D to get the object pose
        # color, depth = self.get_imgs()
        # anchor2camera = send_image_to_server(color, depth, object_name, "query", SERVER_IP, "any6d")
        # if anchor2camera is None:
        #     print("Error in Any6D client.")
        #     return None

        anchor2base = np.dot(self.camera2base, anchor2camera)

        return anchor2base
    
    def transform_action(self, actions, anchor2base):
        grasp_poses, grasp_quats, grasp_forces = actions[:, 0:3], actions[:, 3:7], actions[:, 7:8]
        grasp_poses = np.dot(anchor2base[0:3, 0:3], grasp_poses.T).T + anchor2base[0:3, 3]

        grasp_mats = R.from_quat([grasp_quats[i] for i in range(grasp_quats.shape[0])]).as_matrix()  # convert to rotation matrix
        grasp_mats = anchor2base[0:3, 0:3] @ grasp_mats  # R_new = R_ab * R_old
        grasp_quats = R.from_matrix([grasp_mats[i] for i in range(grasp_mats.shape[0])]).as_quat()  # convert back to quaternion
        actions = np.hstack((grasp_poses, grasp_quats, grasp_forces))

        return actions

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

        anchor2base = self.get_anchor2base(object_name)  # get the object pose
        candidate_actions = self.transform_action(self.anchor_candidate_actions.copy(), anchor2base)  # transform candidate actions to current object pose
        action_from = "random"
        
        memory_path = os.path.join("Memory_Grasp/memory", f"{object_name}_memory.npy")
        anchor_memory = self.load_memory(memory_path)
        if anchor_memory is None:  # no memory yet
            print(f"No memory found for {object_name}. Setting anchor frame and initializing memory.")
            # self.set_anchor(object_name)
            action_id = self.choose_action_from_random(candidate_actions)
            action = candidate_actions[action_id]
        
        else:  # already have memory
            memory = anchor_memory.copy()
            memory[:, :-1] = self.transform_action(anchor_memory[:, :-1].copy(), anchor2base)  # transform memory to current object pose
            memory_mat = R.from_quat(memory[:, 3:7]).as_matrix()
            valid_memory_idx = np.where(memory_mat[:, 2, 2] > 0.97)[0]
            # print(f"Loaded memory for {object_name}, shape: {memory.shape}.")
            
            if np.random.random() < -1.0 + np.max(memory[:, -1]) and len(valid_memory_idx) > 0:  # use max reward to adjust the greedy threshold
                action_id = self.choose_action_from_memory(memory, top_k=5)
                action = memory[action_id]
                action_from = "memory"
                print(f"Chose action from memory.")
            else:
                action_id = self.choose_action_from_random(candidate_actions)
                action = candidate_actions[action_id]
                print(f"Chose action from random.")

        observation, reward, done, truncated, info = self.env.step(action)
        print(f"Executed action, received reward: {reward}, done: {done}, truncated: {truncated}.")
        anchor2base = self.get_anchor2base(object_name)  # get the updated object pose
        anchor_action = self.transform_action(action[np.newaxis, :].copy(), np.linalg.inv(anchor2base))[0]  # transform action to anchor frame
        
        if action_from == "memory" and reward == -1.0:
            print(f"Warning: action from memory resulted in failure. Delete this memory.")
            anchor_memory = np.delete(anchor_memory, action_id, axis=0)
            self.save_memory(anchor_memory, None, reward, memory_path)

        if action_from == "random" and reward > -1.0:
            print(f"Saved updated memory for {object_name} with reward: {reward}.")
            self.save_memory(anchor_memory, anchor_action, reward, memory_path)
        
        if reward > -1.0:
            print(f"Successful grasp! Resetting the environment.")
            self.env.reset(switch_scene=False)

def generate_candidate_actions(num_samples=500, OBJECT_ID="005"):
    # Load the point cloud
    try:
        file_path = f"cad/assets/{OBJECT_ID}/downsampled.ply"  # Replace with your point cloud file path
        point_cloud = o3d.io.read_point_cloud(file_path)
        scale = OBJECT_SCALES.get(OBJECT_ID, 1.0)
        point_cloud.scale(scale, center=point_cloud.get_center())
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    # Sample grasp points, normals, and depths
    grasp_points, grasp_quats, grasp_depths = [], [], []
    while len(grasp_points) < num_samples:
        try:
            grasp_points_sample = sample_grasp_point(point_cloud, 10 * num_samples) # 根据经验，每次采样10倍的数量
            grasp_quats_sample = sample_grasp_quat(10 * num_samples)
            grasp_depths_sample = sample_grasp_depth(10 * num_samples, min_depth=-1e-2, max_depth=-1e-3)
            grasp_collisions_sample = sample_grasp_collision(point_cloud, grasp_points_sample, grasp_quats_sample, grasp_depths_sample, initialize_gripper())
            print(f"Sampled {10 * num_samples} grasps, with {sum(grasp_collisions_sample)} collisions detected.")
            grasp_points.extend(grasp_points_sample[np.logical_not(grasp_collisions_sample)])
            grasp_quats.extend(grasp_quats_sample[np.logical_not(grasp_collisions_sample)])
            grasp_depths.extend(grasp_depths_sample[np.logical_not(grasp_collisions_sample)])            
        except Exception as e:
            print(f"Error sampling grasps: {e}")
            return

    # # Visualize the sampled grasps
    # try:
    #     initial_gripper = initialize_gripper()
    #     visualize_grasp(point_cloud, grasp_points, grasp_quats, grasp_depths, initial_gripper)
    # except Exception as e:
    #     print(f"Error visualizing grasps: {e}")
    #     return
    
    grasp_mats = R.from_quat(grasp_quats).as_matrix()
    grasp_poses = grasp_points + np.array(grasp_depths)[:, np.newaxis] * grasp_mats[:, :, 2]  # move along the grasp normal direction
    candidate_actions = np.hstack((grasp_poses, grasp_quats, 10.0 * np.ones((len(grasp_poses), 1))))

    return candidate_actions

if __name__ == "__main__":
    agent = MemoryGraspAgent()
    
    for i in range(50):
        print(f"--- Running turn {i+1} ---")
        agent.run_one_turn()
        
    # color, depth = agent.get_imgs()
    # print(f"Color image shape: {color.shape}, Depth image shape: {depth.shape}.")
    # cv2.imwrite("./Memory_Grasp/results/debug/color.png", color)
    # cv2.imwrite("./Memory_Grasp/results/debug/depth.png", depth)

