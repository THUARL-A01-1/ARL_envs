from cad.grasp_sampling import sample_grasp_point, sample_grasp_normal, sample_grasp_angle, sample_grasp_depth, sample_grasp_collision, initialize_gripper, visualize_grasp
from Memory_Grasp.memory_grasp_env import MemoryGraspEnv
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R


class MemoryGraspAgent:
    """
    The GraspMemory contains the scene register, the database search, and the data storage.
    coordinate frames definition:
        - anchor frame: the frame of the anchor mesh model
        - camera frame: the camera frame of the RGB-D sensor
        - base frame: the robot base frame
    """
    def __init__(self):
        self.tracker = None  # Any6DTracker()
        self.camera2base = np.eye(4)  # camera2base transformation matrix
        self.env = MemoryGraspEnv(render_mode="human")
        self.env.reset()
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
        else:  # already have memory
            memory = np.vstack((memory, np.hstack((anchor_action, reward))))
        np.save(memory_path, memory)
        pass
        
    def choose_action_from_memory(self, memory, top_k=5):
        # Choose the action with the highest reward
        sorted_memory = memory[memory[:, -1].argsort()[::-1]]
        top_actions = sorted_memory[:top_k, :-1]
        chosen_action = top_actions[np.random.randint(0, top_k)]

        return chosen_action
    
    def choose_action_from_random(self, candidate_actions):
        chosen_action = candidate_actions[np.random.randint(0, len(candidate_actions))]
        
        return chosen_action
    
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
        print(f"Recognized object: {object_name}.")
        
        memory_path = os.path.join("memory", f"{object_name}_memory.npy")
        memory = self.load_memory(memory_path)
        if memory is None:  # no memory yet
            print(f"No memory found for {object_name}. Setting anchor frame and initializing memory.")
            # self.tracker.set_anchor(object_name)
            self.candidate_actions = generate_candidate_actions(num_samples=30)
            anchor_action = self.choose_action_from_random(self.candidate_actions)
        else:  # already have memory
            print(f"Loaded memory for {object_name}, shape: {memory.shape}.")
            if np.max(memory[:, -1]) > -0.7 and np.random.random() > 0.5:  # if max reward > 1.0, use memory to choose action
                anchor_action = self.choose_action_from_memory(memory, top_k=1)
            else:
                anchor_action = self.choose_action_from_random(self.candidate_actions)
        
        print(f"Chosen action in anchor frame:\n{anchor_action}.")
        anchor_grasp_pose, anchor_approach_vector, alpha, grasp_force = anchor_action[0:3], anchor_action[3:6], anchor_action[6], anchor_action[7]
        
        # TODO: anchor2camera = self.tracker.track_one_frame(object_name)
        translation, rotation_wxyz = self.env.mj_data.qpos[8:11], self.env.mj_data.qpos[11:15]
        rotation_matrix = R.from_quat(np.array([rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]])).as_matrix()
        anchor2camera = np.eye(4)
        anchor2camera[0:3, 0:3] = rotation_matrix
        anchor2camera[0:3, 3] = translation
        print(f"Tracked one frame for {object_name}, anchor2camera:\n{anchor2camera}.")

        anchor2base = np.dot(self.camera2base, anchor2camera)
        grasp_pose = anchor_grasp_pose + anchor2base[0:3, 3]
        approach_vector = np.array([0,0,1])#anchor_approach_vector @ anchor2base[0:3, 0:3]
        action = np.hstack((grasp_pose, approach_vector, alpha, grasp_force))
        print(f"Chosen action in base frame:\n{action}.")

        observation, reward, done, truncated, info = self.env.step(action)
        print(f"Executed action, received reward: {reward}, done: {done}, truncated: {truncated}.")
        
        if reward > -1.0:
            self.save_memory(memory, anchor_action, reward, memory_path)
            print(f"Saved updated memory for {object_name} with reward: {reward}.")

def generate_candidate_actions(num_samples=500, OBJECT_ID="005"):
    # Load the point cloud
    try:
        file_path = f"cad/assets/{OBJECT_ID}/downsampled.ply"  # Replace with your point cloud file path
        point_cloud = o3d.io.read_point_cloud(file_path)
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    # Sample grasp points, normals, and depths
    grasp_points, grasp_normals, grasp_angles, grasp_depths = [], [], [], []
    while len(grasp_points) < num_samples:
        try:
            grasp_points_sample = sample_grasp_point(point_cloud, 30 * num_samples) # 根据经验，每次采样30倍的数量
            grasp_normals_sample = sample_grasp_normal(30 * num_samples)
            grasp_angles_sample = sample_grasp_angle(30 * num_samples)
            grasp_depths_sample = sample_grasp_depth(30 * num_samples)
            grasp_collisions_sample = sample_grasp_collision(point_cloud, grasp_points_sample, grasp_normals_sample, grasp_angles_sample, grasp_depths_sample, initialize_gripper())
            print(f"Sampled {30 * num_samples} grasps, with {sum(grasp_collisions_sample)} collisions detected.")
            grasp_points.extend(grasp_points_sample[np.logical_not(grasp_collisions_sample)])
            grasp_normals.extend(grasp_normals_sample[np.logical_not(grasp_collisions_sample)])
            grasp_angles.extend(grasp_angles_sample[np.logical_not(grasp_collisions_sample)])
            grasp_depths.extend(grasp_depths_sample[np.logical_not(grasp_collisions_sample)])            
        except Exception as e:
            print(f"Error sampling grasps: {e}")
            return

    # # Visualize the sampled grasps
    # try:
    #     initial_gripper = initialize_gripper()
    #     visualize_grasp(point_cloud, grasp_points, grasp_normals, grasp_angles, grasp_depths, initial_gripper)
    # except Exception as e:
    #     print(f"Error visualizing grasps: {e}")
    #     return
    
    grasp_poses = grasp_points + np.array(grasp_depths)[:, np.newaxis] * np.array(grasp_normals)
    candidate_actions = np.hstack((grasp_poses, grasp_normals, np.array(grasp_angles)[:, np.newaxis], 10.0 * np.ones((len(grasp_poses), 1))))

    return candidate_actions

if __name__ == "__main__":
    agent = MemoryGraspAgent()
    for i in range(100):
        print(f"--- Running turn {i+1} ---")
        agent.run_one_turn()
