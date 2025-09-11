from cad.grasp_sampling import sample_grasp_point, sample_grasp_quat, sample_grasp_depth, sample_grasp_collision, initialize_gripper, visualize_grasp
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
OBJECT_SCALES = {'005': 1.0, '010': 0.7}


def camera2world():
    pos = [0, -0.2, 0.4]
    R_init = np.diag([1, -1, -1])
    quat_wxyz = [0.9926, 0.1216, 0, 0]
    quat_xyzw = [0.1216, 0, 0, 0.9926]
    R_quat = R.from_quat(quat_xyzw).as_matrix()
    R_total = R_quat @ R_init

    T_camera2world = np.eye(4)
    T_camera2world[0:3, 0:3] = R_total
    T_camera2world[0:3, 3] = pos

    return T_camera2world

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

def transform_action(actions, anchor2base):
    """
    Transform the actions from anchor frame to base frame
    """
    grasp_poses, grasp_quats, grasp_forces = actions[:, 0:3], actions[:, 3:7], actions[:, 7:8]
    grasp_poses = np.dot(anchor2base[0:3, 0:3], grasp_poses.T).T + anchor2base[0:3, 3]

    grasp_mats = R.from_quat([grasp_quats[i] for i in range(grasp_quats.shape[0])]).as_matrix()  # convert to rotation matrix
    grasp_mats = anchor2base[0:3, 0:3] @ grasp_mats  # R_new = R_ab * R_old
    grasp_quats = R.from_matrix([grasp_mats[i] for i in range(grasp_mats.shape[0])]).as_quat()  # convert back to quaternion
    actions = np.hstack((grasp_poses, grasp_quats, grasp_forces))

    return actions
    
def encode_action(action, hand_offset, approach_offset):
    """
    Transform the 8D action to the target position and rotation of the hand.
    The action is in the form of a 8D vector:
    1. grasp_pos (3D): the target grasp position in 3D space.
    2. grasp_quat (4D): the target grasp rotation in quaternion (xyzw) format.
    4. grasp force (1D): the force applied to the object during grasping.
    The output is the target position and rotation of the hand.
    1. approach_pos (3D): the position to approach the grasp position.
    2. target_rot (3D): the target rotation of the hand in euler angles.
    3. target_pos (3D): the target position of the hand.
    4. grasp_force (1D): the force applied to the object during grasping
    """

    grasp_pos, grasp_quat, grasp_force = action[0:3], action[3:7], action[7]
    if np.linalg.norm(grasp_quat) < 1e-6:
        print("Error: grasp_quat is zero.")
        return None, None, None, None
    
    grasp_mat = R.from_quat(grasp_quat).as_matrix()

    target_pos = grasp_pos + grasp_mat[:, 2] * hand_offset
    approach_pos = grasp_pos + grasp_mat[:, 2] * approach_offset

    target_rot = R.from_matrix(grasp_mat).as_euler('XYZ', degrees=False)
    
    return approach_pos, target_rot, target_pos, grasp_force