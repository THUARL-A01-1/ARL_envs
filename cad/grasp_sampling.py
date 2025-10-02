"""
给定点云数据point_cloud, 按照如下顺序采样：
Step 1: 从点云中采样一个点作为抓取点p
Step 2: 从球面空间中采样一个方向作为抓取平面n
Step 3: 采样一维距离作为抓取深度d
"""
import cad.collision_detection
import copy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def initialize_gripper(mode="collsion detection"):
    """
    Initialize the gripper geometry for collision detection and visualization.
    The center of the gripper is at the 1.5cm above the finger bottom.
    Args:
        mode (str): "collsion detection" for collision detection gripper, "visualization" for visualization gripper with smaller vloume.
    
    Returns:
        list: A list (len(initial_grippers)) of Open3D geometries representing the gripper.
    """
    if mode == "collsion detection":
        handle = o3d.geometry.TriangleMesh.create_cylinder(radius=1e-2, height=8e-2)
        hinge = o3d.geometry.TriangleMesh.create_cylinder(radius=1e-3, height=1.6e-1)
        finger = o3d.geometry.TriangleMesh.create_cylinder(radius=1.5e-2, height=8e-2)
    elif mode == "visualization":
        handle = o3d.geometry.TriangleMesh.create_cylinder(radius=1e-3, height=8e-2)
        hinge = o3d.geometry.TriangleMesh.create_cylinder(radius=1e-3, height=1.6e-1)
        finger = o3d.geometry.TriangleMesh.create_cylinder(radius=1e-3, height=8e-2)
    handle.translate([0, 0, 8e-2 + 2.5e-2])
    handle.paint_uniform_color([0.7, 0.7, 0.7])
    hinge.rotate(R.from_euler('XYZ', np.array([0, np.pi / 2, 0]), degrees=False).as_matrix(), center=np.zeros(3))
    hinge.translate([0, 0, 8e-2 / 2 + 2.5e-2])
    finger.translate([0, 0, 2.5e-2])

    # define 10 grippers with width from 0.02m to 0.065m
    initial_grippers = [o3d.geometry.TriangleMesh() for _ in range(10)]
    for i in range(10):
        initial_grippers[i] += handle
        initial_grippers[i] += hinge
        gripper_width = 0.03 + 0.01 * (i + 1)
        finger_left, finger_right = copy.deepcopy(finger), copy.deepcopy(finger)
        finger_left.translate([-gripper_width / 2, 0, 0])
        finger_right.translate([gripper_width / 2, 0, 0])
        initial_grippers[i] += finger_left
        initial_grippers[i] += finger_right

    return initial_grippers

def sample_grasp_point(point_cloud, num_samples=1):
    """
    Sample grasp points from the point cloud.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        num_samples (int): The number of grasp points to sample.
    
    Returns:
        list: A list of sampled grasp points.
    """
    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    
    # Randomly sample grasp points
    indices = np.random.choice(len(points), num_samples, replace=True)
    grasp_points = points[indices]
    
    return grasp_points

def sample_grasp_quat(num_samples=1):
    """
    Sample grasp normals from a uniform distribution on the unit sphere.
    
    Args:
        num_samples (int): The number of grasp normals to sample.
    
    Returns:
        list: A list of sampled grasp normals.
    """
    random_xyzw = R.random(num_samples).as_quat()
    grasp_quats = np.array(random_xyzw)  # shape: (num_samples, 4), [x, y, z, w]
    
    return grasp_quats

def sample_grasp_depth(num_samples=1, min_depth=-1.5e-2, max_depth=1.5e-2):
    """
    Sample grasp depths from a uniform distribution.
    
    Args:
        num_samples (int): The number of grasp depths to sample.
        min_depth (float): The minimum grasp depth.
        max_depth (float): The maximum grasp depth.
    
    Returns:
        list: A list of sampled grasp depths.
    """
    grasp_depths = np.random.uniform(min_depth, max_depth, num_samples)
    
    return grasp_depths

def calculate_grasp_collision_labels(point_cloud, grasp_points, grasp_quats, grasp_depths):
    """
    Sample grasp collision detection.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        grasp_points (list): A list of sampled grasp points.
        grasp_quats (list): A list of sampled grasp quaternions.
        grasp_depths (list): A list of sampled grasp depths.
        initial_grippers (o3d.geometry.TriangleMesh): The initial gripper geometry.
    
    Returns:
        grasp_collision_labels(np.array): (len(grasp_points), len(initial_grippers)) of collision results for each grasp point and width.
        grasp_widths_idx(np.array): (len(grasp_points),) of minimum index of non-collision width for each grasp point, -1 means all widths collide.
    """
    initial_grippers = initialize_gripper(mode="collsion detection")
    grasp_collision_labels = np.zeros((len(grasp_points), len(initial_grippers)), dtype=bool)
    grasp_widths_idx = np.zeros(len(grasp_points), dtype=int) - 1  # -1 means all widths collide
    
    for i in range(len(grasp_points)):
        for j in range(len(initial_grippers)):
            gripper_copy = copy.deepcopy(initial_grippers[j])
            grasp_mat = R.from_quat(grasp_quats[i]).as_matrix()
            gripper_copy.rotate(grasp_mat, center=np.zeros(3))  # rotate to grasp normal
            gripper_copy.translate(grasp_points[i] + grasp_mat[:, 2] * grasp_depths[i])  # translate to grasp point  # translate along the normal direction

            collision = cad.collision_detection.distance_collision_detection(gripper_copy, point_cloud)
            grasp_collision_labels[i, j] = collision

        # Find the minimum index of non-collision width for each grasp point
        noncollision_idx = np.nonzero(~grasp_collision_labels[i])[0]
        if len(noncollision_idx) == 0:
            grasp_widths_idx[i] = -1
        else:
            grasp_widths_idx[i] = noncollision_idx.min()
    
    return grasp_collision_labels, grasp_widths_idx

def visualize_grasp(point_cloud, grasp_points, grasp_quats, grasp_depths, grasp_widths_idx):
    """
    Visualize the grasp points, normals, and depths on the point cloud.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        grasp_points (list): A list of sampled grasp points.
        grasp_quats (list): A list of sampled grasp quaternions.
        grasp_depths (list): A list of sampled grasp depths.
        grasp_widths_idx (list): A list of minimum index of non-collision width for each grasp point, -1 means all widths collide.
    """
    # Create a copy of the point cloud for visualization
    point_cloud_copy = copy.deepcopy(point_cloud)
    initial_grippers = initialize_gripper(mode="visualization")
    
    # 在点云中标出sphere: 抓取点和深度偏移后的抓取点
    spheres = []
    for i in range(len(grasp_points)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        sphere.translate(grasp_points[i])
        sphere.paint_uniform_color([1, 0, 0])  # Red color for grasp points
        spheres.append(sphere)
        sphere_depth = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        grasp_mat = R.from_quat(grasp_quats[i]).as_matrix()
        sphere_depth.translate(grasp_points[i] + grasp_mat[:, 2] * grasp_depths[i])
        sphere_depth.paint_uniform_color([0, 0, 1])  # Blue color for depth offset points
        spheres.append(sphere_depth)
    
    # 在点云中标出gripper: 抓取法线, 角度和深度
    grippers = [initial_grippers[-1]]
    for i in range(len(grasp_points)):
        gripper_copy = copy.deepcopy(initial_grippers[grasp_widths_idx[i]])
        grasp_mat = R.from_quat(grasp_quats[i]).as_matrix()
        gripper_copy.rotate(grasp_mat, center=np.zeros(3))  # rotate to grasp normal
        gripper_copy.translate(grasp_points[i] + grasp_mat[:, 2] * grasp_depths[i])  # translate to grasp point  # translate along the normal direction

        collision = cad.collision_detection.distance_collision_detection(gripper_copy, point_cloud_copy)
        if collision:
            gripper_copy.paint_uniform_color([0.7, 0.7, 0])  # Yellow color for collision
        else:
            gripper_copy.paint_uniform_color([0, 1, 0])  # Green color for no collision

        grippers.append(gripper_copy)
    
    # Combine all geometries for visualization
    geometries = [point_cloud_copy] + spheres + grippers
    
    # Visualize the combined geometries
    o3d.visualization.draw_geometries(geometries)

def main(num_samples=500, OBJECT_ID="000"):
    # Load the point cloud
    try:
        file_path = f"cad/assets/{OBJECT_ID}/downsampled.ply"  # Replace with your point cloud file path
        point_cloud = o3d.io.read_point_cloud(file_path)
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    # Sample grasp points, normals, and depths
    grasp_points, grasp_quats, grasp_depths, grasp_widths_idx = [], [], [], []
    while len(grasp_points) < num_samples:
        try:
            grasp_points_sample = sample_grasp_point(point_cloud, 30 * num_samples) # 根据经验，每次采样30倍的数量
            grasp_quats_sample = sample_grasp_quat(30 * num_samples)
            grasp_depths_sample = sample_grasp_depth(30 * num_samples)
            grasp_collision_labels, grasp_widths_idx_sample = calculate_grasp_collision_labels(point_cloud, grasp_points_sample, grasp_quats_sample, grasp_depths_sample)
            print(f"Sampled {30 * num_samples} grasps, with {np.sum(grasp_widths_idx_sample == -1)} collisions detected.")

            grasp_points.extend(grasp_points_sample[grasp_widths_idx_sample != -1])
            grasp_quats.extend(grasp_quats_sample[grasp_widths_idx_sample != -1])
            grasp_depths.extend(grasp_depths_sample[grasp_widths_idx_sample != -1])
            grasp_widths_idx.extend(grasp_widths_idx[grasp_widths_idx_sample != -1])
        except Exception as e:
            print(f"Error sampling grasps: {e}")
            return

    # Visualize the sampled grasps
    try:
        visualize_grasp(point_cloud, grasp_points, grasp_quats, grasp_depths, grasp_widths_idx)
    except Exception as e:
        print(f"Error visualizing grasps: {e}")
        return
    
    return grasp_points, grasp_quats, grasp_depths, grasp_widths_idx

if __name__ == "__main__":
    main()
