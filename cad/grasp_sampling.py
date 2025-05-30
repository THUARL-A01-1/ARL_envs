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


def initialize_gripper():
    """
    Initialize the gripper geometry.
    
    Returns:
        list: A list of Open3D geometries representing the gripper.
    """
    handle = o3d.geometry.TriangleMesh.create_cylinder(radius=1e-3, height=8e-2)
    hinge = o3d.geometry.TriangleMesh.create_cylinder(radius=1e-3, height=1.4e-1)
    finger_left, finger_right = copy.deepcopy(handle), copy.deepcopy(handle)
    handle.translate([0, 0, 8e-2])
    handle.paint_uniform_color([0.7, 0.7, 0.7])
    hinge.rotate(R.from_euler('XYZ', np.array([0, np.pi / 2, 0]), degrees=False).as_matrix(), center=np.zeros(3))
    hinge.translate([0, 0, 8e-2 / 2])
    finger_left.translate([-1.4e-1 / 2, 0, 0])
    finger_right.translate([1.4e-1 / 2, 0, 0])

    initial_gripper = o3d.geometry.TriangleMesh()
    initial_gripper += handle
    initial_gripper += hinge
    initial_gripper += finger_left
    initial_gripper += finger_right

    return initial_gripper

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

def sample_grasp_normal(num_samples=1):
    """
    Sample grasp normals from a uniform distribution on the unit sphere.
    
    Args:
        num_samples (int): The number of grasp normals to sample.
    
    Returns:
        list: A list of sampled grasp normals.
    """
    phi = np.arccos(1 - 2 * np.random.rand(num_samples))  # [0, pi]
    theta = 2 * np.pi * np.random.rand(num_samples)        # [0, 2pi]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    grasp_normals = np.column_stack((x, y, z))
    
    return grasp_normals

def sample_grasp_angle(num_samples=1):
    """
    Sample grasp angles from a uniform distribution.
    
    Args:
        num_samples (int): The number of grasp angles to sample.
    
    Returns:
        list: A list of sampled grasp angles.
    """
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    
    return angles

def sample_grasp_depth(num_samples=1, min_depth=-1e-1, max_depth=2e-2):
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

def sample_grasp_collision(point_cloud, grasp_points, grasp_normals, grasp_angles, grasp_depths, initial_gripper):
    """
    Sample grasp collision detection.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        grasp_points (list): A list of sampled grasp points.
        grasp_normals (list): A list of sampled grasp normals.
        grasp_angles (list): A list of sampled grasp angles.
        grasp_depths (list): A list of sampled grasp depths.
        initial_gripper (o3d.geometry.TriangleMesh): The initial gripper geometry.
    
    Returns:
        list: A list of collision results for each grasp.
    """
    grasp_collisions = []
    
    for i in range(len(grasp_points)):
        gripper_copy = copy.deepcopy(initial_gripper)
        R_to_normal, _ = R.align_vectors([grasp_normals[i]], [[0, 0, 1]])
        R_about_normal = R.from_rotvec(grasp_angles[i] * grasp_normals[i])  # The normal vector is aleady the z-axis
        rotation_matrix = R_about_normal * R_to_normal
        rotation_xyz = rotation_matrix.as_euler('XYZ', degrees=False)
        gripper_copy.rotate(R.from_euler('XYZ', rotation_xyz, degrees=False).as_matrix(), center=np.zeros(3))  # rotate to grasp normal
        gripper_copy.translate(grasp_points[i] + grasp_normals[i] * grasp_depths[i])  # translate to grasp point  # translate along the normal direction

        collision = cad.collision_detection.distance_collision_detection(gripper_copy, point_cloud)
        grasp_collisions.append(collision)
    
    return grasp_collisions

def visualize_grasp(point_cloud, grasp_points, grasp_normals, grasp_angles, grasp_depths, initial_gripper):
    """
    Visualize the grasp points, normals, and depths on the point cloud.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        grasp_points (list): A list of sampled grasp points.
        grasp_normals (list): A list of sampled grasp normals.
        grasp_angles (list): A list of sampled grasp angles.
        grasp_depths (list): A list of sampled grasp depths.
    """
    # Create a copy of the point cloud for visualization
    point_cloud_copy = copy.deepcopy(point_cloud)
    
    # 在点云中标出sphere: 抓取点
    spheres = []
    for i in range(len(grasp_points)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        sphere.translate(grasp_points[i])
        sphere.paint_uniform_color([1, 0, 0])  # Red color for grasp points
        spheres.append(sphere)
    
    # 在点云中标出gripper: 抓取法线, 角度和深度
    grippers = [initial_gripper]
    for i in range(len(grasp_points)):
        gripper_copy = copy.deepcopy(initial_gripper)
        R_to_normal, _ = R.align_vectors([grasp_normals[i]], [[0, 0, 1]])
        R_about_normal = R.from_rotvec(grasp_angles[i] * grasp_normals[i])
        rotation_matrix = R_about_normal * R_to_normal
        rotation_xyz = rotation_matrix.as_euler('XYZ', degrees=False)
        gripper_copy.rotate(R.from_euler('XYZ', rotation_xyz, degrees=False).as_matrix(), center=np.zeros(3))  # rotate to grasp normal
        gripper_copy.translate(grasp_points[i] + grasp_normals[i] * grasp_depths[i])  # translate to grasp point  # translate along the normal direction

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

    # Visualize the sampled grasps
    try:
        initial_gripper = initialize_gripper()
        visualize_grasp(point_cloud, grasp_points, grasp_normals, grasp_angles, grasp_depths, initial_gripper)
    except Exception as e:
        print(f"Error visualizing grasps: {e}")
        return
    
    return grasp_points, grasp_normals, grasp_angles, grasp_depths

if __name__ == "__main__":
    main()
