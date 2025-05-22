"""
给定点云数据point_cloud, 按照如下顺序采样：
Step 1: 从点云中采样一个点作为抓取点p
Step 2: 从球面空间中采样一个方向作为抓取平面n
Step 3: 采样一维距离作为抓取深度d
"""
import numpy as np
import open3d as o3d
import os
import copy
import math
import random
import matplotlib.pyplot as plt

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
    indices = np.random.choice(len(points), num_samples, replace=False)
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

def sample_grasp_depth(num_samples=1, min_depth=0.01, max_depth=0.1):
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

def visualize_grasp(point_cloud, grasp_points, grasp_normals, grasp_angles, grasp_depths):
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
    vis_point_cloud = copy.deepcopy(point_cloud)
    
    # 在点云中标出sphere: 抓取点
    spheres = []
    for i in range(len(grasp_points)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(grasp_points[i])
        sphere.paint_uniform_color([1, 0, 0])  # Red color for grasp points
        spheres.append(sphere)
    
    # 在点云中标出gripper: 抓取法线, 角度和深度
    grippers = []
    for i in range(len(grasp_normals)):
        handle = o3d.geometry.TriangleMesh.create_cylinder(radius=1e-3, height=5e-3)
        handle.translate([0, 0, 5e-3/2])  # 底部在原点
        handle.paint_uniform_color([0.7, 0.7, 0.7])
        

        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cone_radius=0.005, cone_height=0.01)
        arrow.translate(grasp_points[i])
        arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz(grasp_normals[i]))
        arrow.paint_uniform_color([0, 1, 0])  # Green color for grasp normals
        grippers.append(arrow)
    
    # Create cylinders for grasp depths
    cylinders = []
    for i in range(len(grasp_depths)):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=grasp_depths[i])
        cylinder.translate(grasp_points[i] + np.array(grasp_normals[i]) * (grasp_depths[i] / 2))
        cylinder.rotate(o3d.geometry.get_rotation_matrix_from_xyz(grasp_normals[i]))
        cylinder.paint_uniform_color([0, 0, 1])  # Blue color for grasp depths
        cylinders.append(cylinder)
    
    # Combine all geometries for visualization
    geometries = [vis_point_cloud] + spheres + grippers + cylinders
    
    # Visualize the combined geometries
    o3d.visualization.draw_geometries(geometries)

def main():
    # Load the point cloud
    try:
        file_path = "cad/assets/dexhand_base.ply"  # Replace with your point cloud file path
        point_cloud = o3d.io.read_point_cloud(file_path)
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    # Sample grasp points, normals, and depths
    num_samples = 5
    try:
        grasp_points = sample_grasp_point(point_cloud, num_samples)
        grasp_normals = sample_grasp_normal(num_samples)
        grasp_angles = sample_grasp_angle(num_samples)
        grasp_depths = sample_grasp_depth(num_samples)
        print(f"Sampled {num_samples} grasps.")
    except Exception as e:
        print(f"Error sampling grasps: {e}")
        return

    # Visualize the sampled grasps
    try:
        visualize_grasp(point_cloud, grasp_points, grasp_normals, grasp_angles, grasp_depths)
    except Exception as e:
        print(f"Error visualizing grasps: {e}")
        return

if __name__ == "__main__":
    main()
