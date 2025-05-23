"""
碰撞检测：给定gripper的几何体，判断point_cloud是否与几何体干涉
"""

import open3d as o3d
import numpy as np
import copy

def distance_collision_detection(gripper, point_cloud, threshold=2e-2):
    """
    Perform collision detection between a gripper and a point cloud.
    
    Args:
        gripper (o3d.geometry.TriangleMesh): The gripper geometry.
        point_cloud (o3d.geometry.PointCloud): The point cloud to check for collisions.
        threshold (float): The distance threshold for collision detection.
    
    Returns:
        bool: True if there is a collision, False otherwise.
    """
    # Create a copy of the gripper
    gripper_copy = copy.deepcopy(gripper)
    gripper_point_cloud = gripper_copy.sample_points_uniformly(number_of_points=len(gripper_copy.vertices))
    
    # Compute the distance between the gripper and the point cloud
    distances = point_cloud.compute_point_cloud_distance(gripper_point_cloud)
    if np.any(np.array(distances) < threshold):
        return True
    
    return False