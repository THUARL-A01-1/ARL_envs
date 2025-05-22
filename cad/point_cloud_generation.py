"""
将mesh网格进行下采样，获得体素空间voxel中均匀分布的稀疏点云
"""
import open3d as o3d

def load_mesh(file_path):
    """
    Load a mesh from a file.

    Args:
        file_path (str): Path to the mesh file.
    
    Returns:
        o3d.geometry.TriangleMesh: The loaded mesh.
    """
    # Load the mesh
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None
    
    return mesh

def downsample_mesh(mesh, voxel_size=0.01):
    """
    Downsample a mesh using voxel downsampling.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh to be downsampled.
        voxel_size (float): The size of the voxel for downsampling.
    
    Returns:
        o3d.geometry.TriangleMesh: The downsampled mesh.
    """
    # 将 mesh 转为点云
    point_cloud = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))
    # 体素下采样
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    return downsampled_point_cloud

def visualize(object):
    """
    Visualize the mesh or point cloud.

    Args:
        object (o3d.geometry.TriangleMesh or o3d.geometry.PointCloud): The object to visualize.
        mode (str): "mesh" for mesh visualization, "pointcloud" for point cloud visualization.
    """
    
    try:
        o3d.visualization.draw_geometries([object])
    except Exception as e:
        print(f"Error visualizing object: {e}")

def save_point_cloud(downsampled_point_cloud, file_path):
    """
    Save the downsampled point cloud to a file.

    Args:
        downsampled_point_cloud (o3d.geometry.PointCloud): The downsampled point cloud.
        file_path (str): Path to save the point cloud.
    """
    try:
        o3d.io.write_point_cloud(file_path, downsampled_point_cloud)
    except Exception as e:
        print(f"Error saving point cloud: {e}")

def main():
    # Define the file paths
    mesh_file_path = "cad/assets/dexhand_base.obj"  # Replace with your mesh file path
    downsampled_point_cloud_file_path = "cad/assets/dexhand_base.ply"  # Replace with your desired output path

    # Load the mesh
    mesh = load_mesh(mesh_file_path)
    if mesh is None:
        return
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    
    # Downsample the mesh
    downsampled_point_cloud = downsample_mesh(mesh, voxel_size=0.01)
    print(f"Downsampled point cloud has {len(downsampled_point_cloud.points)} points.")

    # Visualize the downsampled point cloud
    visualize(downsampled_point_cloud)

    # Save the downsampled point cloud
    save_point_cloud(downsampled_point_cloud, downsampled_point_cloud_file_path)

if __name__ == "__main__":
    main()
