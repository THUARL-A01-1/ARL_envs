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
    # 减少mesh的面数
    downsampled_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
    # 将 mesh 转为点云
    point_cloud = downsampled_mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))
    # 体素下采样
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    return downsampled_mesh, downsampled_point_cloud

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

def save_mesh_and_point_cloud(downsampled_mesh, downsampled_point_cloud, downsampled_mesh_file_path, downsampled_point_cloud_file_path):
    """
    Save the downsampled mesh and point cloud to two files.

    Args:
        downsampled_point_cloud (o3d.geometry.PointCloud): The downsampled point cloud.
        file_path (str): Path to save the point cloud.
    """
    try:
        o3d.io.write_triangle_mesh(downsampled_mesh_file_path, downsampled_mesh)
        o3d.io.write_point_cloud(downsampled_point_cloud_file_path, downsampled_point_cloud)
    except Exception as e:
        print(f"Error saving point cloud: {e}")

def main(OBJECT_ID="000"):
    # Define the file paths
    mesh_file_path = f"cad/assets/{OBJECT_ID}/textured.obj"  # Replace with your mesh file path
    downsampled_mesh_file_path = f"cad/assets/{OBJECT_ID}/downsampled_mesh.obj"  # Replace with your desired output path
    downsampled_point_cloud_file_path = f"cad/assets/{OBJECT_ID}/downsampled.ply"  # Replace with your desired output path

    # Load the mesh
    mesh = load_mesh(mesh_file_path)
    if mesh is None:
        return
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    
    # Downsample the mesh
    downsampled_mesh, downsampled_point_cloud = downsample_mesh(mesh, voxel_size=0.01)
    print(f"Downsampled point cloud has {len(downsampled_point_cloud.points)} points.")

    # Visualize the downsampled point cloud
    visualize(downsampled_point_cloud)

    # Save the downsampled mesh and point cloud
    save_mesh_and_point_cloud(downsampled_mesh, downsampled_point_cloud, downsampled_mesh_file_path, downsampled_point_cloud_file_path)

if __name__ == "__main__":
    for OBJECT_ID in ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015", "016", "017", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030", "031", "032", "033", "034", "035", "036", "037", "038", "039", "040", "041", "042", "043", "044", "045", "046", "047", "048", "049", "050", "051", "052", "053", "054", "055", "056", "057", "058", "059", "060", "061", "062", "063", "064", "065", "066", "067", "068", "069", "070", "071", "072", "073", "074", "075", "076", "077", "078", "079", "080", "081", "082", "083", "084", "085", "086", "087", "088"]:
        print(f"Processing object {OBJECT_ID}...")
        main(OBJECT_ID=OBJECT_ID)
