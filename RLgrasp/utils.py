import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy.sparse import coo_matrix, lil_matrix, linalg
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
import triangle

def extract_contour(depth_image):
    """
    Extract the contour from the depth image.
    :param depth_image: A 2D numpy array representing the depth image.
    :return: A 2D numpy array of contour points.
    """
    # Find contours in the depth image
    _, binary = cv2.threshold(depth_image, 220, 255, cv2.THRESH_BINARY_INV)  # 50 可调整
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()
    
    # Resample the contour to 100 points
    # 计算弧长参数
    diff = np.diff(contour, axis=0, append=contour[:1])
    seg_lengths = np.linalg.norm(diff, axis=1)
    arc_lengths = np.cumsum(seg_lengths)
    arc_lengths = np.insert(arc_lengths, 0, 0)
    arc_lengths = arc_lengths[:-1]
    total_length = arc_lengths[-1] + seg_lengths[-1]
    uniform_dist = np.linspace(0, total_length, num=100, endpoint=False)

    # 对x和y分别插值
    fx = scipy.interpolate.interp1d(arc_lengths, contour[:,0], kind='linear', fill_value="extrapolate")
    fy = scipy.interpolate.interp1d(arc_lengths, contour[:,1], kind='linear', fill_value="extrapolate")
    contour_interp = np.stack([fx(uniform_dist), fy(uniform_dist)], axis=1)

    # # 可视化检查
    # cv2.imshow('Contour', depth_uint8)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('Contour', binary)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    # plt.plot(contour_interp[:,0], contour_interp[:,1])
    # plt.gca().invert_yaxis()
    # plt.axis('equal')
    # plt.title('Bunny Contour')
    # plt.show()

    return contour_interp
    
def harmonic_mapping(vertices, segments, max_area=100):
    """
    基于MeshPy的非凸区域保体积映射（保留凹陷结构）
    :param vertices: 边界顶点列表 [(x1,y1), (x2,y2), ...]
    :param segments: 边界线段列表 [(start_idx0, end_idx0), (start_idx1, end_idx1), ...]
    :param max_area: 网格最大面积（控制网格密度）
    :return: 
        uv_coords - 映射到单位圆的坐标
        mesh_points - 生成的网格点
        mesh_tris - 三角形索引
    """
    # 1. 使用MeshPy生成非凸网格
    A = dict(vertices=vertices, segments=segments)
    t = triangle.triangulate(A, f'pqa{max_area}')
    mesh_points = t['vertices']
    mesh_tris = t['triangles']
    
    # 2. 构建余切权重拉普拉斯矩阵
    n = len(mesh_points)
    I, J, V = [], [], []
    for tri in mesh_tris:
        i, j, k = tri
        # 计算三角形边向量
        vec_ij = mesh_points[j] - mesh_points[i]
        vec_jk = mesh_points[k] - mesh_points[j]
        vec_ki = mesh_points[i] - mesh_points[k]
        
        # 计算余切权重[1](@ref)
        cot_j = np.dot(vec_ij, -vec_ki) / np.abs(np.cross(vec_ij, vec_ki))
        cot_k = np.dot(vec_jk, -vec_ij) / np.abs(np.cross(vec_jk, vec_ij))
        cot_i = np.dot(vec_ki, -vec_jk) / np.abs(np.cross(vec_ki, vec_jk))
        
        # 填充矩阵元素
        for (m, n, w) in [(i, j, 0.5*cot_k), (j, k, 0.5*cot_i), 
                          (k, i, 0.5*cot_j)]:
            I.extend([m, n])
            J.extend([n, m])
            V.extend([-w, -w])
            I.extend([m, n])
            J.extend([m, n])
            V.extend([w, w])
    
    # 构建稀疏矩阵
    L = coo_matrix((V, (I, J)), shape=(len(mesh_points), len(mesh_points))).tolil()
    
    # 3. 非凸边界处理
    # 获取原始边界点索引（保留凹陷）
    boundary_idx = list(set(seg for seg_pair in segments for seg in seg_pair))
    
    # 按弧长映射到单位圆（保持凹陷顺序）
    boundary_pts = mesh_points[boundary_idx]
    arc_lengths = np.linalg.norm(np.diff(boundary_pts, axis=0), axis=1)
    cum_length = np.cumsum(np.insert(arc_lengths, 0, 0))
    theta = 2 * np.pi * cum_length / cum_length[-1]
    target = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # 4. 求解调和映射
    A = L.copy()
    b = np.zeros((len(mesh_points), 2))
    for idx, pt_idx in enumerate(boundary_idx):
        A[pt_idx, :] = 0
        A[pt_idx, pt_idx] = 1
        b[pt_idx] = target[idx]

    A = A.tocsr()  # 转换为CSR格式以提高求解效率
    uv_coords = linalg.spsolve(A, b)  # 求解稀疏系统
    
    return uv_coords, mesh_points, mesh_tris

def inverse_mapping(uv, points, action_map):
    # uv: (N,2) 单位圆坐标
    # points: (N,2) 原空间坐标
    # action_map: (n,2) 单位圆采样点
    delaunay = Delaunay(uv)
    simplex = delaunay.find_simplex(action_map)
    X = delaunay.transform[simplex, :2]
    Y = action_map - delaunay.transform[simplex, 2]
    bary = np.c_[np.einsum('ijk,ik->ij', X, Y), 1 - np.sum(np.einsum('ijk,ik->ij', X, Y), axis=1)]
    tri_indices = delaunay.simplices[simplex]
    # 插值原空间坐标
    mapped = np.einsum('ij,ijk->ik', bary, points[tri_indices])
    
    return mapped

def visualize_mapping(uv, points, tris, samples_disk, samples_orig):
    """    
    Visualize the mapping from the original shape to the unit disk.
    :param uv: (N, 2) coordinates in the unit disk.
    :param points: (N, 2) coordinates in the original shape.
    :param tris: (M, 3) indices of the triangles in the original shape.
    :param samples_disk: (n, 2) sampled points in the unit disk.
    :param samples_orig: (n, 2) sampled points in the original shape.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(points[:,0], points[:,1], 'k.', alpha=0.2)
    plt.triplot(points[:,0], points[:,1], tris, color='gray', alpha=0.2)

    plt.scatter(samples_orig[:, 0], samples_orig[:, 1], s=30, label='Sampled')
    plt.title("Samples in Original Shape")
    plt.legend()

    plt.subplot(122)
    plt.gca().set_aspect('equal')
    plt.plot(uv[:,0], uv[:,1], 'k.', alpha=0.2)
    plt.triplot(uv[:,0], uv[:,1], tris, color='gray', alpha=0.2)
    plt.scatter(samples_disk[:,0], samples_disk[:,1], s=30, label='Sampled')
    plt.title("Samples in Unit Disk")
    plt.legend()
    plt.tight_layout()
    plt.show()

def camera2world_mapping(u, v, depth):
    """
    Convert camera coordinates (u, v, depth) to world coordinates.
    :param x: x coordinate in camera space.
    :param y: y coordinate in camera space.
    :param depth: depth value in camera space.
    :return: A 3D numpy array representing the world coordinates.
    """
    # Calculate the camera intrinsic parameters
    fovy, height, width = np.pi / 4, 512, 512
    fy = height / (2 * np.tan(fovy / 2))
    fx = 571# fy * width / height
    cx, cy = width / 2, height / 2
    # Calculate the coordinates in the camera space
    camera_pos = np.array([(u - cx) * (depth / fx), (v - cy) * (depth / fy), depth])
    # Convert to world coordinates
    rot = R.from_quat([-0.3827, 0, 0, 0.9239])  # [x, y, z, w]
    target_pos = rot.inv().apply(camera_pos) - np.array([0, -0.5, 0.3])
    target_pos[1:] = -target_pos[1:]  # Invert y and z coordinates to match the world coordinate system
    
    return target_pos


def transform_action(action, depth_image, segmentation_mask, hand_offset, approach_offset):
    """
    Transform the action into the target grasping position and rotation.
    1. Extract the object contour from the depth image.
    2. Calculate the harmonic transform of the object contour.
    3. Convert the polar coordinates (r, beta) to Cartesian coordinates (x, y) based on the inverse harmonic transform.
    4. Calculate the depth based on the depth image and depth factor.
    5. Calcultate the grasp position.
    6. Calculate the approach vector based on the rotation angles (theta, phi).
    7. Calculate the approach position, target rotation, target position, and target force.
    :param action: A 7D vector containing the grasping point and rotation parameters.
    :param depth_image: The depth image of the object.
    :param segmentation_mask: The segmentation mask of the object in the depth image.
    :param hand_offset: The offset distance from the grasp point to the target position.
    :param approach_offset: The offset distance from the grasp point to the approach position.
    :return: A tuple containing the approach position, target rotation, target position, and target force.
    """
    action = (action + 1) / 2  # Transform the range from [-1, 1] to [0, 1]
    r, beta, depth_factor, theta, phi, alpha, grasp_force = action[0], action[1] * 2 * np.pi, action[2], action[3] * 8 * np.pi / 20, action[4] * 2 * np.pi, action[5] * 2 * np.pi, action[6] * 5.0
    
    # Step 1: Extract the object contour from the depth image
    # Normalize the depth image to [0, 1]
    depth_image[segmentation_mask == 0] = 255  # Set the background depth to 255
    min_depth, max_depth = np.min(depth_image[segmentation_mask != 0]) / 255, np.max(depth_image[segmentation_mask != 0]) / 255
    try:
        contour = extract_contour(depth_image)
    except Exception as e:
        print(f"Error in extracting contour: {e}")
        return None, None, None, None
    
    vertices = contour
    segments = [(i, (i+1)%len(vertices)) for i in range(len(vertices))]
    
    # Step 2: Calculate the harmonic transform of the object contour
    try:
        uv, points, tris = harmonic_mapping(vertices, segments, max_area=10)
    except Exception as e:
        print(f"Error in harmonic_mapping: {e}")
        return None, None, None, None

    # Step 3: Convert polar coordinates (r, beta) to Cartesian coordinates (x, y)
    action_map = np.array([[r * np.cos(beta), r * np.sin(beta)]])  # (1, 2)
    try:
        action_origin = inverse_mapping(uv, points, action_map)
    except Exception as e:
        print(f"Error in inverse_mapping: {e}")
        return None, None, None, None
    # visualize_mapping(uv, points, tris, action_map, action_origin)

    # Step 4: Calculate the depth based on the depth image and depth factor
    depth = min_depth + depth_factor * (max_depth - min_depth)

    # Step 5: Calculate the grasp position based on the x-y coordinates and depth
    grasp_pos = camera2world_mapping(action_origin[0, 0], action_origin[0, 1], depth)
    # print(f"action and depth: {action_origin}, {depth}")
    # print(f"Grasp position: {grasp_pos}")

    # Step 6: Calculate the approach vector based on the rotation angles (theta, phi)
    approach_vector = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    # Step 7: Calculate the approach position, target rotation, target position, and target force
    target_pos = grasp_pos + approach_vector * hand_offset
    approach_pos = grasp_pos + approach_vector * approach_offset

    target_R_to_normal, _ = R.align_vectors([approach_vector], [[0, 0, 1]])
    target_R_about_normal = R.from_rotvec(alpha * approach_vector)
    rot = target_R_about_normal * target_R_to_normal
    target_rot = rot.as_euler('XYZ', degrees=False)
    
    return approach_pos, target_rot, target_pos, grasp_force