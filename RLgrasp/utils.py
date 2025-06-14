import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy.sparse import coo_matrix, linalg
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
    try:
        depth_uint8 = (depth_image * 255).astype(np.uint8)
        _, binary = cv2.threshold(depth_uint8, 180, 255, cv2.THRESH_BINARY_INV)  # 50 可调整
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error as e:
        print(f"Error finding contours: {e}")
        return np.array([])
    
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
    L = coo_matrix((V, (I, J)), shape=(len(mesh_points), len(mesh_points))).tocsr()
    
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

def camera2world_mapping(u, v, depth):
    """
    Convert camera coordinates (u, v, depth) to world coordinates.
    :param x: x coordinate in camera space.
    :param y: y coordinate in camera space.
    :param depth: depth value in camera space.
    :return: A 3D numpy array representing the world coordinates.
    """
    # Calculate the camera intrinsic parameters
    fovy, height, width = np.pi / 4, 480, 640
    fy = height / (2 * np.tan(fovy / 2))
    fx = fy * width / height
    cx, cy = width / 2, height / 2
    # Calculate the coordinates in the camera space
    camera_pos = np.array([(u - cx) * (depth / fx), (v - cy) * (depth / fy), -depth])
    # Convert to world coordinates
    rot = R.from_quat([0.3827, 0, 0, 0.9239])  # [x, y, z, w]
    target_pos = rot.apply(camera_pos) + np.array([0, -0.5, 0.5])
    
    return target_pos


def transform_action(action, depth_image, hand_offset, approach_offset):
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
    :param approach_offset: The offset distance from the grasp point to the approach position.
    :return: A tuple containing the approach position, target rotation, target position, and target force.
    """
    r, beta, depth_factor, theta, phi, alpha, grasp_force = action[0], action[1] * 2 * np.pi, action[2], action[3] * np.pi / 2, action[4] * 2 * np.pi, action[5] * 2 * np.pi, action[6] * 3.0
    
    # Step 1: Extract the object contour from the depth image
    # Normalize the depth image to [0, 1]
    depth_image = np.clip(depth_image, 0.0, 1.0)
    min_depth, max_depth = np.min(depth_image), np.max(depth_image)
    depth_image = depth_image / np.max(depth_image)
    contour = extract_contour(depth_image)
    vertices = contour
    segments = [(i, (i+1)%len(vertices)) for i in range(len(vertices))]
    
    # Step 2: Calculate the harmonic transform of the object contour
    uv, points, tris = harmonic_mapping(vertices, segments, max_area=10)

    # Step 3: Convert polar coordinates (r, beta) to Cartesian coordinates (x, y)
    action_map = np.array([[r * np.cos(beta), r * np.sin(beta)]])  # (1, 2)
    action_origin = inverse_mapping(uv, points, action_map)

    # Step 4: Calculate the depth based on the depth image and depth factor
    depth = min_depth + depth_factor * (max_depth - min_depth)
    # TODO: The max_depth is not the real maximum depth of the object, it is the maximum depth of the depth image, debug it.

    # Step 5: Calculate the grasp position based on the x-y coordinates and depth
    grasp_pos = camera2world_mapping(action_origin[0, 0], action_origin[0, 1], depth)

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