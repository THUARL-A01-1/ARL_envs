import cv2
import matplotlib.pyplot as plt
import numpy as np
import triangle
import scipy.interpolate
from scipy.sparse import coo_matrix, linalg
from scipy.spatial import Delaunay
import matplotlib.collections as mc
import matplotlib.pyplot as plt

def extract_contour(image_path):
    """
    从给定的图像路径提取轮廓点
    :param image_path: 图像文件路径
    :return: 二维轮廓点坐标 (N,2)
    """
    # 读取图片（灰度）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 二值化
    _, binary = cv2.threshold(img, 17, 255, cv2.THRESH_BINARY)
    # 提取轮廓
    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except cv2.error as e:
        print(f"Error finding contours: {e}")
        return np.array([])
    # 选择最大轮廓
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze() / 1000  # (N,2)
    # contour = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1], [0.5, 0.5], [0, 0.5], [0, 1], [-0.5, 1], [-0.5, 0], [0, 0]])  # 示例轮廓
    
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
    contour_interp = np.stack([-fx(uniform_dist), -fy(uniform_dist)], axis=1)

    # 可视化检查
    plt.plot(contour_interp[:,0], contour_interp[:,1])
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title('Bunny Contour')
    plt.show()

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
    # ===== 1. 使用MeshPy生成非凸网格 =====
    A = dict(vertices=vertices, segments=segments)
    t = triangle.triangulate(A, f'pqa{max_area}')
    mesh_points = t['vertices']
    mesh_tris = t['triangles']
    
    # ===== 2. 构建余切权重拉普拉斯矩阵 =====
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
    
    # ===== 3. 非凸边界处理 =====
    # 获取原始边界点索引（保留凹陷）
    boundary_idx = list(set(seg for seg_pair in segments for seg in seg_pair))
    
    # 按弧长映射到单位圆（保持凹陷顺序）
    boundary_pts = mesh_points[boundary_idx]
    arc_lengths = np.linalg.norm(np.diff(boundary_pts, axis=0), axis=1)
    cum_length = np.cumsum(np.insert(arc_lengths, 0, 0))
    theta = 2 * np.pi * cum_length / cum_length[-1]
    target = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # ===== 4. 求解调和映射 =====
    A = L.copy()
    b = np.zeros((len(mesh_points), 2))
    for idx, pt_idx in enumerate(boundary_idx):
        A[pt_idx, :] = 0
        A[pt_idx, pt_idx] = 1
        b[pt_idx] = target[idx]
    
    uv_coords = linalg.spsolve(A, b)  # 求解稀疏系统
    
    return uv_coords, mesh_points, mesh_tris

def uniform_points_in_unit_disk(n):
    # 极坐标均匀采样
    r = np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)

def inverse_mapping(uv, points, samples):
    # uv: (N,2) 单位圆坐标
    # points: (N,2) 原空间坐标
    # samples: (n,2) 单位圆采样点
    delaunay = Delaunay(uv)
    simplex = delaunay.find_simplex(samples)
    X = delaunay.transform[simplex, :2]
    Y = samples - delaunay.transform[simplex, 2]
    bary = np.c_[np.einsum('ijk,ik->ij', X, Y), 1 - np.sum(np.einsum('ijk,ik->ij', X, Y), axis=1)]
    tri_indices = delaunay.simplices[simplex]
    # 插值原空间坐标
    mapped = np.einsum('ij,ijk->ik', bary, points[tri_indices])
    
    return mapped

# ===== 使用示例 =====
if __name__ == "__main__":
    # 1. 定义带凹陷的非凸形状（示例：兔子轮廓）
    contour = extract_contour('cad/bunny.png')  # 假设有一张斯坦福兔子的图片
    vertices = contour  # 加载轮廓点
    # 2. 定义边界线段（保留凹陷结构）
    segments = [(i, (i+1)%len(vertices)) for i in range(len(vertices))]
    # 3. 执行非凸Laplace调和映射
    uv, points, tris = harmonic_mapping(
        vertices, segments, max_area=0.01
    )
    
    # 4. 可视化结果
    n_samples = 200
    samples_disk = uniform_points_in_unit_disk(n_samples)
    samples_orig = inverse_mapping(uv, points, samples_disk)
    # 归一化采样点x坐标用于着色
    cval_disk = samples_disk[:, 0]
    cval_disk_norm = (cval_disk - cval_disk.min()) / (cval_disk.ptp() + 1e-8)

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(points[:,0], points[:,1], 'k.', alpha=0.2)
    plt.triplot(points[:,0], points[:,1], tris, color='gray', alpha=0.2)

    plt.scatter(samples_orig[:, 0], samples_orig[:, 1], c=cval_disk_norm, cmap='cool', s=30, label='Sampled')
    plt.title("Samples in Original Shape")
    plt.legend()

    plt.subplot(122)
    plt.gca().set_aspect('equal')
    plt.plot(uv[:,0], uv[:,1], 'k.', alpha=0.2)
    plt.triplot(uv[:,0], uv[:,1], tris, color='gray', alpha=0.2)
    plt.scatter(samples_disk[:,0], samples_disk[:,1], c=cval_disk_norm, cmap='cool', s=30, label='Sampled')
    plt.title("Samples in Unit Disk")
    plt.legend()
    plt.tight_layout()
    plt.show()