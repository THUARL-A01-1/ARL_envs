import cad.grasp_sampling
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import mujoco    
from mujoco import viewer
from dexhand.dexhand import DexHandEnv
from scipy.spatial.transform import Rotation as R

def test_in_GUI():
    model_path = os.path.join('dexhand', 'scene.xml')
    with open(model_path,"r") as f:
        xml_content = f.read()
    model = mujoco.MjModel.from_xml_string(xml_content)
    mj_data = mujoco.MjData(model)
    viewer.launch(model, mj_data)

def calculate_wrench(tactile):
    """Calculate the wrench from tactile sensor data.
    Args: tactile (np.ndarray): Tactile sensor data of shape (3, 20, 20).
    Returns: np.ndarray: Wrench vector of shape (6,), representing the 3d force and 3d torque.
    """
    X, Y = np.meshgrid(np.arange(20), np.arange(20))
    X, Y = X.flatten() - 9.5, Y.flatten() - 9.5  # Center the coordinates around (9.5, 9.5), unit: mm
    fx, fy, fz = tactile[1, ...].flatten(), tactile[2, ...].flatten(), tactile[0, ...].flatten()
    Fx, Fy, Fz = fx.sum(), fy.sum(), fz.sum()
    tx, ty, tz = (X - 9.5) * fz, (Y - 9.5) * fz, (X - 9.5) * fy - (Y - 9.5) * fx
    Tx, Ty, Tz = tx.sum(), ty.sum(), tz.sum()  # Torque unit: N*mm
    wrench = np.array([Fx, Fy, Fz, Tx, Ty, Tz])  # Wrench unit: N and N*mm

    return wrench

def calculate_normal(P_field):
    """Calculate the normal vector from tactile sensor data.
    Args: tactile (np.ndarray): Tactile sensor data of shape (3, 20, 20).
    Returns: np.ndarray: Normal vector of shape (3,), representing the x, y, z components.
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_field)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=8))
    N_field = np.asarray(pcd.normals)

    return N_field

def calculate_tangential(N_field, F_field):
    """Calculate the tangential vector from normal and force vectors.
    Args: N_field (np.ndarray): Normal vector of shape (3, 20, 20). F_field (np.ndarray): Force vector of shape (3, 20, 20).
    Returns: np.ndarray: Tangential vector of shape (3, 20, 20).
    """
    Fn_field = np.sum(N_field * F_field, axis=1)[:, np.newaxis] * N_field
    Ft_field = F_field - Fn_field
    T_field = Ft_field / np.linalg.norm(Ft_field, axis=1)[:, np.newaxis]
    
    return T_field

def calculate_KJ(N_field):
    """Calculate the matrix multiplication K*J from the normal vector.
    Args: N_field (np.ndarray): Normal vector of shape (3, 20, 20).
    Returns: np.ndarray: Jacobian matrix of shape (6, 3).
    """

    KJ = np.zeros((6, 3))
    KJ[0, 0] = N_field[0].sum()
    KJ[1, 1] = N_field[1].sum()
    KJ[2, 2] = N_field[2].sum()
    KJ[3, 0] = -N_field[1].sum()
    KJ[4, 1] = N_field[0].sum()
    KJ[5, 2] = -N_field[0].sum()

    return KJ

def pre_grasp(env, point, normal, depth):
    """Pre-grasp the object by moving the hand to the target position.
    Args: env (DexHandEnv): The DexHand environment. point, normal, depth: Target position of shape (3, 3, 1).
    Note: translation: qpos[0:3], rotation: qpos[3:6]
    """
    translation = point + normal * (depth + 0.15)  # 0.15 is the offset from the base mount to the center of the fingers
    rot, _ = R.align_vectors([normal], [[0, 0, 1]])
    rotation = rot.as_euler('xyz', degrees=False)
    env.mj_data.qpos[0:3] = translation
    env.mj_data.qpos[3] = rotation[0]
    env.mj_data.qpos[4] = rotation[1]
    env.mj_data.qpos[5] = rotation[2]
    env.step(np.array([0, 0, 0, 0, 0, 0, 0]))

def grasp(env):
    """Grasp the object by applying a force to the hand, and then gravity.
    Args: env (DexHandEnv): The DexHand environment.
    """
    # apply grasping force
    env.step(np.array([0, 0, 0, 0, 0, 0, 5]))
    env.step(np.array([0, 0, 0, 0, 0, 0, 10]))
    env.step(np.array([0, 0, 0, 0, 0, 0, 20]))

    # remove the gravity compensation
    body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
    env.mj_model.body_gravcomp[body_id] = 0.0
    env.step(np.array([0, 0, 0, 0, 0, 0, 20]))

def measure(env):
    """
    Measure the contact normals and forces of the grasps.
    Args: env (DexHandEnv): The DexHand environment.
    Consts: rotation_left/right: rotation matrix relative to the hand
            geom_idx_left/right: geom indexes of the mirco units
    Returns: Dict{List[np.adarray]}: positions, normals, and the forces of the fingers
    """
    # 常量，按照手指顺序
    num_fingers = 2
    rotation_list = [np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])]
    geom_idx_list = [[mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_pad_collisions_{i}") for i in range(400)], [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"right_pad_collisions_{i}") for i in range(400)]]

    rotation_hand = env.mj_data.geom_xmat[4].reshape(3, 3)
    measurement = []
    for i, rotation, geom_idx in enumerate(rotation_list, geom_idx_list):
        # Step 1: 初始化PNF矩阵（F矩阵为手指坐标系）
        P_field = env.mj_data.geom_xpos[geom_idx]
        F_field = env.mj_data.sensordata[1200 * i:1200 * (i + 1)].reshape(3, -1).T
        F_field = np.roll(F_field, -1, axis=1)
        N_field = F_field / (0.001 + np.linalg.norm(F_field, axis=1)[:, np.newaxis])
        # Step 2: 使用碰撞对为PN矩阵赋值（PN矩阵为世界坐标系）
        for i in range(env.mj_data.ncon):
            geom_id = env.mj_data.contact[i].geom1  # 获取第i个接触几何体的索引geom_id
            if geom_id in geom_idx:  # 若geom_id属于该手指，则保存该几何体数据
                geom_id = geom_idx.index(geom_id)
                geom_id = 20 * (geom_id // 20) + (19 - geom_id % 20)
                P_field[geom_id] = env.mj_data.contact[i].xpos[:3]
                N_field[geom_id] = env.mj_data.contact[i].frame[:3]
        # Step 3: 使用旋转矩阵计算手指坐标系下的N矩阵, 用于接触力分解
        N_field_finger = N_field @ rotation_hand.T @ rotation
        Fn_field = np.sum(N_field_finger * F_field, axis=1)[:, np.newaxis] * N_field_finger
        Ft_field = F_field - Fn_field

        measurement.append({"P_field": P_field, "N_field": N_field, "Fn_field": Fn_field, "Ft_field": Ft_field})

    return measurement

def post_grasp(env):
    """Post-grasp the object by moving the hand, simulating the disturbance.
    Args: env (DexHandEnv): The DexHand environment.
    """
    for i in range(1):
        env.step(np.array([0, 0, 0.05, 0, 0, 0, 20]))
        env.step(np.array([0, 0, -0.05, 0, 0, 0, 20]))

def calculate_empirical_metric(env):
    """
    calculate the empirical metirc under the action disturbance.
    Args: env (DexHandEnv): The DexHand environment.
    Returns: whether the object contacts the floor.
    """
    post_grasp(env)
    floor_idx = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    success = True
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与地面接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if geom_id1 == floor_idx or geom_id2 == floor_idx:
            success = False
    
    return success

def calculate_FC_metric(measurement):
    pass

def calculate_our_metric(measurement):
    pass

def conduct_simulation(env, grasps):
    metrics = {"FC_metric": [], "our_metric": [], "empirical_metric": []}
    for grasp_point, grasp_normal, grasp_depth in grasps:
        _ = env.reset()
        pre_grasp(env, grasp_point, grasp_normal, grasp_depth)
        grasp(env)
        measurement = measure(env)
        metrics["FC_metric"].append(calculate_FC_metric(measurement))
        metrics["our_metric"].append(calculate_our_metric(measurement))
        metrics["empirical_metric"].append(calculate_empirical_metric(env))

        

        


    
def test_env():
    # initialize the environment
    env = DexHandEnv()
    _ = env.reset()
    rotation_hand = env.mj_data.geom_xmat[4].reshape(3, 3)
    rotation_left, rotation_right = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
    geom_idx = [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_pad_collisions_{i}") for i in range(400)]
    
    grasp_points, grasp_normals, grasp_depths = cad.grasp_sampling.main()

    # pre-grasp the object
    pre_grasp(env, grasp_points[0], grasp_normals[0], grasp_depths[0])
    
    # grasp the object
    grasp(env)
    P_field = env.mj_data.geom_xpos[geom_idx]
    F_field = env.mj_data.sensordata[:1200].reshape(3, -1).T
    F_field = np.roll(F_field, -1, axis=1)
    N_field = F_field / (0.001 + np.linalg.norm(F_field, axis=1)[:, np.newaxis])
    for i in range(env.mj_data.ncon):
        geom_id = env.mj_data.contact[i].geom1
        if geom_id in geom_idx:
            geom_id = geom_idx.index(geom_id)
            geom_id = 20 * (geom_id // 20) + (19 - geom_id % 20)
            N_field[geom_id] = env.mj_data.contact[i].frame[:3]
    N_field = N_field @ rotation_hand.T @ rotation_left
    Fn_field = np.sum(N_field * F_field, axis=1)[:, np.newaxis] * N_field
    Ft_field = F_field - Fn_field
    
    F_mask = np.linalg.norm(Fn_field, axis=1) > 0.1
    ratio = np.linalg.norm(Ft_field, axis=1) / np.linalg.norm(Fn_field, axis=1)

    # print(f"Force: {sum(F_field[:, 0])}, {sum(F_field[:, 1])}, {sum(F_field[:, 2])}")
    # print(f"num_contact: {sum(F_mask)}")
    print(f"Ratio: {sum(ratio[F_mask]) / sum(F_mask)}")
    print(f"score: {sum(ratio[F_mask]) / sum(F_field[:, 0])}")

    # post-grasp the object
    post_grasp(env)
    floor_idx = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    success = True
    for i in range(env.mj_data.ncon):
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if geom_id1 == floor_idx or geom_id2 == floor_idx:
            success = False
    
    print(f"Grasp success: {success}")

    env.render()

if __name__ == '__main__':
    test_env()
    # test_in_GUI()