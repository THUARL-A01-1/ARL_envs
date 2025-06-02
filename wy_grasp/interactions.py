import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def pre_grasp(env, point, normal, angle, depth):
    """Pre-grasp the object by moving the hand to the target position.
    Args: env (DexHandEnv): The DexHand environment. point, normal, depth: Target position of shape (3, 3, 1).
    Note: translation: qpos[0:3], rotation: qpos[3:6]
    """
    translation = point + normal * (depth + 0.15)  # 0.13 is the offset from the base mount to the center of the fingers
    R_to_normal, _ = R.align_vectors([normal], [[0, 0, 1]])
    R_about_normal = R.from_rotvec(angle * normal)
    rot = R_about_normal * R_to_normal
    rotation = rot.as_euler('XYZ', degrees=False)
    env.mj_data.qpos[0:3] = translation
    env.mj_data.qpos[3:6] = rotation

def grasp(env):
    """Grasp the object by applying a force to the hand, and then gravity.
    Args: env (DexHandEnv): The DexHand environment.
    Returns: The measurement before and after applying the gravity, for calculating force closure metric and our metric respectively.
    """
    # apply grasping force
    env.step(np.array([0, 0, 0, 0, 0, 0, 3]))
    measurement1 = measure(env)
    
    # remove the gravity compensation
    body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
    env.mj_model.body_gravcomp[body_id] = 0.0
    env.step(np.array([0, 0, 0.02, 0, 0, 0, 3]), sleep=False)
    measurement2 = measure(env)

    return measurement1, measurement2

def measure(env):
    """
    Measure the contact normals and forces of the grasps.
    Args: env (DexHandEnv): The DexHand environment.
    Consts: rotation_left/right: rotation matrix relative to the hand
            geom_idx_left/right: geom indexes of the mirco units
    Returns: Dict{List[np.adarray]}: positions, normals, and the forces of the fingers
    """
    # 常量，按照手指顺序
    num_fingers = 2  # 黑色driver为左，白色driver为右
    finger_rotation_list = [np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])]
    finger_geom_idx_list = [[mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_pad_collisions_{i}") for i in range(400)], [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"right_pad_collisions_{i}") for i in range(400)]]

    rotation_hand = env.mj_data.geom_xmat[4].reshape(3, 3)
    object_pos = env.mj_data.qpos[8:11].copy()
    measurement = []
    for i in range(num_fingers):
        rotation, geom_idx = finger_rotation_list[i], finger_geom_idx_list[i]
        # Step 1: 初始化PNF矩阵（F矩阵为手指坐标系）
        P_field = env.mj_data.geom_xpos[geom_idx]
        F_field = env.mj_data.sensordata[1200 * i:1200 * (i + 1)].reshape(3, -1).T
        F_field = np.roll(F_field, -1, axis=1)  # [z, x, y] -> [x, y, z]
        F_field[:, 0] = -F_field[:, 0]  # x-axis is flipped
        N_field = F_field / (0.001 + np.linalg.norm(F_field, axis=1)[:, np.newaxis])
        # Step 2: 使用碰撞对为PN矩阵赋值（PN矩阵为世界坐标系）
        for i in range(env.mj_data.ncon):
            geom_id = env.mj_data.contact[i].geom1  # 获取第i个接触几何体的索引geom_id
            if geom_id in geom_idx:  # 若geom_id属于该手指，则保存该几何体数据
                geom_id = geom_idx.index(geom_id)
                geom_id = 20 * (geom_id // 20) + (19 - geom_id % 20)
                P_field[geom_id] = env.mj_data.contact[i].pos[:3]
                N_field[geom_id] = env.mj_data.contact[i].frame[:3]
        # Step 3: 使用旋转矩阵计算手指坐标系下的N矩阵, 用于接触力分解
        N_field_finger = N_field @ rotation_hand @ rotation
        Fn_field = np.sum(N_field_finger * F_field, axis=1)[:, np.newaxis] * N_field_finger
        Ft_field = F_field - Fn_field

        # Step 4: 额外计算用于抵抗重力的竖直力
        F_field_world = F_field @ rotation.T @ rotation_hand.T   # 将F矩阵转换到世界坐标系
        Fv = np.sum(F_field_world[:, 2])
        F_mask = np.linalg.norm(F_field, axis=1) > 0.1

        measurement.append({"P_field": P_field.tolist(), "N_field": N_field.tolist(), "N_field_finger": N_field_finger.tolist(), "Fn_field": Fn_field.tolist(), "Ft_field": Ft_field.tolist(), "Fv": Fv.tolist(), "F_mask": F_mask.tolist(), "rotation_hand": rotation_hand.tolist(), "object_pos": object_pos.tolist()})

    return measurement