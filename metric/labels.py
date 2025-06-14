import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def contact_success(env):
    """Check if the object is in contact with the hand.
    Args: env (DexHandEnv): The DexHand environment.
    Returns: bool: True if the object is in contact with the hand, False otherwise.
    """
    finger_geom_idx_list = [[mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_pad_collisions_{i}") for i in range(400)], [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"right_pad_collisions_{i}") for i in range(400)]]
    object_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "object")
    success = False
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与手指接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if (geom_id1 == object_id and any(geom_id2 in sublist for sublist in finger_geom_idx_list)) or (geom_id2 == object_id and any(geom_id1 in sublist for sublist in finger_geom_idx_list)):
            success = True
            break

    return success

def grasp_success(env):
    """
    calculate the empirical metirc under the action disturbance.
    Args: env (DexHandEnv): The DexHand environment.
    Returns: whether the object contacts the floor.
    """
    success = True
    object_quat0 = env.mj_data.qpos[11:].copy()  # 记录抓取前的物体姿态
    env.step(np.array([0, 0, 0, 0, 0, 0, 1]), sleep=True)  # post-grasp the object to simulate the disturbance
    object_quat1 = env.mj_data.qpos[11:].copy()
    rot = R.from_quat(object_quat1[[1,2,3,0]]) * R.from_quat(object_quat0[[1,2,3,0]]).inv()  # 计算物体的旋转矩阵
    angle_rad = rot.magnitude()  # 旋转弧度
    success = bool(angle_rad < 0.79)  # threshold need to be modified

    if not contact_success(env):
        success = False
        
    floor_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与地面接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if geom_id1 == floor_id or geom_id2 == floor_id:
            success = False
            break
    
    return success