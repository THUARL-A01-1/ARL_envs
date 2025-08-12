import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def contact_labels(env):
    """Check if the object is in contact with the hand and the floor.
    Args: env (DexHandEnv): The DexHand environment.
    Returns: bool: True if the object is in contact, False otherwise.
    """
    finger_geom_idx_list = [[mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_pad_collisions_{i}") for i in range(400)], [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"right_pad_collisions_{i}") for i in range(400)]]
    object_idx_list = [i for i in range(env.mj_model.ngeom) if env.mj_model.geom_bodyid[i] == mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "composite_object")]
    contact_hand = False
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与手指接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if ((geom_id1 in object_idx_list) and any(geom_id2 in sublist for sublist in finger_geom_idx_list)) or ((geom_id2 in object_idx_list) and any(geom_id1 in sublist for sublist in finger_geom_idx_list)):
            contact_hand = True
            break

    contact_floor = False
    floor_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与地面接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if geom_id1 == floor_id or geom_id2 == floor_id:
            contact_floor = True
            break

    return contact_hand, contact_floor

def grasp_success(env):
    """
    calculate the empirical metirc under the action disturbance.
    Args: env (DexHandEnv): The DexHand environment.
    Returns: whether the object contacts the floor.
    """
    success = True
    object_quat0 = env.mj_data.qpos[11:].copy()  # 记录抓取前的物体姿态
    # env.step(np.array([0, 0, 0, 0, 0, 0, 10]), sleep=True)  # post-grasp the object to simulate the disturbance
    object_quat1 = np.array([0, 0, 0, -1])#env.mj_data.qpos[11:].copy()
    rot = R.from_quat(object_quat1) * R.from_quat(object_quat0).inv()  # 计算物体的旋转矩阵
    angle_rad = rot.magnitude()  # 旋转弧度
    success = bool(angle_rad < 0.79)  # threshold need to be modified

    contact_hand, contact_floor = contact_labels(env)
    if not contact_hand:
        success = False
        
    floor_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与地面接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if geom_id1 == floor_id or geom_id2 == floor_id:
            success = False
            break
    
    return success