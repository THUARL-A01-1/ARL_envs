import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import mujoco    
from mujoco import viewer
from dexhand.dexhand import DexHandEnv

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

def pre_grasp(env, target_pos):
    """Pre-grasp the object by moving the hand to the target position.
    Args: env (DexHandEnv): The DexHand environment. target_pos (np.ndarray): Target position of shape (7,).
    """
    # loosen the hand
    env.step(np.array([0, 0, 0, 0, 0, 0, -10]))
    # move the hand to the target position
    env.step(target_pos)

def grasp(env):
    """Grasp the object by applying a force to the hand.
    Args: env (DexHandEnv): The DexHand environment.
    """
    # apply force and lift to a certain height
    env.step(np.array([0, 0, 0, 0, 0, 0, 20]))
    env.step(np.array([0, 0, 0.05, 0, 0, 0, 20]))

    # remove the gravity compensation
    body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
    env.mj_model.body_gravcomp[body_id] = 0.0
    env.step(np.array([0, 0, 0, 0, 0, 0, 20]))

def post_grasp(env):
    """Post-grasp the object by moving the hand, simulating the disturbance.
    Args: env (DexHandEnv): The DexHand environment.
    """
    env.step(np.array([0, 0, 0.05, 0, 0, 0, 20]))
    env.step(np.array([0, 0, -0.05, 0, 0, 0, 20]))

def test_env():
    # initialize the environment
    env = DexHandEnv()
    _ = env.reset()
    rotation_hand = env.mj_data.geom_xmat[4].reshape(3, 3)
    rotation_left, rotation_right = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
    geom_idx = [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_pad_collisions_{i}") for i in range(400)]
    
    # pre-grasp the object
    pre_grasp(env, np.array([0.0, -0.04, -0.09, 0, 0, 0, 0]))
    pre_grasp(env, np.array([0.0, 0, 0, 0, 0, 0.02, 0]))
    
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