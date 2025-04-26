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
    fx, fy, fz = tactile[1, ...].sum(), tactile[2, ...].sum(), tactile[0, ...].sum()
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

def test_env():
    env = DexHandEnv()
    _ = env.reset()
    env.step(np.array([0, 0, 0, 0, 0, 0, -10]))
    print(env.mj_data.qpos)
    env.step(np.array([0.0, 0.0, -0.05, 0, 0, 0, 0]))
    print(env.mj_data.qpos)
    env.step(np.array([0, 0, 0, 0, 0, 0, 20]))
    print(env.mj_data.qpos)
    observation, _, _, _  = env.step(np.array([0, 0, 0.04, 0, 0, 0, 20]))
    print("left wrench:", calculate_wrench(observation['tactile_left']))
    print("right wrench:", calculate_wrench(observation['tactile_right']))
    # observation, _, _, _  = env.step(np.array([0, 0, 0, 0, 0, 0.05, 10]))
    env.step(np.array([0, 0, -0.04, 0, 0, 0, 20]))
    env.step(np.array([0, 0, 0, 0, 0, 0, -10]))
    
    env.render()

if __name__ == '__main__':
    # test_env()
    test_in_GUI()