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

def test_env():
    env = DexHandEnv()
    env.step(np.array([0, 0, 0, 0, 0, 0, -10]))
    env.step(np.array([0.0, 0.0, -0.06, 0, 0, 0, 0]))
    env.step(np.array([0, 0, 0, 0, 0, 0, 20]))
    observation, _, _, _  = env.step(np.array([0, 0, 0.04, 0, 0, 0, 20]))
    print("left wrench:", calculate_wrench(observation['tactile_left']))
    print("right wrench:", calculate_wrench(observation['tactile_right']))
    # observation, _, _, _  = env.step(np.array([0, 0, 0, 0, 0, 0.05, 10]))
    env.step(np.array([0, 0, -0.04, 0, 0, 0, 20]))
    env.step(np.array([0, 0, 0, 0, 0, 0, -10]))
    
    env.render()

if __name__ == '__main__':
    test_env()
    # test_in_GUI()