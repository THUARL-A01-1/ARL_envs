import numpy as np
from scipy.spatial.transform import Rotation as R


def encode_action(action, hand_offset, approach_offset):
    """
    Transform the 8D action to the target position and rotation of the hand.
    The action is in the form of a 8D vector:
    1. grasp_pos (3D): the target grasp position in 3D space.
    2. grasp_quat (4D): the target grasp rotation in quaternion (xyzw) format.
    4. grasp force (1D): the force applied to the object during grasping.
    The output is the target position and rotation of the hand.
    1. approach_pos (3D): the position to approach the grasp position.
    2. target_rot (3D): the target rotation of the hand in euler angles.
    3. target_pos (3D): the target position of the hand.
    4. grasp_force (1D): the force applied to the object during grasping
    """

    grasp_pos, grasp_quat, grasp_force = action[0:3], action[3:7], action[7]
    if np.linalg.norm(grasp_quat) < 1e-6:
        print("Error: grasp_quat is zero.")
        return None, None, None, None
    
    grasp_mat = R.from_quat(grasp_quat).as_matrix()

    target_pos = grasp_pos + grasp_mat[:, 2] * hand_offset
    approach_pos = grasp_pos + grasp_mat[:, 2] * approach_offset

    target_rot = R.from_matrix(grasp_mat).as_euler('XYZ', degrees=False)
    
    return approach_pos, target_rot, target_pos, grasp_force