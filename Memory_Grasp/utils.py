import numpy as np
from scipy.spatial.transform import Rotation as R


def encode_action(action, hand_offset, approach_offset):
    """
    Transform the 8D action to the target position and rotation of the hand.
    The action is in the form of a 8D vector:
    1. grasp_pos (3D): the target grasp position in 3D space.
    2. approach vector (3D): the target approach vector in 3D space
    3. alpha (1D): the rotation angle around the approach vector.
    4. grasp force (1D): the force applied to the object during grasping.
    The output is the target position and rotation of the hand.
    1. approach_pos (3D): the position to approach the grasp position.
    2. target_rot (3D): the target rotation of the hand in euler angles.
    3. target_pos (3D): the target position of the hand.
    4. grasp_force (1D): the force applied to the object during grasping
    """

    grasp_pos, approach_vector, alpha, grasp_force = action[0:3], action[3:6], action[6], action[7]

    target_pos = grasp_pos + approach_vector * hand_offset
    approach_pos = grasp_pos + approach_vector * approach_offset

    target_R_to_normal, _ = R.align_vectors([approach_vector], [[0, 0, 1]])
    target_R_about_normal = R.from_rotvec(alpha * approach_vector)
    rot = target_R_about_normal * target_R_to_normal
    target_rot = rot.as_euler('XYZ', degrees=False)
    
    return approach_pos, target_rot, target_pos, grasp_force