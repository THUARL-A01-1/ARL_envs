import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


pos = [0, -0.2, 0.4]
R_init = np.diag([1, -1, -1])
quat_wxyz = [0.9926, 0.1216, 0, 0]
quat_xyzw = [0.1216, 0, 0, 0.9926]
R_quat = R.from_quat(quat_xyzw).as_matrix()
R_total = R_quat @ R_init

T_camera2world = np.eye(4)
T_camera2world[0:3, 0:3] = R_total
T_camera2world[0:3, 3] = pos

T_world2camera = np.linalg.inv(T_camera2world)

def load_data(object_name):
    anchor_dir = f"./Memory_Grasp/results/anchor/{object_name}"
    query_dir = f"./Memory_Grasp/results/query/{object_name}"
    anchor2camera_truth = T_world2camera @ np.loadtxt(os.path.join(anchor_dir, "anchor2camera_truth.txt"))
    anchor2camera_pred = np.loadtxt(os.path.join(anchor_dir, f"{object_name}_initial_pose.txt"))

    query2camera_truth_list = []
    query2camera_pred_list = []
    distance_error_list = []
    angle_error_list = []
    for frame_id in range(10):
        query2camera_truth = T_world2camera @ np.loadtxt(os.path.join(query_dir, f"frame_{frame_id:04d}_anchor2camera_truth.txt"))
        query2camera_pred = np.loadtxt(os.path.join(query_dir, f"frame_{frame_id:04d}/pred_pose.txt"))
        query2camera_truth_list.append((query2camera_truth))
        query2camera_pred_list.append((query2camera_pred))

        distance_error = np.linalg.norm((query2camera_truth[0:3, 3] - anchor2camera_truth[0:3, 3]) - (query2camera_pred[0:3, 3] - anchor2camera_pred[0:3, 3]))
        angle_error = R.from_matrix((query2camera_truth[0:3, 0:3]) @ np.linalg.inv(anchor2camera_truth[0:3, 0:3])).inv() * R.from_matrix((query2camera_pred[0:3, 0:3]) @ np.linalg.inv(anchor2camera_pred[0:3, 0:3]))
        angle_error = angle_error.magnitude()
        distance_error_list.append(distance_error)
        if angle_error > np.pi / 2:
            angle_error = np.pi - angle_error
        angle_error_list.append(angle_error)

    return query2camera_truth_list, query2camera_pred_list, distance_error_list, angle_error_list

if __name__ == "__main__":
    distance_errors, angle_errors = [], []
    object_name_list = ['banana', 'bottle', 'box', 'drill', 'lion', 'part']
    plt.figure()
    for idx, object_name in enumerate(object_name_list):
        _, _, distance_error_list, angle_error_list = load_data(object_name)
        plt.scatter(
            distance_error_list,
            np.array(angle_error_list) * 180 / np.pi,
            label=object_name
        )
    plt.xlabel("Distance error (m)")
    plt.ylabel("Angle error (deg)")
    plt.title("Error scatter plot")
    plt.grid()
    plt.legend()
    plt.savefig("./Memory_Grasp/results/debug/any6d_error.png")
    plt.show()



    