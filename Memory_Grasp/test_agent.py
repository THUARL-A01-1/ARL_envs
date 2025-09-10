from cad.grasp_sampling import sample_grasp_point, sample_grasp_normal, sample_grasp_angle, sample_grasp_depth, sample_grasp_collision, initialize_gripper, visualize_grasp
import cv2
import matplotlib.pyplot as plt
from Memory_Grasp.memory_grasp_env import MemoryGraspEnv
from Memory_Grasp.server_test import send_image_to_server
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
SERVER_IP = "183.173.80.82"


class MemoryGraspAgent:
    """
    The GraspMemory contains the scene register, the database search, and the data storage.
    coordinate frames definition:
        - anchor frame: the frame of the anchor mesh model
        - camera frame: the camera frame of the RGB-D sensor
        - base frame: the robot base frame
    """
    def __init__(self):
        self.camera2base = np.eye(4)  # camera2base transformation matrix
        self.env = MemoryGraspEnv(render_mode="rgb_array")
        self.env.reset()
        self.scene_xml_list = [f"RLgrasp/scenes/{i:03d}.xml" for i in range(75) if i not in [64, 68, 70]]
        pass

    def get_imgs(self):
        self.env.mj_renderer_rgb.update_scene(self.env.mj_data, camera="main")
        self.env.mj_renderer_depth.update_scene(self.env.mj_data, camera="main")
        color = self.env.mj_renderer_rgb.render().astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = (self.env.mj_renderer_depth.render() * 1000).astype(np.uint16)
        # TODO: adjust the relative params.
        
        return color, depth

    def test_clip(self):
        for scene_xml in self.scene_xml_list:
            self.env.model_path = scene_xml
            self.env.reset()
            color, depth = self.get_imgs()
            success, encoded_image = cv2.imencode('.png', color)
            if success:
                color_bytes = encoded_image.tobytes()
            else:
                color_bytes = None
            object_name = send_image_to_server(color_bytes, None, None, None, SERVER_IP, server_name="clip")
            print(f"Recognized object: {object_name} in scene {scene_xml}.")

    def test_anchor(self):
        for scene_xml in self.scene_xml_list[31:32]:
            self.env.model_path = scene_xml
            self.env.reset()
            
            translation, rotation_wxyz = self.env.mj_data.qpos[8:11], self.env.mj_data.qpos[11:15]
            rotation_matrix = R.from_quat(np.array([rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]])).as_matrix()
            anchor2camera_truth = np.eye(4)
            anchor2camera_truth[0:3, 0:3] = rotation_matrix
            anchor2camera_truth[0:3, 3] = translation
            
            color, depth = self.get_imgs()
            # save the images
            cv2.imwrite("./Memory_Grasp/results/debug/color.png", color)
            cv2.imwrite("./Memory_Grasp/results/debug/depth.png", depth)
            with open("./Memory_Grasp/results/debug/color.png", "rb") as f:
                color_bytes = f.read()
            with open("./Memory_Grasp/results/debug/depth.png", "rb") as f:
                depth_bytes = f.read()
            object_name = "part"#send_image_to_server(color_bytes, None, None, None, SERVER_IP, server_name="clip")
            
            # Use Any6D to set the anchor frame
            task = "anchor"
            anchor2camera = send_image_to_server(color_bytes, depth_bytes, object_name.encode('utf-8'), task.encode('utf-8'), SERVER_IP, "any6d")
            print("Ground truth anchor2camera:\n", anchor2camera_truth)
            print("Predicted anchor2camera:\n", anchor2camera)
            os.makedirs(f"./Memory_Grasp/results/anchor/{object_name}", exist_ok=True)
            np.savetxt(f"./Memory_Grasp/results/anchor/{object_name}/anchor2camera_truth.txt", anchor2camera_truth)
            # np.save(f"./Memory_Grasp/results/anchor/{object_name}/anchor2camera_pred.npy", anchor2camera)
        
    def test_query(self):
        # for scene_xml in self.scene_xml_list[:1]:
        #     self.env.model_path = scene_xml
        self.env.model_path = self.scene_xml_list[31]
        for i in range(10):
            self.env.reset()
            
            translation, rotation_wxyz = self.env.mj_data.qpos[8:11], self.env.mj_data.qpos[11:15]
            rotation_matrix = R.from_quat(np.array([rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]])).as_matrix()
            anchor2camera_truth = np.eye(4)
            anchor2camera_truth[0:3, 0:3] = rotation_matrix
            anchor2camera_truth[0:3, 3] = translation
            
            color, depth = self.get_imgs()
            # save the images
            cv2.imwrite("./Memory_Grasp/results/debug/color.png", color)
            cv2.imwrite("./Memory_Grasp/results/debug/depth.png", depth)
            with open("./Memory_Grasp/results/debug/color.png", "rb") as f:
                color_bytes = f.read()
            with open("./Memory_Grasp/results/debug/depth.png", "rb") as f:
                depth_bytes = f.read()
            object_name = "part"#send_image_to_server(color_bytes, None, None, None, SERVER_IP, server_name="clip")
            
            # Use Any6D to set the anchor frame
            task = "query"
            anchor2camera = send_image_to_server(color_bytes, depth_bytes, object_name.encode('utf-8'), task.encode('utf-8'), SERVER_IP, "any6d")
            print("Ground truth anchor2camera:\n", anchor2camera_truth)
            print("Predicted anchor2camera:\n", anchor2camera)
            os.makedirs(f"./Memory_Grasp/results/query/{object_name}", exist_ok=True)
            np.savetxt(f"./Memory_Grasp/results/query/{object_name}/frame_{i:04d}_anchor2camera_truth.txt", anchor2camera_truth)
            # np.save(f"./Memory_Grasp/results/query/{object_name}/frame_{i:04d}_anchor2camera_pred.npy", anchor2camera)

def check_prediction():
    anchor_dir = "./Memory_Grasp/results/anchor/box"
    anchor_truth = np.load(os.path.join(anchor_dir, "anchor2camera_truth.npy"))
    anchor_pred = np.load(os.path.join(anchor_dir, "anchor2camera_pred.npy"))
    print("Anchor frame check:")
    print("Ground truth anchor2camera:\n", anchor_truth)
    print("Predicted anchor2camera:\n", anchor_pred)
    query_dir = "./Memory_Grasp/results/query/box"
    for frame in os.listdir(query_dir):
        frame_path = os.path.join(query_dir, frame)
        query_truth = np.linalg.inv(anchor_truth) @ np.load(os.path.join(frame_path, "anchor2camera_truth.npy"))
        query_pred = np.linalg.inv(anchor_pred) @ np.load(os.path.join(frame_path, "anchor2camera_pred.npy"))
        print(f"Frame {frame}:")
        print("Ground truth query2anchor:\n", query_truth)
        print("Predicted query2anchor:\n", query_pred)
        print("Translation error (m):", np.linalg.norm(query_truth[0:3, 3] - query_pred[0:3, 3]))
        rot_diff = query_truth[0:3, 0:3].T @ query_pred[0:3, 0:3]
        angle_error = np.arccos((np.trace(rot_diff) - 1) / 2) * 180 / np.pi
        print("Rotation error (deg):", angle_error)



if __name__ == "__main__":
    agent = MemoryGraspAgent()
    # agent.test_clip()
    # agent.test_anchor()
    agent.test_query()
    # check_prediction()
            