import json
import numpy as np
import open3d as o3d
import os
import shutil

# My modules
import cad.grasp_sampling
from dexhand.dexhand import DexHandEnv
from wy_grasp.interactions import pre_grasp, grasp
from wy_grasp.labels import contact_success, grasp_success
from wy_grasp.metrics import calculate_our_metric, calculate_antipodal_metric, calculate_closure_metric


ROOT_DIR = "E:/2 - 3_Technical_material/Simulator/ARL_envs"

def simulate(OBJECT_ID, num_samples=500):
    """
    Simulate the grasping process for a given object ID by sampling grasps from the CAD model and evaluating them in the DexHand environment.
    Args:
        OBJECT_ID (str): The ID of the object to simulate.
        num_samples (int): The number of grasps to sample from the CAD model.
    1. Initialize the DexHand environment.
    2. Sample grasps from the CAD model.
    3. For each grasp:
        - Reset the environment.
        - Pre-grasp the object using the sampled grasp parameters.
        - Attempt to grasp the object and measure the contact.
        - If the contact is successful, calculate various metrics (our metric, antipodal metric, closure metric).
        - Post-grasp the object to check for grasp success.
    4. Save the results to a JSON file.  
    """
    # Step 0: copy the downsampled mesh to the assets folder
    src = os.path.join(ROOT_DIR, f"cad/assets/{OBJECT_ID}/downsampled.ply")
    dst = os.path.join(ROOT_DIR, f"cad/assets/downsampled.ply")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        print(f"Source not found: {src}")

    # Step 1: initialize the environment
    env = DexHandEnv()
    _ = env.reset()
    if not os.path.exists(os.path.join(ROOT_DIR, f"results/{OBJECT_ID}")):
        os.makedirs(os.path.join(ROOT_DIR, f"results/{OBJECT_ID}"))

    # Step 2: sample grasps from the CAD model
    try:
        grasp_points, grasp_normals, grasp_angles, grasp_depths = cad.grasp_sampling.main(num_samples=num_samples, OBJECT_ID=OBJECT_ID)
    except Exception as e:
        print(f"未生成无碰撞抓取, Error sampling grasps: {e}")
        return
    file_path = os.path.join(ROOT_DIR, f"cad/assets/{OBJECT_ID}/downsampled.ply")  # Replace with your point cloud file path
    point_cloud = o3d.io.read_point_cloud(file_path)
    initial_centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    # print(f"The initial centroid of the object {OBJECT_ID}: {initial_centroid}")

    # Step 3: for each grasp, reset the environment, pre-grasp the object, grasp the object, and post-grasp the object
    for i in range(len(grasp_points)):
        _ = env.reset()
        # pre-grasp the object
        pre_grasp(env, grasp_points[i], grasp_normals[i], grasp_angles[i], grasp_depths[i])
        
        # grasp the object
        measurement1, measurement2 = grasp(env)
        contact_result = contact_success(env)
        if contact_result == False:
            print(f"Grasp {i+1}/{len(grasp_points)}: Contact Failed, skipping...")
            # env.render()
            continue
  
        if measurement1[0]["F_mask"].count(True) < 10 or measurement1[1]["F_mask"].count(True) < 10:  # Check if the contact is sufficient
            print(f"Grasp {i+1}/{len(grasp_points)}: Insufficient contact, skipping...")
            continue

        centroid = initial_centroid + np.array(measurement1[0]["object_pos"])
        our_metric, Fv = calculate_our_metric(measurement2)
        antipodal_metric, distance = calculate_antipodal_metric(measurement1, centroid)
        closure_metric = calculate_closure_metric(measurement1, centroid)

        # post-grasp the object
        grasp_result = grasp_success(env)
        
        print(f"Grasp {i+1}/{len(grasp_points)}: Contact Success: {contact_result}, Grasp Success: {grasp_result} \n Our Metric: {np.mean(our_metric):.2f}, antipodal Metric: {np.sum(antipodal_metric):.2f}, closure_metric: {closure_metric:.2f}, Distance: {distance:.2f}, Fv: {np.sum(Fv):.2f}")

        # save the results
        result = {
            "contact_result": contact_result,
            "grasp_result": grasp_result,
            "measurement1": measurement1,
            "measurement2": measurement2}
        with open(f"results/{OBJECT_ID}/grasp_results.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # # if (closure_metric > 0.5 and np.mean(our_metric) > 0.5) or (closure_metric < 0.2 and np.mean(our_metric) < 0.3):  # Filter out the grasps that are not rational
        # if (distance < 0.005 and grasp_result == False) or ((distance > 0.015 and grasp_result == True)):
        #     our_metric, Fv = metrics.calculate_our_metric(measurement)
        #     antipodal_metric, distance = metrics.calculate_antipodal_metric(measurement)
        #     closure_metric = metrics.calculate_closure_metric(measurement, centroid, draw=True)          
        #     env.render()