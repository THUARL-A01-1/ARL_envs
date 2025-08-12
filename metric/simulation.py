import json
import numpy as np
# np.random.seed(42)  # Set a seed for reproducibility
import open3d as o3d
import os
import shutil

# My modules
import cad.grasp_sampling
from dexhand.dexhand import DexHandEnv
from metric.interactions import pre_grasp, grasp
from metric.labels import contact_labels, grasp_success
from metric.metrics import calculate_our_metric, calculate_antipodal_metric, calculate_closure_metric


ROOT_DIR = "E:/2 - 3_Technical_material/Simulator/ARL_envs"
# ROOT_DIR = "/home/ad102/AutoRobotLab/projects/Simulation/ARL_envs"

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
    src = os.path.join(ROOT_DIR, f"cad/assets/{OBJECT_ID}/downsampled_mesh.obj")
    dst = os.path.join(ROOT_DIR, f"cad/assets/downsampled_mesh.obj")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f"Object mesh {OBJECT_ID} loaded.")
    else:
        print(f"Source not found: {src}")

    # Step 1: initialize the environment
    env = DexHandEnv(model_path="dexhand/scene.xml", render_mode="rgb_array")
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
        
        for friction_coef in [1.0]:#[0.5, 0.75, 1.0, 1.25, 1.5]:
        # # randomize the friction coefficient
        # friction_coef = np.random.uniform(0.5, 1.5)
            _ = env.reset()
            env.mj_model.geom_friction[:] = [friction_coef, 0.005, 0.0001].copy()

            # pre-grasp the object
            pre_grasp(env, grasp_points[i], grasp_normals[i], grasp_angles[i], grasp_depths[i])
            
            # grasp the object
            measurement1, measurement2 = grasp(env)
            contact_result, _ = contact_labels(env)
            # if contact_result == False:
            #     print(f"Grasp {i+1}/{len(grasp_points)}: Contact Failed, skipping...")
            #     # env.render()
            #     continue
    
            if measurement1[0]["F_mask"].count(True) < 3 or measurement1[1]["F_mask"].count(True) < 3:  # Check if the contact is sufficient
                print(f"Grasp {i+1}/{len(grasp_points)}: Insufficient contact, skipping...")
                # env.replay()
                continue

            centroid = initial_centroid + np.array(measurement1[0]["object_pos"])
            our_metric, Fv = calculate_our_metric(measurement2)
            antipodal_metric, distance = calculate_antipodal_metric(measurement1, centroid)
            closure_metric = calculate_closure_metric(measurement2, centroid, friction_coef)

            # post-grasp the object
            grasp_result = grasp_success(env)
            
            print(f"Grasp {i+1}/{len(grasp_points)}: Friction_coef: {friction_coef:.2f}, Contact Success: {contact_result}, Grasp Success: {grasp_result} \n Our Metric: {np.mean(our_metric):.2f}, antipodal Metric: {np.sum(antipodal_metric):.2f}, closure_metric: {closure_metric:.2f}, Distance: {distance:.2f}, Fv: {np.sum(Fv):.2f}")

            # save the results
            result = {
                "object_id": OBJECT_ID,
                "grasp_id": i,
                "grasp_point": grasp_points[i].tolist(),
                "grasp_normal": grasp_normals[i].tolist(),
                "grasp_angle": grasp_angles[i],
                "grasp_depth": grasp_depths[i],

                "friction_coef": friction_coef,
                "contact_result": contact_result,
                "grasp_result": grasp_result,
                # "measurement1": measurement1,
                # "measurement2": measurement2,

                "our_metric": our_metric.tolist(),
                "antipodal_metric": antipodal_metric.tolist(),
                "closure_metric": closure_metric,
                "distance": distance,
                "Fv": Fv.tolist()}
            
            json_file = os.path.join(ROOT_DIR, f"metric/results/{OBJECT_ID}/grasp_results.json")
            if not os.path.exists(json_file):
                with open(json_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                with open(json_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # if (closure_metric > 0.5 and np.mean(our_metric) > 0.5) or (closure_metric < 0.2 and np.mean(our_metric) < 0.3):  # Check the grasps that are not rational
            # if (np.mean(our_metric) / friction_coef >= 0.8 and closure_metric > 0.5):
            #     our_metric, Fv = calculate_our_metric(measurement2)
            #     antipodal_metric, distance = calculate_antipodal_metric(measurement2, centroid)
            #     print(f"Contact points: {measurement2[0]['F_mask'].count(True)}, {measurement2[1]['F_mask'].count(True)}")
            #     closure_metric = calculate_closure_metric(measurement2, centroid, friction_coef, draw=True)          
            #     env.render()
            # env.replay()  # Render the environment to visualize the grasp