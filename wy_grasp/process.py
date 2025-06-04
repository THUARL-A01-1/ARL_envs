import json
from wy_grasp import metrics
import numpy as np
import open3d as o3d
import os


# ROOT_DIR = "E:/2 - 3_Technical_material/Simulator/ARL_envs"
ROOT_DIR = "/home/ad102/AutoRobotLab/projects/Simulation/ARL_envs"

def preprocess_results(OBJECT_ID):
    """
    Preprocess the grasp results for a given object ID by calculating various metrics.
    Args:
        OBJECT_ID (str): The ID of the object to process.
    """
    metrics_list = []
    # 读取对应的mesh
    file_path = os.path.join(ROOT_DIR, f"cad/assets/{OBJECT_ID}/downsampled.ply")  # Replace with your point cloud file path
    point_cloud = o3d.io.read_point_cloud(file_path)
    initial_centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    # print(f"The initial centroid of the object {OBJECT_ID}: {initial_centroid}")

    with open(os.path.join(ROOT_DIR, f"results/{OBJECT_ID}/grasp_results.json"), "r", encoding="utf-8") as f:
        print(f"Number of valid grasps: {len(f.readlines())}")
        f.seek(0)
        for line in f:
            result = json.loads(line)
            object_id, grasp_id = result["object_id"], result["grasp_index"]
            fricion_coef = result["friction_coef"]
            contact_result = result["contact_result"]
            grasp_result = result["grasp_result"]
            measurement1, measurement2 = result["measurement1"], result["measurement2"]
            if contact_result == True:
                centroid = initial_centroid + np.array(measurement1[0]["object_pos"])  # Update the centroid with the object position
                our_metric, Fv = metrics.calculate_our_metric(measurement2)
                antipodal_metric, distance = metrics.calculate_antipodal_metric(measurement2, centroid)
                closure_metric = metrics.calculate_closure_metric(measurement2, centroid, fricion_coef)

                metrics_list.append({
                    "object_id": object_id,
                    "grasp_id": grasp_id,
                    "grasp_result": grasp_result,
                    "our_metric": our_metric,
                    "antipodal_metric": antipodal_metric,
                    "closure_metric": closure_metric,
                    "friction_coef": fricion_coef,
                    "distance": distance,
                    "Fv": Fv
                })
    # save the results to a npz file
    np.savez(os.path.join(ROOT_DIR, f"results/{OBJECT_ID}/grasp_metrics.npz"), 
             object_ids=[result["object_id"] for result in metrics_list],
             grasp_ids=[result["grasp_id"] for result in metrics_list],
             grasp_results=[result["grasp_result"] for result in metrics_list],
             our_metrics=[result["our_metric"] for result in metrics_list],
             antipodal_metrics=[result["antipodal_metric"] for result in metrics_list],
             closure_metrics=[result["closure_metric"] for result in metrics_list],
             friction_coefs=[result["friction_coef"] for result in metrics_list],
             distances=[result["distance"] for result in metrics_list],
             Fvs=[result["Fv"] for result in metrics_list])
    
def combine_results(OBJECT_IDS):
    """
    Combine the grasp results from all objects into a single file.
    This function reads the results from individual object files, aggregates them, and saves them into a single file.
    """
    object_ids_all, grasp_ids_all, grasp_results_all, our_metrics_all, antipodal_metrics_all, closure_metrics_all, friction_coefs_all, distances_all, Fvs_all  = [], [], [], [], [], [], [], [], []

    for i in OBJECT_IDS:
        # if i in [85]:#[18, 19, 27, 29, 31, 34, 58, 64, 72, 79, 81, 88]: 
        #     continue
        OBJECT_ID = f"{i:03d}"
        # print(f"Processing object {OBJECT_ID}...")
        data = np.load(os.path.join(ROOT_DIR, f"results/{OBJECT_ID}/grasp_metrics.npz"))
        object_ids_all.extend(data['object_ids'])
        grasp_ids_all.extend(data['grasp_ids'])
        grasp_results_all.extend(data['grasp_results'])
        our_metrics_all.extend(data['our_metrics'])
        antipodal_metrics_all.extend(data['antipodal_metrics'])
        closure_metrics_all.extend(data['closure_metrics'])
        friction_coefs_all.extend(data['friction_coefs'])
        distances_all.extend(data['distances'])
        Fvs_all.extend(np.abs(data['Fvs']))  # Use absolute value of Fv
    # save the results to a npz file
    if not os.path.exists(os.path.join(ROOT_DIR, "results/all")):
        os.makedirs(os.path.join(ROOT_DIR, "results/all"))
    np.savez(os.path.join(ROOT_DIR, "results/all/grasp_metrics.npz"),
             object_ids=object_ids_all,
             grasp_ids=grasp_ids_all,
             grasp_results=grasp_results_all,
             our_metrics=our_metrics_all,
             antipodal_metrics=antipodal_metrics_all,
             closure_metrics=closure_metrics_all,
             friction_coefs=friction_coefs_all,
             distances=distances_all,
             Fvs=Fvs_all)