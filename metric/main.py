"""
Main script for collecting and processing data for the WyGrasp dataset.
1. The object index defition: 000-089 meshes, and "all" is a specified object that combines the results for all meshes.
2. The workflow: simulate -> preprocess results -> combine results -> analyze results.
"""

import os
from metric.process import preprocess_results, combine_results
from metric.simulation import simulate
from metric.analysis import analyze_results


ROOT_DIR = "E:/2 - 3_Technical_material/Simulator/ARL_envs"
# ROOT_DIR = "/home/ad102/AutoRobotLab/projects/Simulation/ARL_envs"

if __name__ == "__main__":
    OBJECT_IDS = [i for i in range(0, 1) if i not in [9, 27, 33, 46, 88]]
    # OBJECT_IDS = [0, 3, 4, 5, 6, 7, 8, 10, 24, 25, 31, 33, 36, 38, 39, 40, 41, 44, 45, 46, 48, 49, 50, 51, 52, 55, 58, 62, 63, 65, 66, 67, 68, 74]
    # OBJECT_IDS = range(89)  # Collect data for all objects in the dataset
    
    # Collect data for all objects in the dataset
    for i in OBJECT_IDS:
        OBJECT_ID = f"{i:03d}"
        print(f"Simulating object {OBJECT_ID}...")
        # simulate(OBJECT_ID=OBJECT_ID, num_samples=200)

    # # Collect data for all objects in the dataset
    # for i in range(89):
    #     OBJECT_ID = f"{i:03d}"
    #     print(f"Prepocessing object {OBJECT_ID}...")
    #     preprocess_results(OBJECT_ID=OBJECT_ID)
    #     # analyze_results(OBJECT_ID=OBJECT_ID)
    
    # # Combine results from all objects into a single file
    # print("Combining results...")
    # combine_results(range(89))

    # # Analyze the results for a specific object
    # print("Analyzing results...")
    # analyze_results(OBJECT_ID="all")

    import json
    json_file = os.path.join(ROOT_DIR, f"metric/results/{OBJECT_ID}/grasp_results.json")
    grasp_points, grasp_normals, our_metrics, antipodal_metrics, closure_metrics = [], [], [], [], []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            grasp_points.append(data["grasp_point"])
            grasp_normals.append(data["grasp_normal"])
            our_metrics.append(sum(data["our_metric"]))
            antipodal_metrics.append(sum(data["antipodal_metric"]))
            closure_metrics.append(data["closure_metric"])
    
    import matplotlib.pyplot as plt
    import numpy as np
    grasp_points, grasp_normals = np.array(grasp_points), np.array(grasp_normals)
    # plt.scatter(our_metrics, closure_metrics, c=antipodal_metrics, cmap='viridis', s=5)
    # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(grasp_points[:, 0], grasp_points[:, 1], grasp_points[:, 2], c=our_metrics, cmap='viridis', s=5)
    # plt.colorbar(sc, label='our_metric')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    # fit the point-normal and force closure metrics
    from sklearn.linear_model import LinearRegression
    X = np.column_stack((grasp_points, grasp_normals))
    y = np.array(closure_metrics)
    model = LinearRegression()
    model.fit(X, y)
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R^2 score: {model.score(X, y)}")
    print(f"Mean absolute error: {np.mean(np.abs(model.predict(X) - y))}")
    # Plot the predicted vs actual values
    plt.scatter(y, model.predict(X), s=5)
    plt.show()


