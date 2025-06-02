"""
Main script for collecting and processing data for the WyGrasp dataset.
1. The object index defition: 000-089 meshes, and "all" is a specified object that combines the results for all meshes.
2. The workflow: simulate -> preprocess results -> combine results -> analyze results.
"""

import os
from wy_grasp.process import preprocess_results, combine_results
from wy_grasp.simulation import simulate
from wy_grasp.analysis import analyze_results


ROOT_DIR = "E:/2 - 3_Technical_material/Simulator/ARL_envs"

if __name__ == "__main__":
    OBJECT_IDS = range(89)
    # OBJECT_IDS = [000]
    
    # # Collect data for all objects in the dataset
    # for i in OBJECT_IDS:
    #     OBJECT_ID = f"{i:03d}"
    #     print(f"Simulating object {OBJECT_ID}...")
    #     simulate(OBJECT_ID=OBJECT_ID, num_samples=30)

    # Collect data for all objects in the dataset
    for i in OBJECT_IDS:
        OBJECT_ID = f"{i:03d}"
        print(f"Prepocessing object {OBJECT_ID}...")
        preprocess_results(OBJECT_ID=OBJECT_ID)
    
    # Combine results from all objects into a single file
    print("Combining results...")
    combine_results(OBJECT_IDS)

    # Analyze the results for a specific object
    print("Analyzing results...")
    analyze_results(OBJECT_ID="all")

