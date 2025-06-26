import shutil
import numpy as np
import pandas as pd
import argparse

"""Creates an XML file for pad collisions."""
def edit_pad_collisions(path, coordinates, num_points, scale, size_x, size_y, size_z):
    with open(path, "w") as f:
        f.write("<mujoco>\n")
        
        for i in range(num_points):
            offsetx, offsety, offsetz = 0, 0, 0
            pos_x, pos_y, pos_z = (coordinates.iloc[i, 0] + offsetx) * scale, (coordinates.iloc[i, 1] + offsety) * scale, (coordinates.iloc[i, 2] + offsetz) * scale
            rgb = 0.6 + 0.1 * i / num_points
            f.write(f'\t<geom class="pad" name="pf5_pad_collisions_{i}" type="sphere" pos="{pos_x} {pos_y} {pos_z}" size="0.0005 0.0005 0.0005" rgba="{rgb} {rgb} {rgb} 1"/>\n')  # define the pad collision
        
        f.write("</mujoco>")

"""Parse command-line arguments for pad collision and touch sensor editing."""
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate pad collision XML files with custom parameters.")
    
    parser.add_argument("--num_rows", type=int, default=20, help="Number of rows of markers")
    parser.add_argument("--num_cols", type=int, default=20, help="Number of columns of markers")
    parser.add_argument("--scale", type=float, default=0.0011, help="Scale factor for the coordinates")
    parser.add_argument("--size_x", type=float, default=0.0005, help="Size of the pad in the x direction")
    parser.add_argument("--size_y", type=float, default=0.0005, help="Size of the pad in the y direction")
    parser.add_argument("--size_z", type=float, default=0.0005, help="Thickness of each tactile pad")

    return parser.parse_args()

"""Main function to generate pad collision XML files and touch sensor configuration file."""
def main():
    args = parse_arguments()

    # Load coordinates from the fixed CSV file
    coordinates = pd.read_csv("./allegro/assets/tactile/tactile_markers.csv", header=None)
    num_points = args.num_rows * args.num_cols

    # Paths are hardcoded as requested
    path = "./allegro/assets/tactile/pf5_pad_collisions.xml"

    # Print the settings used
    if args.num_rows != args.num_cols:
        print("Running with the following settings:")
        print(f"Number of rows: {args.num_rows}")
        print(f"Number of columns: {args.num_cols}")
        print(f"Number of points: {num_points}")
        print(f"Scale: {args.scale}")
        print(f"Size (x): {args.size_x}")
        print(f"Size (y): {args.size_y}")
        print(f"Size (z): {args.size_z}")
        print(f"Path (left): {path}")
        print("---------------------------------------------------")

    # Create XML files for left and right pads

    edit_pad_collisions(path=path, coordinates=coordinates, num_points=num_points, scale=args.scale, size_x=args.size_x, size_y=args.size_y, size_z=args.size_z)
    
    print("pad collision flie has been successfully generated.")

if __name__ == "__main__":
    main()
