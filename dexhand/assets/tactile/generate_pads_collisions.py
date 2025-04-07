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
            # f.write(f'<body name="right_tactile_pad{i}" pos="{pos_x} {pos_y} {pos_z}">\n')
            # f.write(f'\t<site name="right_tactile_site_{i}" type="sphere" size="0.0005" rgba="1 0 0 0.1"/>\n')  # define the site for touch sensor
            # f.write(f'\t<joint type="slide" axis="0 0 1" range="-0.01 0.01" damping="1" solreflimit="1000 0.05"/>\n')
            # f.write(f'\t<geom class="visual" type="box" pos="{pos_x} {pos_y} {pos_z}" size="0.0005 0.0005 0.005" rgba="{rgb} {rgb} {rgb} 1"/>\n')  # define the pad visualization
            f.write(f'\t<geom class="pad" type="sphere" pos="{pos_x} {pos_y} {pos_z}" size="0.0005 0.0005 0.0005" rgba="{rgb} {rgb} {rgb} 0"/>\n')  # define the pad collision
            # f.write(f'</body>\n')
            # f.write(f'<geom class="pad" type="box" pos="{pos_x} {pos_y} {pos_z}" size="{size_x} {size_y} {size_z}" rgba="{rgb} {rgb} {rgb} 1"/>\n')  # define the pad collision
            # f.write(f'<geom class="visual" type="box" pos="{pos_x} {pos_y} {pos_z}" size="{size_x} {size_y} {size_z}" rgba="{rgb} {rgb} {rgb} 0.5"/>\n')  # define the pad visualization
        
        # f.write('<sensor>\n')
        # for i in range(num_points):
        #     f.write(f'\t<force name="left_force_sensor_{i}" site="left_tactile_site_{i}"/>\n')
        # f.write('</sensor>\n')
        
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
    coordinates = pd.read_csv("./tactile_envs/dexhand/tactile/tactile_markers.csv", header=None)
    num_points = args.num_rows * args.num_cols

    # Paths are hardcoded as requested
    path_left = "./tactile_envs/dexhand/tactile/left_pad_collisions.xml"
    path_right = "./tactile_envs/dexhand/tactile/right_pad_collisions.xml"

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
        print(f"Path (left): {path_left}")
        print(f"Path (right): {path_right}")
        print("---------------------------------------------------")

    # Create XML files for left and right pads
    edit_pad_collisions(path=path_left, coordinates=coordinates, num_points=num_points, scale=args.scale, size_x=args.size_x, size_y=args.size_y, size_z=args.size_z)

    edit_pad_collisions(path=path_right, coordinates=coordinates, num_points=num_points, scale=args.scale, size_x=args.size_x, size_y=args.size_y, size_z=args.size_z)

    # Create sensor configuration file
    touch_sensor_string = f"""<mujoco>
    <sensor>
        <plugin name="touch_right" plugin="mujoco.sensor.touch_grid" objtype="site" objname="right_pad_site">
            <config key="size" value="{args.num_rows} {args.num_rows}"/>
            <config key="fov" value="18 18"/>
            <config key="gamma" value="0"/>
            <config key="nchannel" value="3"/>
        </plugin>
    </sensor>
    <sensor>
        <plugin name="touch_left" plugin="mujoco.sensor.touch_grid" objtype="site" objname="left_pad_site">
            <config key="size" value="{args.num_rows} {args.num_rows}"/>
            <config key="fov" value="18 18"/>
            <config key="gamma" value="0"/> 
            <config key="nchannel" value="3"/>
        </plugin>
    </sensor>
</mujoco>"""

    # Save the sensor configuration file
    with open("./tactile_envs/dexhand/tactile/touch_sensors.xml", "w") as f:
        f.write(touch_sensor_string)
    
    print("pad collision and touch sensor are successfully generated.")

if __name__ == "__main__":
    main()
