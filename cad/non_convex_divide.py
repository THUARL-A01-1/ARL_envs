import coacd
import trimesh
import numpy as np
import os


def split_coacd_obj(input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 直接加载OBJ文件
    mesh = trimesh.load(input_path)
    
    # 如果mesh是Scene对象，获取所有几何体
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
    else:
        geometries = [mesh]

    # 导出每个部分
    for i, geom in enumerate(geometries):
        filename = os.path.join(output_dir, f"convex_{i}.obj")
        geom.export(filename)
        print(f"Exported: {filename}")

    print(f"Done. Exported {len(geometries)} mesh pieces to '{output_dir}'.")

def generate_mujoco_snippet(n_parts, object_id):
    xml_mesh = ""
    xml_geom = ""

    for i in range(n_parts):
        mesh_name = f"convex_{i}"
        xml_mesh += f'        <mesh name="{mesh_name}" file="cad/assets/{object_id}/convex_parts/{mesh_name}.obj"/>\n'
        xml_geom += f'            <geom class="object" name="{mesh_name}_geom" mesh="{mesh_name}" material="white"/>\n'

    full_snippet = f"""<?xml version="1.0" ?>
<mujoco>
    <asset>
{xml_mesh}    </asset>

    <worldbody>
        <body name="composite_object" pos="0 0 0" gravcomp="0">
        <joint name="object_joint" type="free" frictionloss="0.001" damping="0.001"/>
{xml_geom}        </body>
    </worldbody>
</mujoco>"""
    return full_snippet

# 示例调用
if __name__ == "__main__":
    OBJECT_IDS = [i for i in range(70, 89) if i not in []]
    for i in OBJECT_IDS:
        object_id = f"{i:03d}"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(current_dir, f"assets/{object_id}/downsampled_mesh.obj")
        mesh = trimesh.load(input_file, force="mesh")
        print(f"Loaded mesh with {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces.")
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(mesh, threshold=0.05) # a list of convex hulls.

        # 创建输出目录
        output_folder = os.path.join(current_dir, f"assets/{object_id}/convex_parts")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 直接保存每个部分
        for i, (vs, fs) in enumerate(parts):
            part_mesh = trimesh.Trimesh(vs, fs)
            part_mesh.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
            # 直接保存每个部分
            filename = os.path.join(output_folder, f"convex_{i}.obj")
            part_mesh.export(filename)
            print(f"Exported: {filename}")

        print(f"Done. Exported {len(parts)} mesh pieces to '{output_folder}'.")

        """生成xml的导入文件"""
        snippet = generate_mujoco_snippet(len(parts), object_id)
        
        # 保存XML文件
        xml_file = os.path.join(current_dir, f"assets/{object_id}/object.xml")
        with open(xml_file, 'w') as f:
            f.write(snippet)
        print(f"XML file saved to: {xml_file}")
