import os
import shutil


src_xml = "E:/2 - 3_Technical_material/Simulator/ARL_envs/RLgrasp/scenes/000.xml"

for i in range(3, 89):
    idx = f"{i:03d}"
    dst_xml = f"E:/2 - 3_Technical_material/Simulator/ARL_envs/cad/assets/{idx}/object.xml"
    # 替换 include 行
    with open(dst_xml, "r", encoding="utf-8") as f:
        content = f.read()
    num_geoms = content.count('<geom class="object"')
    for j in range(num_geoms):
        content = content.replace(
            f'<geom class="object" name="convex_{j}_geom" mesh="convex_{j}" material="white"/>\n<geom class="visual" name="convex_{j}_geom_visual" mesh="convex_{j}" material="white"/>',
            f'<geom class="object" name="convex_{j}_geom" mesh="convex_{j}" material="white"/>'
        )
    content = content.replace(
        f'<mesh name="convex_0" file="cad/assets/{idx}/convex_parts/convex_0.obj"/>',
        f'<mesh name="whole" file="cad/assets/{idx}/downsampled_mesh.obj"/>\n        <mesh name="convex_0" file="cad/assets/{idx}/convex_parts/convex_0.obj"/>'
    )
    content = content.replace(
        f'<geom class="object" name="convex_0_geom" mesh="convex_0" material="white"/>',
        f'<geom class="visual" name="whole_geom" mesh="whole" material="white"/>\n            <geom class="object" name="convex_0_geom" mesh="convex_0" material="white"/>'
    )
        
    with open(dst_xml, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"num_geoms: {num_geoms}")
    print(f"Generated: {dst_xml}")

# for i in range(1, 89):
#     idx = f"{i:03d}"
#     dst_xml = f"E:/2 - 3_Technical_material/Simulator/ARL_envs/RLgrasp/scenes/{idx}.xml"
#     # 复制原文件
#     shutil.copy(src_xml, dst_xml)
#     # 替换 include 行
#     with open(dst_xml, "r", encoding="utf-8") as f:
#         content = f.read()
#     content = content.replace(
#         '<include file="cad/assets/000/object.xml"/>',
#         f'<include file="cad/assets/{idx}/object.xml"/>'
#     )
#     with open(dst_xml, "w", encoding="utf-8") as f:
#         f.write(content)
#     print(f"Generated: {dst_xml}")