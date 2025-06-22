# 拷贝assets到assets_copy，但每个assets/{id}下只拷贝object.xml
import os
import shutil
import glob

def copy_xmls(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历源目录assets下的所有子目录
    for i in range(89):
        id = f'{i:03d}'
        src_path = os.path.join(src_dir, id)
        dst_path = os.path.join(dst_dir, id)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        # 查找object.xml文件
        xml_file = os.path.join(src_path, 'object.xml')
        if xml_file:
            # 复制object.xml到目标目录
            shutil.copy(xml_file, dst_path)
        else:
            print(f"No object.xml found in {src_path}")

if __name__ == "__main__":
    src_directory = 'cad/assets'
    dst_directory = 'cad/assets_copy'
    
    copy_xmls(src_directory, dst_directory)
    print(f"Copied object.xml files from {src_directory} to {dst_directory}")