import socket
import struct
import os

def send_image_to_server(color_bytes, depth_bytes, object_name_bytes, task_bytes, server_ip='127.0.0.1', server_name='any6d'):
    # 建立连接
    port = 6000 if server_name == 'any6d' else 5000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, port))
    
    # 数据包预处理
    if server_name == 'clip':
        lens = [len(color_bytes)]
        context = [color_bytes]
    if server_name == 'any6d':
        lens = [len(color_bytes), len(depth_bytes), len(object_name_bytes), len(task_bytes)]
        context = [color_bytes, depth_bytes, object_name_bytes, task_bytes]
    
    try:
        # 发送长度
        for l in lens:
            sock.sendall(struct.pack('!I', l))
        # 发送内容
        for c in context:
            sock.sendall(c)

        # 接收返回的recieve_bytes
        raw_len = sock.recv(4)
        pose_len = struct.unpack('!I', raw_len)[0]
        recieve_bytes = b''
        while len(recieve_bytes) < pose_len:
            chunk = sock.recv(pose_len - len(recieve_bytes))
            if not chunk:
                break
            recieve_bytes += chunk

        # 解析返回结果
        if server_name == 'any6d':
            # pred_pose为4x4 float32矩阵
            import numpy as np
            pred_pose = np.frombuffer(recieve_bytes, dtype=np.float32).reshape(4, 4)
            # print("Predicted pose:\n", pred_pose)
            return pred_pose
        if server_name == 'clip':
            # 接收返回的label
            label = recieve_bytes.decode('utf-8')
            print("Predicted label:", label)
            return label
    finally:
        sock.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start ClipServer and Any6DServer")
    parser.add_argument('--name', type=str, default='clip', help='Name of the server to start')
    parser.add_argument('--task', type=str, default='anchor', help='Name of the task to start')
    args = parser.parse_args()
    
    # 示例：从anchor目录读取图片并发送
    anchor_dir = "./results/anchor/test_object"
    color_path = os.path.join(anchor_dir, "color.png")
    depth_path = os.path.join(anchor_dir, "depth.png")
    with open(color_path, 'rb') as f:
        color_bytes = f.read()
    with open(depth_path, 'rb') as f:
        depth_bytes = f.read()
    object_name = "test_object"
    task = args.task  # 或 "query"
    object_name_bytes = object_name.encode('utf-8')
    task_bytes = task.encode('utf-8')
    server_ip = "183.173.80.82"  # 或服务器公网IP
    
    send_image_to_server(color_bytes, color_bytes, object_name_bytes, task_bytes, server_ip, server_name=args.name)