import struct
import tempfile
import os
import streamlit as st
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

def pointcloud2_to_numpy(msg):
    height = msg.height
    width = msg.width
    point_step = msg.point_step
    row_step = msg.row_step
    data = msg.data

    num_points = height * width

    cloud_arr = np.zeros((num_points, 4), dtype=np.float32)

    for i in range(num_points):
        offset = i * point_step
        cloud_arr[i, 0] = struct.unpack_from('f', data, offset)[0]      # x
        cloud_arr[i, 1] = struct.unpack_from('f', data, offset + 4)[0]  # y
        cloud_arr[i, 2] = struct.unpack_from('f', data, offset + 8)[0]  # z
        cloud_arr[i, 3] = struct.unpack_from('f', data, offset + 16)[0] # intensity

    return cloud_arr
                

def extract_pcd_from_bag(bag):
    ret = None
    with tempfile.NamedTemporaryFile(delete=False, mode="wb") as temp_file:
        temp_file.write(bag)
        temp_file_path = temp_file.name

        with Reader(temp_file_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/velodyne_points":
                    cdrdata = ros1_to_cdr(rawdata, connection.msgtype)
                    msg = deserialize_cdr(cdrdata, connection.msgtype)
                    numpy_pcd = pointcloud2_to_numpy(msg)
                    ret = numpy_pcd[:, :3]

                    break

    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return ret