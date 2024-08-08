import streamlit as st
import numpy as np
import cv2
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import base64
import io
import copy
import tempfile
import os
import struct

from calibration import point_to_image
from utils import calibration_utils

default_value = {
    "rotation": (0.0, 0.0, 0.0),
    "translation": (0.0, 0.0, 0.0),
    "fov": 50.0,
    "color": "#00f000",
}

def get_default_calibration_page():
    with st.sidebar:
        # translation x, y, z
        col_translation = st.columns(3)
        with col_translation[0]:
            translation_x = st.number_input(label="translation X", value=default_value["translation"][0], step=0.01)
        with col_translation[1]:
            translation_y = st.number_input(label="translation Y", value=default_value["translation"][1], step=0.01)
        with col_translation[2]:
            translation_z = st.number_input(label="translation Z", value=default_value["translation"][2], step=0.01)
        st.session_state.calib_translation = (translation_x, translation_y, translation_z)

        # rotation x, y, z
        col_rotation = st.columns(3)
        with col_rotation[0]:
            rotation_x = st.number_input(label="rotation X", value=default_value["rotation"][0], step=0.01, min_value=-180.0, max_value=180.0)
        with col_rotation[1]:
            rotation_y = st.number_input(label="rotation Y", value=default_value["rotation"][1], step=0.01, min_value=-180.0, max_value=180.0)
        with col_rotation[2]:
            rotation_z = st.number_input(label="rotation Z", value=default_value["rotation"][2], step=0.01, min_value=-180.0, max_value=180.0)
        st.session_state.calib_rotation = (rotation_x, rotation_y, rotation_z)

        # fov
        fov = st.slider(label="FOV", value=default_value["fov"], step=1.0, min_value=0.0, max_value=180.0)
        st.write(fov)
        st.session_state.calib_fov = fov

        # circle color
        color = st.color_picker(label="Circle color", value="#00f000")
        st.session_state.calib_color = color
    
    if st.button("reset"):
        st.session_state.calib_image = None
        st.session_state.calib_bag = None
        st.rerun()

    if st.session_state.calib_image is None:
        st.error("Image is something wrong! Please reset and upload image again.", icon="🚨")
    else:
        image = copy.deepcopy(st.session_state.calib_image)

        if st.session_state.calib_pcd is not None:
            pcd = st.session_state.calib_pcd
            
            rotation_axis = (-90, 90, 0)
            translation = st.session_state.calib_translation
            rotation = st.session_state.calib_rotation
            fov = st.session_state.calib_fov
            projected_pcd, _ = point_to_image.project_lidar_to_screen(pcd, image, rotation_axis, translation, rotation, fov)

            color = st.session_state.calib_color.lstrip('#')
            color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            for point in projected_pcd:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, color, -1)

        st.image(Image.fromarray(np.uint8(image)).convert('RGB'), caption="result")


def uploader_callback():
    st.session_state.calib_txt_or_bag_changed = True

def get_calibration_page():
    with st.sidebar:
        uploaded_file = st.file_uploader(label="Upload a point cloud", type=["txt", "bag"], on_change=uploader_callback)
        if st.session_state.calib_txt_or_bag_changed and uploaded_file is not None:
            print("process")
            st.session_state.calib_txt_or_bag_changed = False
            file_extension = os.path.splitext(uploaded_file.name)[1]
            if file_extension == ".bag":
                st.session_state.calib_pcd = calibration_utils.extract_pcd_from_bag(uploaded_file.read())
            else:
                st.session_state.calib_pcd = np.genfromtxt(io.StringIO(uploaded_file.read().decode('utf-8')), delimiter=' ', dtype=float)
        elif uploaded_file is None:
            st.session_state.calib_pcd = None

    get_default_calibration_page()
    