import streamlit as st
import numpy as np
import cv2
from PIL import Image
import base64
import io
import copy
import os

from calibration import calibration_page, point_to_image
from utils import favicon

st.set_page_config(page_title="macaron", page_icon=favicon.get_favicon())

if 'calib_image' not in st.session_state:
    st.session_state.calib_image = None
if 'calib_pcd' not in st.session_state:
    st.session_state.calib_pcd = None
if 'calib_txt_or_bag_changed' not in st.session_state:
    st.session_state.calib_txt_or_bag_changed = False
if 'calib_rotation' not in st.session_state:
    st.session_state.calib_rotation = calibration_page.default_value["rotation"]
if 'calib_translation' not in st.session_state:
    st.session_state.calib_translation = calibration_page.default_value["translation"]
if 'calib_fov' not in st.session_state:
    st.session_state.calib_fov = calibration_page.default_value["fov"]
if 'calib_color' not in st.session_state:
    st.session_state.calib_color = calibration_page.default_value["color"]

if st.session_state.calib_image is not None:
    calibration_page.get_calibration_page()
else:
    uploaded_file = st.file_uploader(label="Upload a image.", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.calib_image = np.array(image)
        st.rerun()



