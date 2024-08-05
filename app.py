import streamlit as st
import numpy as np
import cv2
from PIL import Image
import base64
import io
import copy

from calibration import point_to_image
from utils import favicon

default_value = {
    "rotate": (0.0, 0.0, 0.0),
    "transition": (0.0, 0.0, 0.0),
    "fov": 50.0,
    "color": "#00f000",
}

st.set_page_config(page_title="macaron", page_icon=favicon.get_favicon())

st.header("Calibration")

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'uploaded_pcd' not in st.session_state:
    st.session_state.uploaded_pcd = None
if 'rotate' not in st.session_state:
    st.session_state.rotate = default_value["rotate"]
if 'transition' not in st.session_state:
    st.session_state.transition = default_value["transition"]
if 'fov' not in st.session_state:
    st.session_state.fov = default_value["fov"]
if 'color' not in st.session_state:
    st.session_state.color = default_value["color"]


with st.sidebar:
    pcd = st.file_uploader(label="Upload a point cloud", type=["txt"])
    if pcd is not None:
        st.session_state.uploaded_pcd = pcd
    else:
        st.session_state.uploaded_pcd = None

    col_transition = st.columns(3)
    with col_transition[0]:
        transition_x = st.number_input(label="transition X", value=default_value["transition"][0], step=0.01)
    with col_transition[1]:
        transition_y = st.number_input(label="transition Y", value=default_value["transition"][1], step=0.01)
    with col_transition[2]:
        transition_z = st.number_input(label="transition Z", value=default_value["transition"][2], step=0.01)
    st.session_state.transition = (transition_x, transition_y, transition_z)

    col_rotate = st.columns(3)
    with col_rotate[0]:
        rotate_x = st.number_input(label="rotate X", value=default_value["rotate"][0], step=0.01, min_value=-180.0, max_value=180.0)
    with col_rotate[1]:
        rotate_y = st.number_input(label="rotate Y", value=default_value["rotate"][1], step=0.01, min_value=-180.0, max_value=180.0)
    with col_rotate[2]:
        rotate_z = st.number_input(label="rotate Z", value=default_value["rotate"][2], step=0.01, min_value=-180.0, max_value=180.0)
    st.session_state.rotate = np.deg2rad((rotate_x, rotate_y, rotate_z))

    fov = st.slider(label="FOV", value=default_value["fov"], step=1.0, min_value=0.0, max_value=180.0)
    st.write(fov)
    st.session_state.fov = fov

    color = st.color_picker(label="Circle color", value="#00f000")
    st.session_state.color = color

if st.session_state.uploaded_image is None:
    uploaded_image = st.file_uploader(label="Upload a image", type=["png", "jpg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.session_state.uploaded_image = np.array(image)
        st.rerun()
else:
    image = copy.deepcopy(st.session_state.uploaded_image)

    if st.session_state.uploaded_pcd is not None:
        pcd = st.session_state.uploaded_pcd
        print(pcd)
        pcd = np.genfromtxt(io.StringIO(pcd.read().decode('utf-8')), delimiter=' ', dtype=float)
        
        rotate_axis = (-90, 90, 0)
        transition = st.session_state.transition
        rotate = st.session_state.rotate
        fov = st.session_state.fov
        projected_pcd, _ = point_to_image.project_lidar_to_screen(pcd, image, rotate_axis, transition, rotate, fov)

        color = st.session_state.color.lstrip('#')
        color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        for point in projected_pcd:
            cv2.circle(image, (int(point[0]), int(point[1])), 1, color, -1)

    st.image(Image.fromarray(np.uint8(image)).convert('RGB'), caption="result")
    if st.button("reset"):
        st.session_state.uploaded_image = None
        st.rerun()

    
    
