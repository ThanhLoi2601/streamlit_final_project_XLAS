import streamlit as st
import numpy as np
import cv2

from XuLyAnh.chapter3 import Negative, NegativeColor, Logarit, Power, PiecewiseLinear, Histogram, HistEqual

st.set_page_config(page_title="Xử lý ảnh", page_icon="🖼️")
st.header("Xử lý ảnh")

image_file = st.file_uploader("Chọn file ảnh", type=["jpg", "jpeg", "png","tif"])
processed_image = None

def streamlit_image_to_cv2(image_file, grayscale=True):
    imgin = np.array(bytearray(image_file.read()), dtype=np.uint8)
    imgin = cv2.imdecode(imgin, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    return imgin

def cv2_image_to_streamlit(imgin):
    return cv2.imencode(".png", imgin)[1].tobytes()

st.write("Chọn phương pháp xử lý ảnh")

# put buttons in a single row

cols = st.columns(4)

btn_negative = cols[0].button("Negative", key="negative")
btn_negative_color = cols[0].button("Negative Color", key="negative_color")
btn_logarit = cols[1].button("Logarit", key="logarit")
btn_power = cols[1].button("Power", key="power")
btn_piecewise_linear = cols[2].button("Piecewise Linear", key="piecewise_linear")
btn_histogram = cols[2].button("Histogram", key="histogram")
btn_histequal = cols[3].button("Histogram Equalization", key="histequal")

if btn_negative:
    if image_file is not None:
        imgin = streamlit_image_to_cv2(image_file)
        imgout = Negative(imgin)
        processed_image = cv2_image_to_streamlit(imgout)

if btn_negative_color:
    if image_file is not None:
        imgin = streamlit_image_to_cv2(image_file, grayscale=False)
        imgout = NegativeColor(imgin)
        processed_image = cv2_image_to_streamlit(imgout)

if btn_logarit:
    if image_file is not None:
        imgin = streamlit_image_to_cv2(image_file)
        imgout = Logarit(imgin)
        processed_image = cv2_image_to_streamlit(imgout)

if btn_power:
    if image_file is not None:
        imgin = streamlit_image_to_cv2(image_file)
        imgout = Power(imgin)
        processed_image = cv2_image_to_streamlit(imgout)

if btn_piecewise_linear:
    if image_file is not None:
        imgin = streamlit_image_to_cv2(image_file)
        imgout = PiecewiseLinear(imgin)
        processed_image = cv2_image_to_streamlit(imgout)

if btn_histogram:
    if image_file is not None:
        imgin = streamlit_image_to_cv2(image_file)
        imgout = Histogram(imgin)
        processed_image = cv2_image_to_streamlit(imgout)

if btn_histequal:
    if image_file is not None:
        imgin = streamlit_image_to_cv2(image_file)
        imgout = HistEqual(imgin)
        processed_image = cv2_image_to_streamlit(imgout)

col1, col2 = st.columns(2)
if image_file is not None:
    col1.image(image_file, caption="Ảnh gốc")

if processed_image is not None:
    col2.image(processed_image, caption="Ảnh xử lý")

