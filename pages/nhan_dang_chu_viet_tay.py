import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
st.set_page_config(page_title="Chu viet Detection", page_icon="🖊️")

model = None

def tao_anh_ngau_nhien():
    #Tạo 100 số ngẫu nhiên trong phạm vị [0,9999]
    index = np.random.randint(0, 9999, 100)

    sample = np.zeros((100, 28, 28, 1))
    for i in range(0, 100):
        sample[i] = st.session_state.X_test[index[i]]

    #Tạo ảnh để xem
    image = np.zeros((10*28, 10*28), np.uint8)
    k = 0
    for i in range(0, 10):
        for j in range(0,10):
            image[i*28:(i+1)*28, j*28:(j+1)*28] = sample[k,:,:,0]
            k = k+1
    return image, sample

if 'is_load' not in st.session_state:
    OPTIMIZER = optimizers.Adam()

    # load model
    model_architecture = 'pages\digit_config.json'
    model_weights = 'pages\digit_weight.weights.h5'
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)

    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    st.session_state.model = model

    # load data
    (_,_), (X_test, _) = datasets.mnist.load_data()
    X_test = X_test.reshape((10000, 28, 28, 1))
    st.session_state.X_test = X_test

    st.session_state.is_load = True
    print('Lần đầu load model và data')
else:
    print('Đã load model và data rồi')
 
if st.button('Tạo ảnh và nhận dạng'):
    image, data = tao_anh_ngau_nhien()
    st.image(image)
    data = data/255.0
    data = data.astype('float32')
    ket_qua = st.session_state.model.predict(data)
    dem = 0
    s = ''
    for x in ket_qua:
        s = s + '%d ' % (np.argmax(x))
        dem = dem + 1
        if (dem % 10 == 0) and (dem < 100):
            s = s + '\n'    
    st.text(s)