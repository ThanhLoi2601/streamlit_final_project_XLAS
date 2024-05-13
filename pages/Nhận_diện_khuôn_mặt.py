import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import tempfile

score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000
st.set_page_config(page_title="Nhận dạng khuôn mặt", page_icon="👨‍👩‍👧‍👦")
st.subheader('Nhận dạng khuôn mặt')
cap = cv.VideoCapture(0)

option = st.selectbox(
    "Bạn muốn nhận dạng qua phương tiện nào?",
    ("Camera", "Video", "None"),
    index=2,
    placeholder="Chọn phương tiện...")
try:
    if st.session_state["LoadModel"] == True:
        print('Đã load model')
        pass
except:
    st.session_state["LoadModel"] = True
    st.session_state["SVC"]=joblib.load('models\svm_model.pkl')
    st.session_state["detector"] = cv.FaceDetectorYN.create(
        "./models/face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k
    )
    st.session_state["recognizer"] = cv.FaceRecognizerSF.create(
        "./models/face_recognition_sface_2021dec.onnx","")
    print('Load model lần đầu')   
mydict = ['Duc Phu', 'Thanh Loi']

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            face_align = st.session_state["recognizer"].alignCrop(frame, face)
            face_feature = st.session_state["recognizer"].feature(face_align)
            test_predict = st.session_state["SVC"].predict(face_feature)
            result = mydict[test_predict[0]]
            #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.putText(frame,result,(coords[0], coords[1] - 5),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(option)
if option != "None" and option != None:
    cap = None
    if option == "Camera":
        cap = cv.VideoCapture(0)   
    elif option == "Video":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi","mkv","mov", "wmv"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            
            cap = cv.VideoCapture(tfile.name)
            
    if cap is not None:   
        tm = cv.TickMeter()

        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        st.session_state["detector"].setInputSize([frameWidth, frameHeight])
        FRAME_WINDOW = st.image([])
        btn_stop = st.button("Stop")
        while True:
            hasFrame, frame = cap.read()
            if not hasFrame or btn_stop:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            faces = st.session_state["detector"].detect(frame) # faces is a tuple
            tm.stop()
            
            key = cv.waitKey(1)
            if key == 27:
                break

            # Draw results on the input image
            visualize(frame, faces, tm.getFPS())

            # Visualize results
            FRAME_WINDOW.image(frame, channels='BGR')
        cv.destroyAllWindows()
        