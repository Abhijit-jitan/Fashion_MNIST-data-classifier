import streamlit as st
import cv2
import numpy as np
import main

## head
head=st.container()
with head:
    st.title("Fashion Item Classification")


## description
desc=st.container()


## input image
ip=st.container()
IMAGE_SELECT=True
with ip:
    #st.subheader('Enter Image')
    image=st.file_uploader("Upload a photo")

    if image is not None:
        file_bytes=np.asarray(bytearray(image.read()),dtype=np.uint8)
        img=cv2.imdecode(file_bytes,1)
        cv2.imwrite("temp.jpg",img)
        IMAGE_SELECT=True
    else:
        IMAGE_SELECT=False
        st.write("Please Select Image...")


## Detection
detection=st.container()
with detection:
    submit=st.button('Predict')
    if submit & IMAGE_SELECT==True:
        image,label=st.columns(2)
        image.image(img)
        label.write("Predicted Class :")
        label.write(format(main.prediction_function("temp.jpg")))

