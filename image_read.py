import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2
import os
import tempfile
import matplotlib.pyplot as plt

def fix_image(url):
    image =  cv2.imread(url)
    grayscale_i = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(grayscale_i, 64, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    smaller_i = cv2.resize(src=im_bw, dsize=(28,28), interpolation=cv2.INTER_NEAREST)
    inverted_i = cv2.bitwise_not(smaller_i)
    flatten_i = inverted_i.flatten()
    reshaped_i = flatten_i.reshape(1,-1)
    scaler = joblib.load("./scaler_model.joblib")
    scaled_i = scaler.transform(reshaped_i)
    return scaled_i

my_model = joblib.load("./my_model.joblib")

nav = st.sidebar.radio("Navigation Menu",["Upload and predict"])

if nav == "Upload and predict":
	st.title('Digit prediction with SVC')
	st.header('Upload and predict')
	st. write("Here you can upload your picture")

	uploaded_file = st.file_uploader("Upload your file", type=['png','jpg'], accept_multiple_files=False)

	if uploaded_file:
        	temp_dir = tempfile.mkdtemp()
        	path = os.path.join(temp_dir, uploaded_file.name)	
        	with open(path, "wb") as f:
	               	f.write(uploaded_file.getvalue())

	st.subheader("Prediction")
	if st.button("Show me"):
		fixed_image = fix_image(path)
		prediction = my_model.predict(fixed_image)		
		st.success(f"Your number is: {prediction}")
		st.write("Your picture was: ")
		st.image(uploaded_file)
		st.write("Your preprocessed picture looked like: ")
		fixed_image_new = fixed_image.flatten()
		fixed_image_new = fixed_image_new.reshape(28,28)
		fig, ax = plt.subplots()		
		ax.imshow((fixed_image_new), cmap=plt.cm.gray_r, interpolation='nearest')
		st.pyplot(fig)
		


	