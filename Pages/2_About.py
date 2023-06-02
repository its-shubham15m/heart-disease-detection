import streamlit as st

st.write("""
<p style='font-size:60px;'><b>About</b><hr style='border: 2px solid blue;'></p>
<p style='font-size:32px;'><b>Description</b></p>
Welcome to the Heart Disease Predictor! Our web app is designed to provide an efficient and user-friendly tool for predicting the likelihood of heart disease based on a set of input features. 
By leveraging machine learning algorithms, our app analyzes various factors such as age, gender, blood pressure, cholesterol levels, and more, to provide an insightful prediction.

Using this app is simple. Just enter the relevant details about the patient's health and lifestyle, and our advanced prediction model will process the information to determine the probability of heart disease.
<hr style='border: 0.5px solid blue;'></p>
<p style='font-size:32px;'><b>Dataset</b></p>
Here, I used dataset from Kaggle for trained the model.
You can find out the dataset here below.
""", unsafe_allow_html=True)
st.markdown("[Heart-Disease-Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")