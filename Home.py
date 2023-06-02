import streamlit as st
import pandas as pd
import classifier_model_builder as cmb
import pickle
import numpy as np

st.set_page_config(
    page_title="Heart Disease Detector",
    page_icon="❤"
)

st.write("""
# Heart Disease Prediction App

This app predicts whether a person have any heart disease or not

""")

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file]""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def patient_details():
        sex = st.sidebar.selectbox('Sex', ('M', 'F'))
        ChestPainType = st.sidebar.selectbox('Chest Pain Type', ('TA', 'ASY', 'NAP'))
        RestingECG = st.sidebar.selectbox('Resting Electrocardiogram', ('Normal', 'ST', 'LVH'))
        ExerciseAngina = st.sidebar.selectbox('ExerciseAngina', ('Y', 'N'))
        ST_Slope = st.sidebar.selectbox('ST Slope', ('Up', 'Flat', 'Down'))
        Age = st.sidebar.slider('Age', 28, 77)
        RestingBP = st.sidebar.slider('Resting Blood Pressure', 0, 200)
        Cholesterol = st.sidebar.slider('Cholesterol', 0, 603)
        MaxHR = st.sidebar.slider('Maximum Heart Rate', 60, 202)
        Oldpeak = st.sidebar.slider('Old peak', -2, 6)
        FastingBS = st.sidebar.slider('Fasting Blood Sugar', 0, 1)

        data = {'Age': Age,
                'Sex': sex,
                'ChestPainType': ChestPainType,
                'RestingBP': RestingBP,
                'Cholesterol': Cholesterol,
                'FastingBS': FastingBS,
                'RestingECG': RestingECG,
                'MaxHR': MaxHR,
                'ExerciseAngina': ExerciseAngina,
                'Oldpeak': Oldpeak,
                'ST_Slope': ST_Slope, }

        features = pd.DataFrame(data, index=[0])
        return features


    input_df = patient_details()

heart_disease_raw = pd.read_csv('res/heart.csv')
heart = heart_disease_raw.drop(columns=['HeartDisease'])
df = pd.concat([input_df, heart], axis=0)

# Encoding of ordinal features
encode = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # Selects only the first row (the user input data)
df.loc[:, ~df.columns.duplicated()]

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    df = df.loc[:, ~df.columns.duplicated()]
    st.write(df)

# Load the classification models
load_clf_NB = pickle.load(open('res/heart_disease_classifier_NB.pkl', 'rb'))
load_clf_knn = pickle.load(open('res/heart_disease_classifier_KNN.pkl', 'rb'))
load_clf_DT = pickle.load(open('res/heart_disease_classifier_DT.pkl', 'rb'))

# Apply models to make predictions
prediction_NB = load_clf_NB.predict(df)
prediction_proba_NB = load_clf_NB.predict_proba(df)
prediction_knn = load_clf_knn.predict(df)
prediction_proba_knn = load_clf_knn.predict_proba(df)
prediction_DT = load_clf_DT.predict(df)
prediction_proba_DT = load_clf_DT.predict_proba(df)


def NB():
    st.subheader('Naive Bayes Prediction')
    NB_prediction = np.array([0, 1])
    if NB_prediction[prediction_NB] == 1:
        st.write("<p style='font-size:20px;color: yellow'>Heart Disease Detected.</p>", unsafe_allow_html=True)
    else:
        st.write("<p style='font-size:20px;color: green'>You are fine.</p>", unsafe_allow_html=True)
    st.subheader('Naive Bayes Prediction Probability')
    st.write(prediction_proba_NB)
    cmb.plt_NB()


def KNN():
    st.subheader('K-Nearest Neighbour Prediction')
    knn_prediction = np.array([0, 1])
    if knn_prediction[prediction_knn] == 1:
        st.write("<p style='font-size:20px;color: yellow'>Heart Disease Detected.</p>", unsafe_allow_html=True)
    else:
        st.write("<p style='font-size:20px;color: green'>You are fine.</p>", unsafe_allow_html=True)
    st.subheader('KNN Prediction Probability')
    st.write(prediction_proba_knn)
    cmb.plt_KNN()


def DT():
    st.subheader('Decision Tree Prediction')
    DT_prediction = np.array([0, 1])
    if DT_prediction[prediction_DT] == 1:
        st.write("<p style='font-size:20px; color: yellow'>Heart Disease Detected.</p>", unsafe_allow_html=True)
    else:
        st.write("<p style='font-size:20px;color: green'>You are fine.</p>", unsafe_allow_html=True)
    st.subheader('Decision Tree Prediction Probability')
    st.write(prediction_proba_DT)
    cmb.plt_DT()


def select_best_algorithm():
    # Create a dictionary to store the accuracies
    accuracies = {
        'Naive Bayes': cmb.nb_accuracy,
        'KNN': cmb.knn_accuracy,
        'Decision Tree': cmb.dt_accuracy
    }

    # Find the algorithm with the highest accuracy
    best_algorithm = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_algorithm]

    # Display the results
    st.write("<p style='font-size:24px;'>Best Algorithm: {}</p>".format(best_algorithm), unsafe_allow_html=True)


def predict_best_algorithm():
    NB_prediction = np.array([0, 1])
    knn_prediction = np.array([0, 1])
    DT_prediction = np.array([0, 1])
    if NB_prediction[prediction_NB] == 1:
        st.write("<p style='font-size:20px;color: orange'>Heart Disease Detected.</p>", unsafe_allow_html=True)
        cmb.plt_NB()
    elif knn_prediction[prediction_knn] == 1:
        st.write("<p style='font-size:20px;color: orange'>Heart Disease Detected.</p>", unsafe_allow_html=True)
        cmb.plt_KNN()
    elif NB_prediction[prediction_NB] == 1:
        st.write("<p style='font-size:20px;color: orange'>Heart Disease Detected.</p>", unsafe_allow_html=True)
        cmb.plt_DT()
    else:
        st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)


# Displays the user input features
st.subheader('Patient Report')
select_best_algorithm()
predict_best_algorithm()