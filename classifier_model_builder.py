import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

ht = pd.read_csv('res/heart.csv')

# Ordinal feature encoding

df = ht.copy()
encode = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
    del dummy

# Separating X and y
X = df.drop('HeartDisease', axis=1)
Y = df['HeartDisease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""________Naive Bayes Algorithm________"""
# Train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
# Predict using the Naive Bayes classifier
nb_predictions = nb_classifier.predict(X_test)
# Calculate confusion matrix and accuracy for Naive Bayes classifier
nb_cm = confusion_matrix(y_test, nb_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)

def plt_NB():
    def accuracy():
        st.write("<p style='font-size:24px;'>Accuracy (Naive Bayes): {:.2f}%</p>".format(nb_accuracy * 100),
                 unsafe_allow_html=True)
        st.write()

    # Plot confusion matrix for Naive Bayes classifier
    plt.figure()
    plt.imshow(nb_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Naive Bayes')
    plt.colorbar()
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.yticks([0, 1], ['No Disease', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = nb_cm.max() / 2
    for i, j in np.ndindex(nb_cm.shape):
        plt.text(j, i, format(nb_cm[i, j], 'd'), ha='center', va='center',
                 color='white' if nb_cm[i, j] > thresh else 'black')

    # Display the confusion matrix on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    return st.pyplot(), accuracy()


"""________KNN Algorithm________"""
# Train the K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
# Predict using the K-Nearest Neighbors classifier
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

def plt_KNN():
    def accuracy():
        st.write("<p style='font-size:24px;'>Accuracy (KNN): {:.2f}%</p>".format(knn_accuracy * 100),
                 unsafe_allow_html=True)
        st.write()

    # Plot confusion matrix for Naive Bayes classifier
    plt.figure()
    plt.imshow(nb_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - KNN')
    plt.colorbar()
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.yticks([0, 1], ['No Disease', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = nb_cm.max() / 2
    for i, j in np.ndindex(nb_cm.shape):
        plt.text(j, i, format(nb_cm[i, j], 'd'), ha='center', va='center',
                 color='white' if nb_cm[i, j] > thresh else 'black')

    # Display the confusion matrix on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    return st.pyplot(), accuracy()


"""________Decision Tree________"""
# Train the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
# Predict using the Decision Tree classifier
dt_predictions = dt_classifier.predict(X_test)
# Calculate confusion matrix and accuracy for Decision Tree classifier
dt_cm = confusion_matrix(y_test, dt_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)


def plt_DT():
    def accuracy():
        st.write("<p style='font-size:24px;'>Accuracy (Decision Tree): {:.2f}%</p>".format(dt_accuracy * 100),
                 unsafe_allow_html=True)
        st.write()

    # Plot confusion matrix for Naive Bayes classifier
    plt.figure()
    plt.imshow(nb_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Decision Tree')
    plt.colorbar()
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.yticks([0, 1], ['No Disease', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = nb_cm.max() / 2
    for i, j in np.ndindex(nb_cm.shape):
        plt.text(j, i, format(nb_cm[i, j], 'd'), ha='center', va='center',
                 color='white' if nb_cm[i, j] > thresh else 'black')

    # Display the confusion matrix on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    return st.pyplot(), accuracy()


# Saving the model
pickle.dump(nb_classifier, open('res/heart_disease_classifier_NB.pkl', 'wb'))
pickle.dump(knn_classifier, open('res/heart_disease_classifier_KNN.pkl', 'wb'))
pickle.dump(dt_classifier, open('res/heart_disease_classifier_DT.pkl', 'wb'))