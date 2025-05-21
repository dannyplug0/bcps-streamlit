
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    return X, y

X, y = load_data()

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Choose a classifier", ["Random Forest", "Logistic Regression", "Decision Tree"])

# Model setup
if model_name == "Random Forest":
    model = RandomForestClassifier()
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=10000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()

model.fit(X, y)

# App title
st.title("Breast Cancer Prediction App")
st.markdown("Enter patient features to predict if the tumor is benign or malignant.")

# Collect user input
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
input_df = pd.DataFrame([user_input])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    st.success(f"The prediction is: **{result}**")
