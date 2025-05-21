
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df = df.drop(["id", "Unnamed: 32"], axis=1)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

# Train models
def train_model(model_name, X_train, y_train):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=10000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        return None
    model.fit(X_train, y_train)
    return model

# Streamlit UI
st.title("Breast Cancer Prediction System")

df = load_data()
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "Decision Tree"])

if st.sidebar.button("Train Model"):
    model = train_model(model_name, X_train, y_train)
    if model:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("Results")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        st.error("Model training failed.")
