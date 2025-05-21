
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------ DATABASE FUNCTIONS ------------------

def create_connection():
    return sqlite3.connect("patients.db", check_same_thread=False)

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            inputs TEXT,
            prediction TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(model, inputs, prediction):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (model, inputs, prediction, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (model, str(inputs), prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = create_connection()
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df

# ------------------ ML FUNCTIONS ------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df = df.drop(["id", "Unnamed: 32"], axis=1)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

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

# ------------------ MAIN APP ------------------

st.title("Breast Cancer Prediction System with Database")

menu = st.sidebar.radio("Navigation", ["Predict", "Patient Records"])
create_table()
df = load_data()
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if menu == "Predict":
    st.sidebar.title("Model Selection")
    model_name = st.sidebar.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "Decision Tree"])

    if st.sidebar.button("Train and Predict"):
        model = train_model(model_name, X_train, y_train)
        if model:
            user_input = {}
            st.subheader("Enter 30 feature values:")
            for col in X.columns:
                val = st.number_input(col, value=float(df[col].mean()))
                user_input[col] = val

            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            result = "Malignant" if prediction == 1 else "Benign"

            st.success(f"Prediction: {result}")
            insert_prediction(model_name, user_input, result)
        else:
            st.error("Model training failed.")

elif menu == "Patient Records":
    st.subheader("Saved Patient Predictions")
    data = get_all_predictions()
    st.dataframe(data)
