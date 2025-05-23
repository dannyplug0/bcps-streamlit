import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
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

# Initialize database
def init_db(columns):
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    fields = ', '.join([f'"{col}" REAL' for col in columns])
    c.execute(f'''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            {fields},
            prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Save record to database
def save_to_db(input_data, prediction_result):
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    cols = ', '.join(['timestamp'] + list(input_data.keys()) + ['prediction'])
    values = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + list(input_data.values()) + [prediction_result]
    placeholders = ', '.join(['?'] * len(values))
    c.execute(f"INSERT INTO patients ({cols}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()

# Load records from database
def load_records():
    conn = sqlite3.connect("patients.db")
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()
    return df

# Load data and set up
X, y = load_data()
init_db(X.columns)

# Streamlit UI with Tabs
tab1, tab2 = st.tabs(["🔍 Predict", "📋 Patient Records"])

with tab1:
    st.title("Breast Cancer Prediction App")
    st.markdown("Enter the features manually and predict if the tumor is **benign or malignant**.")

    st.sidebar.title("Choose a Model")
    model_option = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest", "Decision Tree"])

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=10000)
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier()

    model.fit(X, y)

    user_input = {}
    st.subheader("Enter feature values:")
    for col in X.columns:
        user_input[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        result = "Malignant" if prediction == 1 else "Benign"
        st.success(f"🧬 Prediction: **{result}**")
        save_to_db(user_input, result)

with tab2:
    st.title("📋 Stored Patient Records")
    records_df = load_records()
    if not records_df.empty:
        st.dataframe(records_df)
    else:
        st.info("No patient records stored yet.")
