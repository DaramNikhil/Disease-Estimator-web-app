import streamlit as st
import pickle
import pandas as pd


st.set_page_config(page_title="Disease estimator",
                   page_icon=":tada:", layout="wide")

# loading the algorithm and feature scaling object in the form of pickle file
data_path = "code/artifacts/Test_data.csv"
with open(data_path) as f:
    diseases = pd.read_csv(f)

# loading the scaling in the form of pickle file
scale_path = "code/artifacts/scaling.pkl"
with open(scale_path, "rb") as s:
    feature_scale = pickle.load(s)

# loading the algorithm and feature scaling object in the form of pickle file
algo_oath = "code/artifacts/algorithm.pkl"
with open(algo_oath, "rb") as a:
    algorithm = pickle.load(a)

# -----header section-----
with st.container():
    st.title("Disease Estimator")
    st.write("Check wether your disease is positive or negative")
    name = st.text_input("Enter your Name", "Type Here...")

    disease = st.selectbox("select disease you suspect ", diseases["Disease"])
    st.write(f"Disease: {disease}")

with st.container():
    Age = st.slider("Enter your age", 5, 80)
    Gender = st.radio("select gender: ", ("Male", "Female"))

# diseases
with st.container():
    fever = st.radio("Fever: ", ("Yes", "No"))
    cough = st.radio("Cough: ", ("Yes", "No"))
    fatigue = st.radio("Fatigue: ", ("Yes", "No"))
    Difficulty_Breathing = st.radio("Difficulty breathing: ", ("Yes", "No"))
    Blood_Presure = st.radio("BloodPresure: ", ("Normal", "High", "Low"))
    Cholesterol_Level = st.radio(
        "Cholesterol Level: ", ("Normal", "High", "Low"))


with st.container():
    if (st.button("Click me to check")):

        my_dict = {
            "Disease": [disease],
            "Age": [Age],
            "Gender": [Gender],
            "Fever": [fever],
            "Cough": [cough],
            "Fatigue": [fatigue],
            "Difficulty Breathing": [Difficulty_Breathing],
            "Blood Pressure": [Blood_Presure],
            "Cholesterol Level": [Cholesterol_Level]}

        df = pd.DataFrame(my_dict)
        scaling = feature_scale.transform(df)

        prediction = algorithm.predict(scaling)

        if (prediction) == 1:

            st.warning("Positive")

            st.text(f"your suspected disease is: {disease}")

            st.warning("please go to the hospital and checked up")

        else:
            st.info("Negative")
