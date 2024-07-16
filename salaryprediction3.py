#!/usr/bin/env python
# coding: utf-8

# In[21]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
model = joblib.load("linear_regression_model.pkl")

st.title("Salary Prediction App")

st.sidebar.header("User Input")

age = st.sidebar.slider("Age", min_value=18, max_value=65, value=30)
hourly_rate = st.sidebar.slider("Hourly Rate", min_value=30, max_value=100, value=50)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike", min_value=0, max_value=25, value=10)
performance_rating = st.sidebar.slider("Performance Rating", min_value=1, max_value=4, value=3)
total_working_years = st.sidebar.slider("Total Working Years", min_value=0, max_value=40, value=10)

st.sidebar.subheader("Department:")
selected_department = st.sidebar.radio("Select Department 1 for R&D, 2 for Sales & 3 for HR", ["1", "2", "3"])

st.sidebar.subheader("Gender:")
selected_gender = st.sidebar.radio("Select Gender 1 for Male & 2 for Female", ["1", "2"])

st.sidebar.subheader("Job Role:")
selected_job_role = st.sidebar.radio("Select Job Role 1 for HealthCare   2 for HR   3 for Lab-Tech   4 for Manager   5 for Manufctr Dir   6 for Research Dir   7 for Research Scientist   8 for Sales Exe   9 for Sales Rep", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])

def predict():
    row = np.array([age,hourly_rate, percent_salary_hike, performance_rating, total_working_years, selected_department, selected_gender, selected_job_role])
    X=pd.DataFrame([row])
    prediction=model.predict(X)[0]
    return prediction
result=""    
if st.button('Predict', on_click=predict):
    result=predict()
st.success("The Output is {}".format(result))


# In[ ]:





# In[ ]:




