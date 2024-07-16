#!/usr/bin/env python
# coding: utf-8

# In[22]:


import streamlit as st
import pandas as pd
import joblib

# Load the trained linear regression model
model = joblib.load("linear_regression_model.pkl")

# Streamlit UI
st.title("Salary Prediction App")

# Sidebar with user input fields
st.sidebar.header("User Input")

age = st.sidebar.slider("Age", min_value=18, max_value=65, value=30)
hourly_rate = st.sidebar.slider("Hourly Rate", min_value=30, max_value=100, value=50)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike", min_value=0, max_value=25, value=10)
performance_rating = st.sidebar.slider("Performance Rating", min_value=1, max_value=4, value=3)
total_working_years = st.sidebar.slider("Total Working Years", min_value=0, max_value=40, value=10)

# Checkboxes for Department and JobRole
st.sidebar.subheader("Department:")

department_rd = st.sidebar.checkbox("Research & Development", key="department_rd")
department_sales = st.sidebar.checkbox("Sales", key="department_sales")

st.sidebar.subheader("Gender:")
gender_female = st.sidebar.checkbox("Female", key="gender_female")
gender_male = st.sidebar.checkbox("Male", key="gender_male")

st.sidebar.subheader("Job Role:")
job_role_hcr = st.sidebar.checkbox("Healthcare Representative", key="job_role_hcr")
job_role_lab_technician = st.sidebar.checkbox("Laboratory Technician", key="job_role_lab_technician")
job_role_manager = st.sidebar.checkbox("Manager", key="job_role_manager")
job_role_manufacturing_director = st.sidebar.checkbox("Manufacturing Director", key="job_role_manufacturing_director")
job_role_research_director = st.sidebar.checkbox("Research Director", key="job_role_research_director")
job_role_research_scientist = st.sidebar.checkbox("Research Scientist", key="job_role_research_scientist")
job_role_sales_executive = st.sidebar.checkbox("Sales Executive", key="job_role_sales_executive")
job_role_sales_representative = st.sidebar.checkbox("Sales Representative", key="job_role_sales_representative")
# Calculate Monthly Income
if st.sidebar.button("Predict Monthly Income"):
    # Prepare user input as a DataFrame
    user_input = pd.DataFrame({
        "Age": [age],
        "HourlyRate": [hourly_rate],
        "PercentSalaryHike": [percent_salary_hike],
        "PerformanceRating": [performance_rating],
        "TotalWorkingYears": [total_working_years],
        "Department_Research & Development": [1 if department_rd else 0],
        "Department_Sales": [1 if department_sales else 0],
        "Gender_Female": [1 if gender_female else 0],
        "Gender_Male": [1 if gender_male else 0],
        "JobRole_Healthcare Representative": [1 if job_role_hcr else 0],
        "JobRole_Laboratory Technician": [1 if job_role_lab_technician else 0],
        "JobRole_Manager": [1 if job_role_manager else 0],
        "JobRole_Manufacturing Director": [1 if job_role_manufacturing_director else 0],
        "JobRole_Research Director": [1 if job_role_research_director else 0],
        "JobRole_Research Scientist": [1 if job_role_research_scientist else 0],
        "JobRole_Sales Executive": [1 if job_role_sales_executive else 0],
        "JobRole_Sales Representative": [1 if job_role_sales_representative else 0]
    })

    # Make salary prediction
    predicted_salary = model.predict(user_input)[0]

    # Display the prediction
    st.write("## Predicted Monthly Income")
    st.write(f"The predicted monthly income is: Rs {predicted_salary:.2f}")


# In[ ]:




