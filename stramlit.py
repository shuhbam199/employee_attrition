import streamlit as st
import pickle
import pandas as pd
import numpy as np

#st.write("""
# My first app
#Hello *world!*
#""")
# Load the model and encoder
with open('rf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('ce_target3.pkl', 'rb') as g:
    encode = pickle.load(g)


st.title("🚀 Employee Attrition Predictor")

st.markdown("""
Enter the following details about the employee to predict whether they are likely to leave.
""")

# Example features – customize based on your X_test/X_sm features
Age = st.slider("Age", 18, 60, 30)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
MonthlyRate = st.number_input("Monthly Rate", min_value=100, max_value=100000, value=1000)
JobSatisfaction = st.selectbox("Job Satisfaction (1: Low, 4: High)", [1, 2, 3, 4])
WorkLifeBalance = st.selectbox("WorkLifeBalance (1: Low, 4: High)", [1, 2, 3, 4])
#YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
OverTime = st.selectbox("OverTime", [1, 0])
#MaritalStatus_Single = st.selectbox("Single",[1, 0])
MaritalStatus_Married = st.selectbox("Married",[1, 0])
EducationField = st.selectbox("Education Field", [
    'Life Sciences', 'Other', 'Medical', 'Marketing',
    'Technical Degree', 'Human Resources'])
StockOptionLevel = st.selectbox('StockOptionLevel', [0,1,2,3])
JobRole = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                     'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                     'Sales Representative', 'Research Director', 'Human Resources'])


# Apply target encoding
# First encode the categorical fields


# Build the full input with all features
df_input = pd.DataFrame([{
    'MaritalStatus_Married': MaritalStatus_Married,
    'StockOptionLevel': StockOptionLevel,
    'JobSatisfaction': JobSatisfaction,
    'MonthlyIncome': MonthlyIncome,
    'MonthlyRate': MonthlyRate,
    'OverTime': OverTime,
    'Age': Age,
   'EducationField': EducationField,
    'WorkLifeBalance': WorkLifeBalance,
    'JobRole': JobRole
}])

# Apply TargetEncoder
df_encoded = encode.transform(df_input)

# Ensure correct column order (same as training)

# Predict
if st.button("Predict"):
    prediction = model.predict(df_encoded)[0]
    if prediction == 1:
        st.error(" This employee is likely to leave the company.")
    else:
        st.success("This employee is likely to stay.")
