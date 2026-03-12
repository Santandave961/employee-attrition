import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

lr_model = joblib.load('lr_attrition_model.pkl')
rf_model = joblib.load('rf_attrition_model.pkl')
scaler = joblib.load('scaler.pkl')

with open('feature_cols.json') as f:
    feature_cols = json.load(f)

# -- PAGE CONFIG ---
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="&&", layout="wide")
st.title(" Employee Attrition Risk Predictor")
st.markdown("Enter employee details below  to predict attrition risk.")

# --- SIDE BAR ---
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose Model:",
    ["Logistic Regression (Better Recall)", "Random Forest (Better Accuracy)"]
)

## --- INPUT FORM ---
st.subheader("Employee Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 60, 35)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    overtime = st.selectbox("OverTime", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    years_at_company = st.slider("Years at Company", 0, 40, 5)

with col2:
    distance_from_home = st.slider("Distance From Home", 1, 29, 10)
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
    num_companies_worked = st.slider("Num Companies Worked", 0, 9, 2)
    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
    years_in_current_role = st.slider("Years In Current Role", 0, 18, 3)

with col3:
    marital_status = st.selectbox("Marital Status", [0, 1, 2],
                                  format_func=lambda x: ["Divorced", "Married", "Single"][x])
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
    years_with_curr_mananger = st.slider("Years with Current Manager", 0, 17, 3)
    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)

 # --- PREDICT BUTTON ---
if st.button("Predict Attrition Risk", use_container_width=True): 

    input_dict = {col: 0 for col in feature_cols}

    # Fill known input
    input_dict.update({
        'Age': age,
        'MonthlyIncome': monthly_income,
        'JobSatisfaction': job_satisfaction,
        'OverTime': overtime,
        'YearsAtCompany': years_at_company,
        'DistanceFromHome': distance_from_home,
        'EnvironmentSatisfaction': environment_satisfaction,
        'NumCompaniesWorked': num_companies_worked,
        'WorkLifeBalance': work_life_balance,
        'YearsInCurrentRole': years_in_current_role,
        'MaritalStatus': marital_status,
        'TotalWorkingYears': total_working_years,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_mananger,
        'TrainingTimesLastYear': training_times_last_year,
    })

    input_df = pd.DataFrame([input_dict])[feature_cols]
    input_scaled = scaler.transform(input_df)

    # Select model
    chosen_model = lr_model if "Logistic" in model_choice else rf_model
    prediction = chosen_model.predict(input_scaled)[0]
    probability = chosen_model.predict_proba(input_scaled)[0][1]


    # ---RESULT ---

    st.divider()
    st.subheader("Prediction Result")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        if prediction == 1:
            st.error(f" HIGH RISK - This employee is likely to leave")
        else:
            st.success(f" LOW RISK - This employee is likely to stay")
    
    with col_r2:
        st.metric("Attrition Probability", f"{probability:.1%}")
        st.progress(float(probability))
                  
