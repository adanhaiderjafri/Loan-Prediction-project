import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = None
try:
    with open('loan_approval_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    try:
        with open(r"C:\Users\DELL\Downloads\loan_approval_model.pkl", 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("❌ Could not find the model file.")

if model:
    st.title('🏦 Loan Approval Prediction System')

    st.write("""
    This app predicts whether your loan will be approved.  
    Fill out a few important details below.
    """)

    # Important Inputs from User
    credit_history = st.selectbox(
        'Credit History (Have you repaid past loans?)',
        ['Good (1)', 'Bad (0)']
    )

    total_income_log = st.number_input(
        'Total Income (Log Value)\n(Higher income increases approval chance)',
        min_value=0.0, max_value=20.0, value=8.5
    )

    loan_amount_log = st.number_input(
        'Loan Amount Requested (Log Value)\n(Higher loans may reduce approval chance)',
        min_value=0.0, max_value=20.0, value=4.5
    )

    property_area = st.selectbox(
        'Property Area (Where do you live?)',
        ['Urban', 'Semiurban', 'Rural']
    )

    # Convert user inputs
    credit_history_num = 1 if credit_history == 'Good (1)' else 0
    property_area_num = {'Urban': 0, 'Semiurban': 1, 'Rural': 2}[property_area]

    # Fill remaining features with default values
    input_data = {
        'Gender': 1,                    # Default: Male
        'Married': 1,                   # Default: Married
        'Dependents': 0,                # Default: No dependents
        'Education': 1,                 # Default: Graduate
        'Self_Employed': 0,             # Default: Not self-employed
        'Credit_History': credit_history_num,
        'Property_Area': property_area_num,
        'ApplicantIncomeLog': 7.5,      # Default log income
        'LoanAmountLog': loan_amount_log,
        'Loan_Amount_Term_Log': 5.5,    # Default loan term
        'Total_Income_Log': total_income_log
    }

    # Create DataFrame
    features = pd.DataFrame([input_data])

    # Predict
    if st.button('Predict Loan Approval'):
        prediction = model.predict(features)
        try:
            probability = model.predict_proba(features)
        except AttributeError:
            probability = None

        if prediction[0] == 1:
            st.success('✅ Loan Approved!')
            if probability is not None:
                st.write(f'Probability of approval: {probability[0][1]:.2f}')
        else:
            st.error('❌ Loan Not Approved')
            if probability is not None:
                st.write(f'Probability of rejection: {probability[0][0]:.2f}')

        # Optional: show importance
        if hasattr(model, "feature_importances_"):
            st.subheader('🔍 Feature Importance')
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': features.columns,
                'Importance': importance
            })
            st.bar_chart(importance_df.set_index('Feature'))

