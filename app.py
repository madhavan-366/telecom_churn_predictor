import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# --- 1. LOAD THE SAVED XGBOOST MODEL AND SCALER ---
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler
model, scaler = load_model()

# --- 2. DEFINE THE COLUMNS FOR THE XGBOOST MODEL ---
MODEL_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'monthly_to_tenure_ratio', 'gender_Male', 'Partner_Yes',
    'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service',
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'tenure_group_Established', 'tenure_group_Veteran',
    'has_premium_services'
]

# --- 3. CONFIGURE THE STREAMLIT PAGE ---
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")
st.title("🚀 Telecom Churn Prediction Platform")

# --- 4. CREATE TABS FOR DIFFERENT MODES ---
tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📂 Batch Analysis Dashboard"])

# --- TAB 1: Simplified Single Customer Prediction ---
with tab1:
    st.header("Predict Churn for an Individual Customer")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Personal Details")
        senior_citizen = st.radio("Senior Citizen?", ("No", "Yes"), key="single_senior")
        partner = st.radio("Has Partner?", ("No", "Yes"), key="single_partner")
        dependents = st.radio("Has Dependents?", ("No", "Yes"), key="single_dependents")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1, key="single_tenure")
    
    with col2:
        st.markdown("##### Service & Contract")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="single_contract")
        internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], key="single_internet")
        paperless_billing = st.radio("PaperlessBilling?", ("No", "Yes"), key="single_paperless")
    
    with col3:
        st.markdown("##### Charges & Payment")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0, format="%.2f", key="single_monthly")
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="single_payment")

    if st.button("Predict Customer Churn", type="primary"):
        def create_feature_set(tenure, monthly_charges, contract, internet_service, paperless_billing, payment_method, senior_citizen, partner, dependents):
            user_input = {
                'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0, 'tenure': tenure, 'MonthlyCharges': monthly_charges,
                'TotalCharges': tenure * monthly_charges, 'monthly_to_tenure_ratio': monthly_charges / tenure if tenure > 0 else 0,
                'gender_Male': 0, 'Partner_Yes': 1 if partner == 'Yes' else 0, 'Dependents_Yes': 1 if dependents == 'Yes' else 0,
                'PhoneService_Yes': 1, 'MultipleLines_No phone service': 0, 'MultipleLines_Yes': 0,
                'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0, 'InternetService_No': 1 if internet_service == 'No' else 0,
                'OnlineSecurity_No internet service': 1 if internet_service == 'No' else 0, 'OnlineSecurity_Yes': 0,
                'OnlineBackup_No internet service': 1 if internet_service == 'No' else 0, 'OnlineBackup_Yes': 0,
                'DeviceProtection_No internet service': 1 if internet_service == 'No' else 0, 'DeviceProtection_Yes': 0,
                'TechSupport_No internet service': 1 if internet_service == 'No' else 0, 'TechSupport_Yes': 0,
                'StreamingTV_No internet service': 1 if internet_service == 'No' else 0, 'StreamingTV_Yes': 0,
                'StreamingMovies_No internet service': 1 if internet_service == 'No' else 0, 'StreamingMovies_Yes': 0,
                'Contract_One year': 1 if contract == 'One year' else 0, 'Contract_Two year': 1 if contract == 'Two year' else 0,
                'PaperlessBilling_Yes': 1 if paperless_billing == 'Yes' else 0,
                'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
                'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
                'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
                'tenure_group_Established': 1 if 12 < tenure <= 48 else 0, 'tenure_group_Veteran': 1 if tenure > 48 else 0,
                'has_premium_services': 0
            }
            df = pd.DataFrame([user_input])
            return df.reindex(columns=MODEL_COLUMNS, fill_value=0)

        initial_df = create_feature_set(tenure, monthly_charges, contract, internet_service, paperless_billing, payment_method, senior_citizen, partner, dependents)
        scaled_df = scaler.transform(initial_df)
        initial_proba = model.predict_proba(scaled_df)[0][1]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        if initial_proba > 0.5:
            st.error(f"**This customer is likely to CHURN.** (Probability: {initial_proba:.2%})")
        else:
            st.success(f"**This customer is likely to STAY.** (Probability: {initial_proba:.2%})")

# --- TAB 2: Batch Analysis Dashboard (Unaffected) ---
with tab2:
    st.header("Batch Customer Churn Analysis")
    with st.expander("ℹ️ Click here for file format instructions and a sample file"):
        st.info("""**Your CSV file must contain all original columns for the model to work correctly.**""")
        sample_data = {'customerID': ['1234-ABCDE'], 'gender': ['Female'], 'SeniorCitizen': [0], 'Partner': ['Yes'], 'Dependents': ['No'], 'tenure': [1], 'PhoneService': ['No'], 'MultipleLines': ['No phone service'], 'InternetService': ['DSL'], 'OnlineSecurity': ['No'], 'OnlineBackup': ['Yes'], 'DeviceProtection': ['No'], 'TechSupport': ['No'], 'StreamingTV': ['No'], 'StreamingMovies': ['No'], 'Contract': ['Month-to-month'], 'PaperlessBilling': ['Yes'], 'PaymentMethod': ['Electronic check'], 'MonthlyCharges': [29.85], 'TotalCharges': [29.85]}
        sample_df = pd.DataFrame(sample_data)
        st.download_button(label="Download Sample CSV Template", data=sample_df.to_csv(index=False).encode('utf-8'), file_name='sample_churn_data.csv', mime='text/csv')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(batch_df.head())
        
        if st.button("Predict Churn", type="primary"):
            try:
                original_df = batch_df.copy()
                processed_df = original_df.copy()
                
                processed_df['TotalCharges'] = pd.to_numeric(processed_df['TotalCharges'], errors='coerce')
                processed_df.dropna(inplace=True)
                processed_df['tenure_group'] = pd.cut(processed_df['tenure'], bins=[0, 12, 48, 100], labels=['New', 'Established', 'Veteran'])
                processed_df['monthly_to_tenure_ratio'] = np.where(processed_df['tenure'] > 0, processed_df['MonthlyCharges'] / processed_df['tenure'], 0)
                categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
                processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
                processed_df['has_premium_services'] = processed_df.apply(lambda row: 1 if (row.get('OnlineSecurity_Yes', 0) == 1 or row.get('OnlineBackup_Yes', 0) == 1 or row.get('DeviceProtection_Yes', 0) == 1 or row.get('TechSupport_Yes', 0) == 1) else 0, axis=1)
                processed_df = processed_df.reindex(columns=MODEL_COLUMNS, fill_value=0)
                
                scaled_batch = scaler.transform(processed_df)
                batch_predictions = model.predict(scaled_batch)
                batch_probabilities = model.predict_proba(scaled_batch)[:, 1]

                def segment_customer(prob):
                    if prob > 0.7: return 'At-Risk'
                    elif prob > 0.3: return 'Could Churn'
                    else: return 'Loyal'
                
                original_df['Churn_Probability'] = batch_probabilities
                original_df['Prediction'] = ['Churn' if p == 1 else 'Stay' for p in batch_predictions]
                original_df['Segment'] = original_df['Churn_Probability'].apply(segment_customer)
                
                st.markdown("---")
                st.header("Churn Analysis Dashboard")
                
                segment_counts = original_df['Segment'].value_counts()
                loyal_count = segment_counts.get('Loyal', 0)
                could_churn_count = segment_counts.get('Could Churn', 0)
                at_risk_count = segment_counts.get('At-Risk', 0)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div style="background-color: #28a745; border-radius: 10px; padding: 20px; color: white; text-align: center;"><h3>Loyal Customers</h3><h2>{loyal_count}</h2></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div style="background-color: #ffc107; border-radius: 10px; padding: 20px; color: black; text-align: center;"><h3>Could Churn</h3><h2>{could_churn_count}</h2></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div style="background-color: #dc3545; border-radius: 10px; padding: 20px; color: white; text-align: center;"><h3>At-Risk Customers</h3><h2>{at_risk_count}</h2></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                viz_col1, viz_col2 = st.columns(2)
                with viz_col1:
                    fig_pie = px.pie(original_df, names='Segment', title='Customer Segment Distribution', color='Segment', color_discrete_map={'Loyal': '#28a745', 'Could Churn': '#ffc107', 'At-Risk': '#dc3545'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with viz_col2:
                    churners_df = original_df[original_df['Prediction'] == 'Churn']
                    fig_bar = px.histogram(churners_df, x='Contract', title='Top Churn Driver: Contract Type', color='Contract', labels={'Contract': 'Contract Type for Churners'})
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("---")
                st.header("Detailed Prediction Results")
                st.dataframe(original_df[['customerID', 'Churn_Probability', 'Prediction', 'Segment']])
                
                @st.cache_data
                def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
                csv = convert_df_to_csv(original_df)
                st.download_button(label="Download Full Results as CSV", data=csv, file_name='churn_predictions_with_segments.csv', mime='text/csv')

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please ensure your CSV file is formatted correctly and contains the necessary columns.")

