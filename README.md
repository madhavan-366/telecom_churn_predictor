# Telecom Customer Churn Prediction Platform

An end-to-end data science project to predict customer churn and segment customers into actionable tiers using an advanced machine learning model.

---

## Live Application

You can access and interact with the live application deployed on Streamlit Community Cloud here:  
https://telecomchurnpredictor-qvdsrerdtjpctfhozzs5ag.streamlit.app/
---

## Project Overview

This project addresses a critical business challenge in the telecom industry — proactively identifying and retaining customers at risk of churning.

Using customer demographic and service data, the application predicts the probability of a customer churning.

The core engine is an XGBoost Classifier model trained on a rich set of features, including engineered variables like:

- tenure_group  
- monthly_to_tenure_ratio  

Predictions are then used to classify customers into three actionable tiers:

- At-Risk  
- Could Churn  
- Loyal  

These segments enable targeted retention campaigns, personalized offers, and proactive customer support to reduce revenue loss.

The entire pipeline — from data cleaning and feature engineering to prediction and visualization — is wrapped into a user-friendly Streamlit web application.

---

## Key Features

### Single Customer Prediction
- Enter a customer's details through an intuitive form.  
- Instantly receive their churn probability score.

### Batch Prediction & Analysis
- Upload a raw customer CSV file.  
- Generate churn predictions and segments for thousands of customers at once.

### Automated Feature Engineering
- The app automatically computes features like tenure_group and monthly_to_tenure_ratio behind the scenes.

### Interactive Dashboard
- Visual KPIs, customer segment distribution (pie chart), and top churn drivers (bar chart).

### Color-Coded Results
- Red: At-Risk  
- Yellow: Could Churn  
- Green: Loyal  

### Data Export
- Download prediction results and customer segments as a CSV for marketing or further analysis.

---

## Tech Stack

| Category | Tools/Frameworks |
|-----------|------------------|
| Language | Python 3.9+ |
| App Framework | Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| Data Manipulation | Pandas, NumPy |
| Visualization | Plotly Express |
| Model Serialization | Joblib |

---

## How to Use the App

### Single Prediction & Scenario Planning
1. Go to the **Single Customer Prediction & Scenario Planning** tab.  
2. Enter the customer's details using the input fields.  
3. Click **Predict Customer Churn**.  
4. View the initial churn probability.  

### Batch Prediction
1. Go to the **Batch Analysis Dashboard** tab.  
2. Download the sample CSV to check the required format.  
3. Upload your own customer data file.  
4. Click **Predict Churn**.  
5. Explore KPIs, charts, and the results table interactively.

---

## How to Run Locally

## 1. Clone the Repository
    ```bash
    git clone https://github.com/madhavan-366/telecom_churn_predictor.git
    cd telecom-churn-predictor

## 2. Create and Activate a Virtual Environment (Recommended)
### For Windows
    ```bash
    python -m venv venv
    venv\Scripts\activate

### For Mac/Linux
    ```bash
    python -m venv venv
    source venv/bin/activate

### 3. Install the Required Dependencies
    ```bash
    pip install -r requirements.txt

### 4. Run the Streamlit App
    ```bash
    streamlit run app.py
