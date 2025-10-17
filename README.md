Advanced Telecom Customer Churn Prediction Platform

An end-to-end data science project to predict customer churn and segment customers into actionable tiers using an advanced machine learning model.

Live Application

You can access and interact with the live application deployed on Streamlit Community Cloud here:
https://telecomchurnpredictor-qvdsrerdtjpctfhozzs5ag.streamlit.app/

Project Overview

This project addresses a critical business challenge in the telecom industry — proactively identifying and retaining customers at risk of churning.
Using customer demographic and service data, the application predicts the probability of a customer churning.

The core engine is an XGBoost Classifier model trained on a rich set of features, including engineered variables like tenure_group and monthly_to_tenure_ratio.
Predictions are then used to classify customers into three actionable tiers:

At-Risk

Could Churn

Loyal

These segments enable targeted retention campaigns, personalized offers, and proactive customer support to reduce revenue loss.

The entire pipeline — from data cleaning and feature engineering to prediction and visualization — is wrapped into a user-friendly Streamlit web application.

Key Features

Single Customer Prediction:
Enter a customer's details through an intuitive form and instantly receive their churn probability score.

"What-If" Scenario Planner:
After an initial prediction, interact with the app to see how retention offers (like changing a contract from "Month-to-month" to "One year") could decrease a customer's churn risk in real-time.

Batch Prediction & Analysis:
Upload a raw customer CSV file to generate churn predictions and segments for thousands of customers at once.

Automated Feature Engineering:
The app automatically computes insightful features like tenure_group and monthly_to_tenure_ratio behind the scenes to power the advanced XGBoost model.

Interactive Dashboard:
View batch results through a comprehensive dashboard with visual KPIs, customer segment distribution (pie chart), and the top churn drivers for at-risk customers (bar chart).

Color-Coded Results:
Segments are clearly marked with Red (At-Risk), Yellow (Could Churn), and Green (Loyal) for immediate visual clarity.

Data Export:
Download the complete prediction results and customer segments as a CSV file for use in marketing platforms or further analysis.

Tech Stack

Category

Tools/Frameworks

Language

Python 3.9+

App Framework

Streamlit

Machine Learning

Scikit-learn, XGBoost

Data Manipulation

Pandas, NumPy

Visualization

Plotly Express

Model Serialization

Joblib

How to Use the App

Single Prediction & Scenario Planning

Go to the Single Customer Prediction & Scenario Planning tab.

Enter the customer's details using the input fields.

Click Predict Customer Churn.

View the initial churn probability.

Use the "Offer a new contract" dropdown to see how a retention offer could change the outcome.

Batch Prediction

Go to the Batch Analysis Dashboard tab.

Download the sample CSV to check the required format.

Upload your own customer data file.

Click Analyze Batch File.

View the full interactive dashboard with KPIs, charts, and the detailed results table.

How to Run Locally

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/telecom-churn-predictor-app.git](https://github.com/YOUR_USERNAME/telecom-churn-predictor-app.git)
cd telecom-churn-predictor-app


Create and activate a virtual environment (recommended):

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux


Install the required dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py
