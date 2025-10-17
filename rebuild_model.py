# rebuild_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

print("Starting definitive model rebuild process...")

# 1. Load and Clean Data
df = pd.read_csv('Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df = df.drop('customerID', axis=1)

# 2. Feature Engineering
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 48, 100], labels=['New', 'Established', 'Veteran'])
df['monthly_to_tenure_ratio'] = np.where(df['tenure'] > 0, df['MonthlyCharges'] / df['tenure'], 0)

# 3. Prepare Data for Model
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_processed['has_premium_services'] = df_processed.apply(
    lambda row: 1 if (row.get('OnlineSecurity_Yes', 0) == 1 or
                     row.get('OnlineBackup_Yes', 0) == 1 or
                     row.get('DeviceProtection_Yes', 0) == 1 or
                     row.get('TechSupport_Yes', 0) == 1) else 0,
    axis=1
)

# 4. Train Model
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

# --- THIS IS THE NEW LINE ---
# Print the final list of columns in the correct order. We will copy this into our app.
print("\nCOPY THIS LIST OF COLUMNS INTO APP.PY:")
print(list(X.columns))
print("\n")
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# 5. Evaluate and Save
predictions = model.predict(X_test_scaled)
print(f"Final Model Accuracy: {accuracy_score(y_test, predictions):.4f}")
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nSUCCESS: Final 'churn_model.pkl' and 'scaler.pkl' have been saved!")


