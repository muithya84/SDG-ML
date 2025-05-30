# train_model.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create sample COVID case data
covid_data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=30),
    'Region': ['Region A'] * 30,
    'cases': [50, 60, 80, 90, 100, 120, 130, 110, 150, 180,
              170, 160, 140, 100, 90, 70, 60, 50, 40, 45,
              60, 75, 85, 95, 110, 130, 140, 155, 160, 170]
})

weather_data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=30),
    'Region': ['Region A'] * 30,
    'temperature': [25 + i % 5 for i in range(30)],
    'humidity': [60 + (i % 10) for i in range(30)]
})

# Save sample CSVs
os.makedirs("data", exist_ok=True)
covid_data.to_csv("data/covid_cases.csv", index=False)
weather_data.to_csv("data/weather_data.csv", index=False)

print("Sample data saved as 'covid_cases.csv' and 'weather_data.csv'")

# Load data
cases = pd.read_csv("data/covid_cases.csv")
weather = pd.read_csv("data/weather_data.csv")

# Merge datasets
data = pd.merge(cases, weather, on=["Date", "Region"])

# Feature engineering
data['cases_last_week'] = data['cases'].shift(7)
data['cases_2_weeks_ago'] = data['cases'].shift(14)
data = data.dropna()

# Define outbreak: if cases > 100
data['outbreak'] = (data['cases'] > 100).astype(int)

# Features and target
features = ['cases_last_week', 'cases_2_weeks_ago', 'temperature', 'humidity']
X = data[features]
y = data['outbreak']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained and saved as 'outbreak_rf_model.pkl'\n Test Accuracy: {accuracy:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/outbreak_rf_model.pkl")
