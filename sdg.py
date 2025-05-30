

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# --- Step 1: Generate Sample Data ---

dates = pd.date_range(start="2022-01-01", periods=30, freq="D").tolist()
regions = ["Region A", "Region B", "Region C"]

covid_records = []
weather_records = []

for region in regions:
    for i, date in enumerate(dates):
        covid_records.append({
            "Date": date,
            "Region": region,
            "cases": 80 + (i % 10) * 5  # cases: 80, 85, ..., 125
        })
        weather_records.append({
            "Date": date,
            "Region": region,
            "temperature": 25 + (i % 5),  # temp: 25 to 29
            "humidity": 60 - (i % 10)     # humidity: 60 to 51
        })

covid_data = pd.DataFrame(covid_records)
weather_data = pd.DataFrame(weather_records)

# Save sample CSVs
covid_data.to_csv("covid_cases.csv", index=False)
weather_data.to_csv("weather_data.csv", index=False)
print("Sample data saved as 'covid_cases.csv' and 'weather_data.csv'")

# --- Step 2: Load and Prepare Data ---

cases = pd.read_csv("covid_cases.csv")
weather = pd.read_csv("weather_data.csv")

# Merge datasets
data = pd.merge(cases, weather, on=["Date", "Region"])

# Create lag features
data['cases_last_week'] = data.groupby('Region')['cases'].shift(7)
data['cases_2_weeks_ago'] = data.groupby('Region')['cases'].shift(14)

# Drop rows with NaNs from lagging
data = data.dropna()

# Define target: outbreak if cases > 100
data['outbreak'] = (data['cases'] > 100).astype(int)

# Select features and label
features = ['cases_last_week', 'cases_2_weeks_ago', 'temperature', 'humidity']
X = data[features]
y = data['outbreak']

# --- Step 3: Train the Model ---

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "outbreak_rf_model.pkl")
print("Model trained and saved as 'outbreak_rf_model.pkl'")

# --- Optional: Display evaluation metrics ---
accuracy = model.score(X_test, y_test)
print(f" Test Accuracy: {accuracy:.2f}")
