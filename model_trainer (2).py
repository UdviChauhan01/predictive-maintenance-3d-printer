import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your data (replace with your actual file)
df = pd.read_csv("C:/Users/vedan/Desktop/pd_maintenance_3Dprinter/3d_printer_data.csv")  # or pd.read_json('your_data.json')

# Assume the last column is the target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model as pickle
joblib.dump(model, "C:/Users/vedan/Desktop/pd_maintenance_3Dprinter/printer_predictive_model.pkl")
print("Model saved as printer_predictive_model.pkl") 