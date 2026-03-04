import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------- SAMPLE TRAINING DATA ----------------
data = {
    "crime_type": [
        "Assault", "Cyber Crime", "Harassment",
        "Theft", "Fraud", "Robbery",
        "Other"
    ],
    "priority": [
        "High", "High", "High",
        "Medium", "Medium", "Medium",
        "Low"
    ]
}

df = pd.DataFrame(data)

# ---------------- ENCODING ----------------
le_crime = LabelEncoder()
le_priority = LabelEncoder()

df["crime_type"] = le_crime.fit_transform(df["crime_type"])
df["priority"] = le_priority.fit_transform(df["priority"])

# ---------------- MODEL ----------------
model = DecisionTreeClassifier()
model.fit(df[["crime_type"]], df["priority"])

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "priority_model.pkl")
joblib.dump(le_crime, "crime_encoder.pkl")
joblib.dump(le_priority, "priority_encoder.pkl")

print("✅ ML model trained and saved")