import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Sample dataset (you can replace with real dataset)
data = pd.DataFrame({
    "hour": [8, 9, 17, 18, 22, 14, 7, 19, 12, 16],
    "day": ["Weekday", "Weekday", "Weekday", "Weekday", "Weekend", "Weekend", "Weekday", "Weekend", "Weekday", "Weekend"],
    "weather": ["Clear", "Rain", "Fog", "Clear", "Rain", "Clear", "Fog", "Rain", "Clear", "Fog"],
    "road_condition": ["Good", "Moderate", "Bad", "Good", "Bad", "Good", "Moderate", "Bad", "Good", "Moderate"],
    "traffic": ["Low", "High", "High", "High", "Medium", "Low", "Medium", "High", "Low", "Medium"]
})

# Encode
X = data.drop("traffic", axis=1)
y = data["traffic"]

X = pd.get_dummies(X)

# Save columns
columns = X.columns

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Models
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
lr = LogisticRegression(max_iter=1000)

# Ensemble
model = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='hard'
)

model.fit(X_train_scaled, y_train)

# Save everything
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(columns, "columns.pkl")

print("✅ Model trained and saved!")