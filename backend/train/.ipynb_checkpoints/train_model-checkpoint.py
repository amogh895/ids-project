import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("../dataset/kddcup.data_10_percent.gz", header=None)

# Select features (must match frontend)
X = data.iloc[:, [0, 4, 5]]   # duration, src_bytes, dst_bytes
y = data.iloc[:, -1]

# Convert labels
y = y.apply(lambda x: 0 if x == 'normal.' else 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/model.pkl")

print("✅ Model trained and saved successfully!")