import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset
data_path = os.path.join(base_dir, "..", "dataset", "kddcup.data_10_percent_corrected")
data = pd.read_csv(data_path, header=None)

# Feature Engineering: Rolling Mean & Standard Deviation & Trend
data['src_bytes_roll_mean'] = data[4].rolling(window=10, min_periods=1).mean()
data['src_bytes_roll_std'] = data[4].rolling(window=10, min_periods=1).std().fillna(0)
data['src_bytes_trend'] = data[4].diff().fillna(0)

# Select features (must match frontend)
# duration (0), src_bytes (4), dst_bytes (5), count (22), srv_count (23)
X = data[[0, 4, 5, 22, 23, 'src_bytes_roll_mean', 'src_bytes_roll_std', 'src_bytes_trend']].copy()
X.columns = X.columns.astype(str)
y = data.iloc[:, -1]

# Convert labels
y = y.apply(lambda x: 0 if x == 'normal.' else 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join(base_dir, "..", "models", "model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print("✅ Model trained and saved successfully!")