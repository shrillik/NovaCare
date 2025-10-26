import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

# Load telecom plans dataset
plans = pd.read_csv("VI_AIRTEL_PLANS.csv")

# Fill missing values
plans.fillna(0, inplace=True)

# Create features for ML
plans['TOTAL_DATA_PER_DAY'] = plans['DATA_PER_DAY'] + plans['ADDITIONAL DATA']
plans['TOTAL_SMS'] = plans['SMS_PER_DAY'] + plans['ADDITIONAL_SMS']

# Select features to train model on
feature_columns = ['OFFER_PRICE', 'VALIDITY', 'TOTAL_DATA_PER_DAY', 'TOTAL_SMS', 'COST_PER_DAY']
X = plans[feature_columns]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train kNN model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_scaled)

# Save the model and scaler for later use
joblib.dump(knn, "plans_knn_model.pkl")
joblib.dump(scaler, "plans_scaler.pkl")
plans.to_csv("plans_with_features.csv", index=False)

print("Model training complete! kNN model saved as 'plans_knn_model.pkl'.")
