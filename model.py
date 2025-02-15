import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
file_path = "crop_yield_data.csv"  
df = pd.read_csv(file_path)

# Handle missing values
df.dropna(inplace=True)

# Encode Categorical Features
label_enc_area = LabelEncoder()
label_enc_item = LabelEncoder()

df['Area'] = label_enc_area.fit_transform(df['Area'])
df['Item'] = label_enc_item.fit_transform(df['Item'])

# Save encoders for later use in Flask app
with open("label_encoders.pkl", "wb") as f:
    pickle.dump((label_enc_area, label_enc_item), f)

# Define Features and Target
X = df.drop(columns=['hg/ha_yield'])  
y = df['hg/ha_yield']  
print("Training Features:", X.columns)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open("crop_yield_model.pkl", "wb"))

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
