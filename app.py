from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load Trained Model and Encoders
model = pickle.load(open("crop_yield_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
label_enc_area, label_enc_item = label_encoders

# Load Dataset to Fetch Unique Values
df = pd.read_csv("crop_yield_data.csv")

# Get Unique Area and Crop Type (Item) Values
unique_areas = sorted(df['Area'].unique())
unique_items = sorted(df['Item'].unique())

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', areas=unique_areas, items=unique_items)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        area = request.form['area']
        item = request.form['item']
        year = int(request.form['year'])
        avg_rainfall = float(request.form['avg_rainfall'])
        pesticides = float(request.form['pesticides'])
        avg_temp = float(request.form['avg_temp'])

        # Encode categorical values
        area_encoded = label_enc_area.transform([area])[0]
        item_encoded = label_enc_item.transform([item])[0]

        # Prepare input features
        input_features = np.array([[area_encoded, item_encoded, year, avg_rainfall, pesticides, avg_temp]])

        # Predict crop yield
        prediction = model.predict(input_features)[0]

        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
