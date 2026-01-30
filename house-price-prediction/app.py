from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URBAN_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'urban_model.pkl')
RURAL_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'rural_model.pkl')

urban_model = joblib.load(URBAN_MODEL_PATH)
rural_model = joblib.load(RURAL_MODEL_PATH)

def format_indian_currency(amount):
    """Formats a number with commas for readability (e.g., ₹1,23,456)"""
    try:
        s = str(int(amount))
        if len(s) <= 3: return f"₹{s}"
        last_three = s[-3:]
        others = s[:-3]
        res = ""
        for i, char in enumerate(reversed(others)):
            if i != 0 and i % 2 == 0:
                res += ","
            res += char
        return f"₹{''.join(reversed(res))},{last_three}"
    except:
        return f"₹{int(amount):,}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate location type
        location = request.form.get('location_type')
        if not location or location not in ['Urban', 'Rural']:
            return render_template('index.html', 
                                 prediction_text='Error: Please select a valid location type (Urban or Rural)',
                                 input_data=request.form)
        
        # Get and validate inputs
        try:
            area = float(request.form.get('area', 0))
            bedrooms = int(request.form.get('bedrooms', 0))
            bathrooms = int(request.form.get('bathrooms', 0))
            stories = int(request.form.get('stories', 0))
            parking = int(request.form.get('parking', 0))
            year_built = int(request.form.get('year_built', 0))
        except (ValueError, TypeError):
            return render_template('index.html',
                                 prediction_text='Error: Please enter valid numeric values for all fields',
                                 input_data=request.form)
        
        # Validate ranges
        if not (300 <= area <= 10000):
            return render_template('index.html',
                                 prediction_text='Error: Area must be between 300 and 10,000 sq ft',
                                 input_data=request.form)
        
        if not (1 <= bedrooms <= 10):
            return render_template('index.html',
                                 prediction_text='Error: Bedrooms must be between 1 and 10',
                                 input_data=request.form)
        
        if not (1 <= bathrooms <= 8):
            return render_template('index.html',
                                 prediction_text='Error: Bathrooms must be between 1 and 8',
                                 input_data=request.form)
        
        if not (1 <= stories <= 5):
            return render_template('index.html',
                                 prediction_text='Error: Stories must be between 1 and 5',
                                 input_data=request.form)
        
        if not (0 <= parking <= 10):
            return render_template('index.html',
                                 prediction_text='Error: Parking must be between 0 and 10',
                                 input_data=request.form)
        
        if not (1950 <= year_built <= 2025):
            return render_template('index.html',
                                 prediction_text='Error: Year Built must be between 1950 and 2025',
                                 input_data=request.form)
        
        # Get categorical inputs
        mainroad = request.form.get('mainroad', 'no')
        guestroom = request.form.get('guestroom', 'no')
        basement = request.form.get('basement', 'no')
        hotwaterheating = request.form.get('hotwaterheating', 'no')
        airconditioning = request.form.get('airconditioning', 'no')
        furnishingstatus = request.form.get('furnishingstatus', 'unfurnished')
        
        # Create data dictionary
        data = {
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'parking': [parking],
            'year_built': [year_built],
            'mainroad': [mainroad],
            'guestroom': [guestroom],
            'basement': [basement],
            'hotwaterheating': [hotwaterheating],
            'airconditioning': [airconditioning],
            'furnishingstatus': [furnishingstatus]
        }
        
        input_df = pd.DataFrame(data)
        
        # Select model and generate explanation
        if location == 'Urban':
            prediction = urban_model.predict(input_df)[0]
            explanation = "Note: Urban prices are heavily influenced by Air Conditioning and Square Footage."
        else:
            prediction = rural_model.predict(input_df)[0]
            explanation = "Note: Rural prices depend more on total Area and Basic Amenities."
        
        # Ensure prediction is positive
        if prediction < 0:
            prediction = abs(prediction)
            
        formatted_price = format_indian_currency(prediction)
        
        return render_template('index.html', 
                             prediction_text=f'Estimated House Price: {formatted_price}',
                             explanation_text=explanation,
                             input_data=request.form)
                             
    except Exception as e:
        # Log the error for debugging
        print(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                             prediction_text='Error: Unable to process prediction. Please check your inputs and try again.',
                             input_data=request.form)

if __name__ == '__main__':
    # For production, use: gunicorn app:app
    # For development:
    app.run(host='127.0.0.1', port=5000, debug=True)
