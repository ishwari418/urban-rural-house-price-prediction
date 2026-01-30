import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'house-price-prediction'))

import pandas as pd
import joblib

# Load models
urban_model = joblib.load('house-price-prediction/model/urban_model.pkl')
rural_model = joblib.load('house-price-prediction/model/rural_model.pkl')

# Test Urban prediction
urban_test = pd.DataFrame({
    'area': [2000],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'parking': [2],
    'year_built': [2020],
    'mainroad': ['yes'],
    'guestroom': ['yes'],
    'basement': ['yes'],
    'hotwaterheating': ['yes'],
    'airconditioning': ['yes'],
    'furnishingstatus': ['furnished']
})

# Test Rural prediction
rural_test = pd.DataFrame({
    'area': [3000],
    'bedrooms': [4],
    'bathrooms': [2],
    'stories': [1],
    'parking': [1],
    'year_built': [2015],
    'mainroad': ['no'],
    'guestroom': ['no'],
    'basement': ['no'],
    'hotwaterheating': ['no'],
    'airconditioning': ['no'],
    'furnishingstatus': ['unfurnished']
})

urban_price = urban_model.predict(urban_test)[0]
rural_price = rural_model.predict(rural_test)[0]

print("=== PREDICTION ACCURACY TEST ===\n")
print(f"Urban House (2000 sqft, 3BR, all amenities, furnished):")
print(f"  Predicted: ₹{urban_price:,.0f}")
print(f"  Expected Range: ₹25-35 Lakhs")
print(f"  Status: {'✓ PASS' if 2500000 <= urban_price <= 3500000 else '✗ FAIL'}\n")

print(f"Rural House (3000 sqft, 4BR, basic amenities, unfurnished):")
print(f"  Predicted: ₹{rural_price:,.0f}")
print(f"  Expected Range: ₹40-60 Lakhs")
print(f"  Status: {'✓ PASS' if 4000000 <= rural_price <= 6000000 else '✗ FAIL'}\n")

# Sanity check: Urban should be more expensive per sqft
urban_per_sqft = urban_price / 2000
rural_per_sqft = rural_price / 3000

print(f"Price per sq ft comparison:")
print(f"  Urban: ₹{urban_per_sqft:,.0f}/sqft")
print(f"  Rural: ₹{rural_per_sqft:,.0f}/sqft")
print(f"  Urban > Rural: {'✓ PASS' if urban_per_sqft > rural_per_sqft else '✗ FAIL'}")
