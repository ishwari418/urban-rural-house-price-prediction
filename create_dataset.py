import pandas as pd
import numpy as np
import os

def create_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    data = {
        'area': np.random.randint(500, 5000, num_samples),
        'bedrooms': np.random.randint(1, 6, num_samples),
        'bathrooms': np.random.randint(1, 4, num_samples),
        'stories': np.random.randint(1, 4, num_samples),
        'parking': np.random.randint(0, 4, num_samples),
        'year_built': np.random.randint(1990, 2024, num_samples),
        'mainroad': np.random.choice(['yes', 'no'], num_samples),
        'guestroom': np.random.choice(['yes', 'no'], num_samples),
        'basement': np.random.choice(['yes', 'no'], num_samples),
        'hotwaterheating': np.random.choice(['yes', 'no'], num_samples),
        'airconditioning': np.random.choice(['yes', 'no'], num_samples),
        'furnishingstatus': np.random.choice(['furnished', 'semi-furnished', 'unfurnished'], num_samples),
        'location_type': np.random.choice(['Urban', 'Rural'], num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Prices (in INR)
    # Urban Properties: Higher sq ft cost, higher importance for AC/Mainroad
    # Rural Properties: Lower sq ft cost, lower importance for luxury items
    
    def calculate_price(row):
        is_urban = row['location_type'] == 'Urban'
        
        if is_urban:
            base = 2500000
            area_weight = 5000
            ac_weight = 1000000
            road_weight = 800000
        else:
            base = 1000000
            area_weight = 2000
            ac_weight = 400000
            road_weight = 200000
            
        price = base + \
                (row['area'] * area_weight) + \
                (row['bedrooms'] * 300000) + \
                (row['bathrooms'] * 400000) + \
                (row['stories'] * 600000) + \
                (row['parking'] * 300000) + \
                (road_weight if row['mainroad'] == 'yes' else 0) + \
                (300000 if row['guestroom'] == 'yes' else 0) + \
                (400000 if row['basement'] == 'yes' else 0) + \
                (200000 if row['hotwaterheating'] == 'yes' else 0) + \
                (ac_weight if row['airconditioning'] == 'yes' else 0) + \
                (800000 if row['furnishingstatus'] == 'furnished' else 
                 400000 if row['furnishingstatus'] == 'semi-furnished' else 0)
        
        # Noise
        return price + np.random.normal(0, 50000)

    df['price'] = df.apply(calculate_price, axis=1)
    
    # Save
    output_path = os.path.join('house-price-prediction', 'data', 'housing.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created Location-Aware Indian House Price Dataset at {output_path}")

if __name__ == "__main__":
    create_synthetic_data()
