import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import shap

# Set paths
DATA_PATH = os.path.join('house-price-prediction', 'data', 'housing.csv')
MODEL_DIR = os.path.join('house-price-prediction', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate_subset(df_subset, subset_name):
    print(f"\n--- Training {subset_name} Model ---")
    X = df_subset.drop(['price', 'location_type'], axis=1)
    y = df_subset['price']
    
    numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'year_built']
    categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"{name} -> R2: {r2:.4f}, MAE: {mae:.2f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_name = name
            
    print(f"Best {subset_name} Model: {best_name} (R2: {best_r2:.4f})")
    
    model_path = os.path.join(MODEL_DIR, f'{subset_name.lower()}_model.pkl')
    joblib.dump(best_model, model_path)
    
    # Explainability
    cat_features_encoded = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_features_encoded)
    
    if best_name == 'Random Forest':
        importances = best_model.named_steps['regressor'].feature_importances_
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        print(f"Top 5 {subset_name} Drivers:\n{imp_df.head(5)}")
    else:
        coefs = best_model.named_steps['regressor'].coef_
        imp_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
        imp_df['abs_coef'] = imp_df['coef'].abs()
        imp_df = imp_df.sort_values(by='abs_coef', ascending=False)
        print(f"Top 5 {subset_name} Drivers:\n{imp_df[['feature', 'coef']].head(5)}")

def train_all():
    df = pd.read_csv(DATA_PATH)
    urban_df = df[df['location_type'] == 'Urban']
    rural_df = df[df['location_type'] == 'Rural']
    
    train_and_evaluate_subset(urban_df, 'Urban')
    train_and_evaluate_subset(rural_df, 'Rural')

if __name__ == "__main__":
    train_all()
