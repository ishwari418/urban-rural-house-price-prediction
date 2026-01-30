# Urban vs Rural House Price Prediction using Machine Learning and Flask

A location-aware machine learning system that predicts house prices by differentiating between Urban and Rural property markets using separate regression models and a Flask web interface.

## Problem Statement

House price prediction is a critical task in the real estate industry. However, a one-size-fits-all model fails to capture the fundamental differences between Urban and Rural housing markets:

- **Urban Markets**: Prices are heavily influenced by proximity to amenities (AC, main road access), infrastructure quality, and space efficiency (price per sq. ft.).
- **Rural Markets**: Prices depend more on total land area, basic construction quality, and fundamental features rather than luxury amenities.

This project addresses this challenge by training **two separate regression models** to capture these distinct market behaviors.

## Key Features

- **Location-Aware Prediction**: Automatically routes user input to the appropriate model based on Urban or Rural selection
- **Dual Regression Models**: Separate optimized models for Urban and Rural properties
- **Feature Engineering**: Transformation of raw property data into meaningful predictive features
- **Model Explainability**: Feature importance analysis showing which factors drive prices in each market
- **Production-Ready Flask Application**: Clean web interface with input validation and error handling
- **Indian Market Focus**: Predictions formatted in Indian Rupees (₹) with culturally appropriate number formatting

## Dataset Description

The dataset contains **1,000 synthetic property records** split evenly between Urban and Rural locations. Each record includes the following features:

### Core Property Features
- **Area**: Total property area in square feet (500 - 5,000 sq ft)
- **Bedrooms**: Number of bedrooms (1 - 6)
- **Bathrooms**: Number of bathrooms (1 - 4)
- **Stories**: Number of floors/stories (1 - 4)
- **Parking**: Number of parking spaces (0 - 4)
- **Year Built**: Construction year (1990 - 2024)

### Location & Connectivity Features
- **Main Road Access**: Whether property faces a main road (`yes`/`no`)

### Amenities
- **Guest Room**: Presence of a dedicated guest room (`yes`/`no`)
- **Basement**: Presence of a basement (`yes`/`no`)
- **Air Conditioning**: Central or split AC availability (`yes`/`no`)
- **Hot Water Heating**: Availability of hot water heating system (`yes`/`no`)

### Property Characteristics
- **Furnishing Status**: Level of furnishing (`furnished`, `semi-furnished`, `unfurnished`)

### Location Type
- **Location Type**: Market category (`Urban` or `Rural`)

### Target Variable
- **Price**: Property value in Indian Rupees (₹)

## Feature Engineering

The following transformations were applied to the raw data:

1. **Property Age Calculation**: `year_built` is used as-is; models learn temporal patterns
2. **Dataset Separation**: Data split into Urban and Rural subsets for independent model training
3. **Categorical Encoding**: One-hot encoding applied to all binary (`yes`/`no`) and categorical features (`furnishingstatus`)
4. **Numerical Scaling**: StandardScaler applied to continuous variables for consistent model training
5. **Missing Value Imputation**: Median imputation for numerical features, constant fill for categorical

## Modeling Approach

### Why Separate Models?

Urban and Rural housing markets exhibit fundamentally different pricing patterns:
- Urban properties show high sensitivity to amenities and infrastructure
- Rural properties prioritize land area and basic construction over luxury features

A single model would either underfit one market or overfit the other. **Separate models ensure optimal accuracy for each market type.**

### Models Trained

For each market type (Urban and Rural), the following regression models were evaluated:

1. **Linear Regression**: Baseline model for interpretability
2. **Ridge Regression**: L2 regularization to prevent overfitting
3. **Lasso Regression**: L1 regularization for feature selection
4. **Random Forest Regressor**: Ensemble method capturing non-linear relationships

### Model Selection Criteria

- **R² Score**: Measures proportion of variance explained by the model
- **Mean Absolute Error (MAE)**: Average prediction error in Indian Rupees

The **Random Forest Regressor** was selected as the best model for both Urban and Rural datasets based on superior generalization and built-in feature importance capabilities.

## Model Explainability

Feature importance analysis was performed to understand which factors drive prices in each market:

### Urban Model: Top Influencing Features
1. **Area**: Total square footage (highest weight)
2. **Air Conditioning**: Major price differentiator in urban markets
3. **Main Road Access**: Significant premium for connectivity
4. **Furnishing Status**: Furnished properties command higher prices
5. **Stories**: Multi-story homes valued higher

### Rural Model: Top Influencing Features
1. **Area**: Dominant factor (even more important than in urban markets)
2. **Stories**: Construction quality indicator
3. **Bathrooms**: Basic infrastructure importance
4. **Bedrooms**: Functional capacity
5. **Parking**: Moderate influence

**Key Insight**: Air conditioning adds ₹10 Lakhs premium in Urban areas but only ₹4 Lakhs in Rural areas, validating the need for separate models.

## Web Application (Flask)

### Features
- **User Input Form**: Clean interface for entering property details
- **Location Selection**: Dropdown to choose Urban or Rural
- **Dynamic Model Selection**: Backend automatically routes to the appropriate model
- **Prediction Display**: Results shown in Indian Rupee format (₹) with comma-separated values
- **Contextual Explanation**: Brief note on key influencing features for the selected location type
- **Input Validation**: Comprehensive range checks and error handling

### User Workflow
1. User selects **Location Type** (Urban or Rural)
2. User enters property details (area, bedrooms, amenities, etc.)
3. System validates inputs and selects the appropriate model
4. Prediction is displayed with an explanation of key price drivers

## Project Structure

```
house-price-prediction/
├── app.py                    # Flask web application with routing and prediction logic
├── model/
│   ├── urban_model.pkl       # Trained Urban property model (Random Forest)
│   └── rural_model.pkl       # Trained Rural property model (Random Forest)
├── data/
│   └── housing.csv           # Training dataset (1000 records)
├── templates/
│   └── index.html            # Web interface HTML template
├── static/
│   └── style.css             # Professional UI styling
├── train_model.py            # Model training script with evaluation
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction/house-price-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Models (Optional)
```bash
# Models are already trained and saved in model/ directory
# To retrain from scratch:
python train_model.py
```

### Step 4: Run the Flask Application
```bash
python app.py
```

### Step 5: Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## Results & Insights

### Model Performance

| Metric | Urban Model | Rural Model |
|--------|-------------|-------------|
| R² Score | 0.99+ | 0.99+ |
| Best Algorithm | Random Forest | Random Forest |

Both models achieve high accuracy on their respective test sets, validating the separate modeling approach.

### Key Insights

1. **Urban Premium on Amenities**: Air conditioning, main road access, and furnishing status have 2-3x higher impact in urban areas
2. **Rural Area Dominance**: Square footage is the overwhelming price driver in rural markets, accounting for ~60% of variance
3. **Price per Sq Ft**: Urban properties average ₹9,752/sqft vs ₹3,251/sqft for rural properties (typical test case)
4. **Model Divergence**: A unified model would have misrepresented both markets; separate models provide tailored accuracy

## Limitations

This project has the following limitations that should be considered:

1. **Synthetic Dataset**: The dataset is synthetically generated for demonstration purposes; real-world data would improve model robustness
2. **Limited Geographic Scope**: No specific city/region data; proximity-based features (distance to city center, schools, hospitals) are absent
3. **Sample Size**: 1,000 records may not capture the full complexity of real estate markets
4. **Static Models**: Models do not account for temporal price trends or seasonality
5. **Binary Location Classification**: Real markets exist on a spectrum (suburban, semi-urban); binary classification is a simplification
6. **Feature Completeness**: Missing potentially important factors like crime rates, school quality, property condition, and legal status

## Future Improvements

To enhance this system for production deployment, the following improvements are recommended:

### Data Enhancements
- **Real-World Data Integration**: Partner with real estate platforms for actual transaction data
- **Geospatial Features**: Add latitude/longitude, distance to amenities (schools, hospitals, markets)
- **Temporal Data**: Include historical price trends and seasonal variations
- **Neighborhood Features**: Crime rates, school ratings, public transit access

### Model Enhancements
- **Multi-Class Location Types**: Expand beyond Urban/Rural to include Suburban, Semi-Urban, and Tier-based classifications
- **Time-Series Modeling**: Predict future price trends using ARIMA or LSTM
- **Ensemble Stacking**: Combine multiple model predictions for improved accuracy
- **Hyperparameter Tuning**: Grid search optimization for each model

### Application Enhancements
- **Real Estate API Integration**: Pull live property listings
- **User Authentication**: Save searches and prediction history
- **Comparative Analysis**: Compare multiple properties side-by-side
- **Price Range Predictions**: Provide confidence intervals instead of point estimates
- **Mobile Application**: React Native or Flutter mobile app

### Deployment
- **Cloud Hosting**: Deploy on AWS, Azure, or Heroku for public access
- **CI/CD Pipeline**: Automated testing and deployment
- **Model Monitoring**: Track prediction accuracy degradation over time
- **A/B Testing**: Experiment with different model architectures

## Technologies Used

- **Python 3.9+**: Core programming language
- **Flask**: Web framework for serving predictions
- **Scikit-learn**: Machine learning models and preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Joblib**: Model serialization
- **HTML/CSS**: Frontend user interface

## License

This project is open-source and available for educational and commercial use.

## Contact

For questions, suggestions, or collaboration opportunities, please feel free to reach out or open an issue on GitHub.

---

**Built with ❤️ for the Indian Real Estate Market**
