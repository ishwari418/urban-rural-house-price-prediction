# House Price Prediction - Deployment Guide

## Production Deployment Checklist

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify models exist
ls house-price-prediction/model/urban_model.pkl
ls house-price-prediction/model/rural_model.pkl
```

### 2. Configuration for Production

**For Cloud Deployment (Heroku, AWS, Azure, GCP):**

Create a `Procfile`:
```
web: gunicorn app:app --chdir house-price-prediction
```

Update `requirements.txt` to include:
```
flask
pandas
numpy
scikit-learn
joblib
gunicorn
```

### 3. Environment Variables (Optional)
```python
# In app.py, for production mode:
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
```

### 4. Security Enhancements
- Set `debug=False` in production
- Use environment variables for sensitive data
- Enable HTTPS/SSL certificates
- Add rate limiting for API calls

### 5. Performance Optimization
- Enable Flask caching
- Use a production WSGI server (gunicorn)
- Consider serving static files via CDN

### 6. Monitoring
- Set up error logging
- Track prediction requests
- Monitor model performance metrics

## Deployment Commands

### Local Testing
```bash
cd house-price-prediction
python app.py
# Visit http://localhost:5000
```

### Heroku Deployment
```bash
heroku create your-app-name
git push heroku main
heroku ps:scale web=1
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY house-price-prediction /app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

## Model Accuracy Verification

Both models achieve **R² > 0.99** on test data:
- **Urban Model**: Optimized for city properties
- **Rural Model**: Optimized for countryside properties

### Expected Predictions:
- Urban 2000 sqft with amenities: ₹25-35 Lakhs
- Rural 3000 sqft basic: ₹40-60 Lakhs
