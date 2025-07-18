# Building and Deploying ML

## Overview

This guide covers the practical aspects of building, deploying, and maintaining machine learning systems in production, including MLOps, model serving, data pipelines, monitoring, and scalability.

## 1. MLOps Fundamentals
- **Version Control**: Use git for code, DVC for data/models
- **CI/CD**: Automated testing and deployment
- **Monitoring**: Track model/data drift, performance

## 2. Model Serving
- **REST APIs**: Serve models via HTTP endpoints (Flask, FastAPI)
- **Microservices**: Containerize with Docker, orchestrate with Kubernetes

#### Example: Simple Flask API for Model Serving
```python
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})
if __name__ == '__main__':
    app.run()
```

## 3. Data Pipelines
- **ETL**: Extract, Transform, Load
- **Feature Stores**: Centralized feature management
- **Data Validation**: Detect anomalies, missing values

## 4. Model Monitoring
- **Drift Detection**: Monitor input/output distributions
- **Performance Tracking**: Log metrics, alert on degradation

## 5. Scalability
- **Distributed Training**: Use multiple GPUs/TPUs
- **Model Serving at Scale**: Load balancing, autoscaling

## Best Practices
- Automate as much as possible
- Monitor models continuously
- Use containers for reproducibility
- Document everything

## Applications
- Real-time prediction APIs
- Batch inference pipelines
- Scalable ML in the cloud

## Summary
- MLOps enables reliable, scalable ML deployment
- Model serving and monitoring are critical for production
- Data pipelines and validation ensure data quality
- Scalability is key for real-world impact 