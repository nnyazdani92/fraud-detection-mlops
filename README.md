# End-to-End Credit Card Fraud Detection Machine Learning Pipeline

[![MLflow](https://img.shields.io/badge/mlflow-%2331A8FF.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

An enterprise-grade machine learning pipeline for credit card fraud detection, demonstrating professional ML engineering practices and production-ready patterns.

## 🚀 Key Features

- **MLflow Integration**: Full experiment tracking, model registry, and artifact logging
- **Modular Pipeline Design**: Clean separation of EDA, preprocessing, feature engineering, and modeling
- **Reproducible Workflows**: Versioned data, models, and configurations
- **Production-Ready Patterns**:
  - Hyper parameter tuning (Optuna)
  - Advanced custom automated feature extraction (PyTorch Autoencoders)
  - Deep Learning integration (PyTorch Autoencoders)
  - Class imbalance handling (SMOTE)

## 🛠 Technical Stack

**Core ML**  
`Python` `PyTorch` `imbalanced-learn` `XGboost` `Optuna`

**MLOps**  
`MLflow` `scikit-learn` `Pandas` `NumPy`

**Infrastructure**  
`pipenv` `Docker` `Jupyter Lab`


## 🔍 Results Highlights

### Model Performance

```
              precision    recall  f1-score   support

   Not Fraud   0.999733  0.999522  0.999627     71079
       Fraud   0.753623  0.845528  0.796935       123

    accuracy                       0.999256     71202
   macro avg   0.876678  0.922525  0.898281     71202
weighted avg   0.999307  0.999256  0.999277     71202

```

### MLflow Tracking

    Experiments: 120+

    Registered models: 15 versions

    Logged artifacts: Features, metrics, signatures

## 💻 Professional Practices

- ✅ Testing & Validation
- ✅ CI/CD Readiness
- ✅ Documentation
- ✅ Error Handling
- ✅ Scalability

## 🚀 Getting Started
```
export MLFLOW_TRACKING_URI=<TRACKING_URI>
mlflow run --env-manager local .
```

**Built with engineering rigour**

[Connect on LinkedIn](https://www.linkedin.com/in/nnyazdani92)
