# End-to-End Credit Card Fraud Detection Machine Learning Pipeline

[![MLflow](https://img.shields.io/badge/mlflow-%2331A8FF.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

An enterprise-grade machine learning pipeline for credit card fraud detection, demonstrating professional ML engineering practices and production-ready patterns.

## 🚀 Key Features

- **MLflow Integration**: Full experiment tracking, model registry, and artifact logging
- **Modular Pipeline Design**: Clean separation of EDA, preprocessing, feature engineering, and modeling
- **Reproducible Workflows**: Versioned data, models, and configurations
- **Production-Ready Patterns**:
  - Hyper parameter tuning (Optuna)
  - Advanced custom automated feature extraction (Autoencoder)
  - Deep Learning integration (PyTorch)
  - Class imbalance handling (SMOTE)

## 🛠 Technical Stack

**Core ML**  
`Python` `PyTorch` `imbalanced-learn` `XGboost` `Optuna`

**MLOps**  
`MLflow` `scikit-learn` `Pandas` `NumPy`

**Infrastructure**  
`pipenv` `Docker` `Jupyter Lab`


## 🔍 Results Highlights

### Model Performance (unseen test set)

```
               precision    recall    f1-score   support

   Not Fraud   0.9997       0.9995    0.9996       71079
       Fraud   0.7536       0.8455    0.7969         123

    accuracy                          0.9992       71202
   macro avg   0.8766       0.9225    0.8982       71202
weighted avg   0.9993       0.9992    0.9992       71202

```

### MLflow Tracking

    Experiments: 500+

    Registered models: 50 versions

    Logged artifacts: features, metrics, signatures

## 💻 Professional Practices

- ✅ Testing & Validation
- ✅ CI/CD Readiness
- ✅ Documentation
- ✅ Error Handling
- ✅ Scalability

## 🚀 Getting Started
```
export KAGGLE_USERNAME=<YOUR_KAGGLE_USERNAME>
export KAGGLE_KEY=<YOUR_KAGGLE_API_KEY>

docker build -t <TAG_NAME> .
docker run --gpus all -it -e KAGGLE_USERNAME=<YOUR_KAGGLE_USERNAME> -e KAGGLE_KEY=<YOUR_KAGGLE_API_KEY> <TAG_NAME>
```

This will run the mlflow pipeline from data acquisition to model training and (eventual) deployment.


**Built with engineering rigour**

[Connect on LinkedIn](https://www.linkedin.com/in/nnyazdani92)
