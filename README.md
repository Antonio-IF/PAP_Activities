# Credit Prediction API and MLOps Project

## Overview

This project implements a Machine Learning model to predict credit risk, integrated into a Flask-based API. The project is structured to follow MLOps principles, including model training, deployment, monitoring, and dynamic retraining when data distribution shifts. This repository also includes code to perform statistical tests (Chi-Squared and Kolmogorov-Smirnov) to detect significant changes in data distributions.

## Features

- **Training scripts** for Random Forest, XGBoost, and Logistic Regression models.
- **Flask API** with a `/predict` route for serving model predictions.
- **Health Check Endpoint** to monitor the status of the API.
- **Statistical Tests** (Chi-Squared and Kolmogorov-Smirnov) to detect data distribution changes.
- **Dynamic Model Retraining** based on new data.
- **Automated prediction logging** to update datasets with new predictions.

## Technologies Used

- **Python**
- **Flask**
- **scikit-learn**
- **XGBoost**
- **pandas**
- **matplotlib**
- **Kolmogorov-Smirnov Test**
- **Chi-Squared Test**

## Setup Instructions

### 1. Clone the repository:

```
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

### 2. Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the required dependencies:

```
pip install -r requirements.txt
```

### 4. Set up the dataset

Ensure you have the dataset credit_train.csv in the data folder. This is used for model training and prediction.

### 5. Run the Flask API:

```
python app.py
The API will be available at http://127.0.0.1:5000.
```

### 6. Send a prediction request

You can send a POST request to http://127.0.0.1:5000/predict using a tool like Postman, or use the provided user.py script to send the request.

### 7. Health Check

You can access the health check endpoint at http://127.0.0.1:5000/ to verify if the API is running.

#### API Endpoints

/: Health check endpoint. Returns the status of the API.
/predict: POST endpoint. Takes in JSON-formatted feature data and returns a prediction.

##### Example JSON Payload for /predict:

```
{
    "X1": 50000, "X2": 1, "X3": 2, "X4": 3, "X5": 45, 
    "X6": 0, "X7": 0, "X8": 0, "X9": 0, "X10": 0, 
    "X11": 0, "X12": 0, "X13": 0, "X14": 0, "X15": 0, 
    "X16": 0, "X17": 0, "X18": 0, "X19": 0, "X20": 0, 
    "X21": 0, "X22": 0, "X23": 0
}
```

##### Retraining the Model:

The model is automatically retrained when 1000 new data points are collected. It uses statistical tests to detect significant shifts in the data distribution and triggers a retraining process if a shift is detected.

##### Testing:

You can test the API using the provided user.py script, which sends a prediction request to the /predict endpoint.

##### Statistical Tests:

The project applies Chi-Squared and Kolmogorov-Smirnov tests to determine whether there is a significant shift in the data distribution. If the tests fail, a message is printed, and the model continues using the existing data until a shift is detected.

## License

This project is licensed under the MIT License.