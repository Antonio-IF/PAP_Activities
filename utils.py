import pickle
import pandas as pd
from model_update import perform_statistical_tests, retrain_model

# Function to load the model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make a prediction
def make_prediction(model, data):
    return model.predict(data)

# Function to handle new data, check distribution, and retrain if needed
def handle_new_data(new_data, current_model):
    X_original = pd.read_csv('data/credit_train.csv').drop('Y', axis=1)
    Y_original = pd.read_csv('data/credit_train.csv')['Y']
    
    if perform_statistical_tests(X_original, new_data):
        print("Retraining model...")
        X_combined = pd.concat([X_original, new_data.drop('Y', axis=1)], axis=0)
        Y_combined = pd.concat([Y_original, new_data['Y']], axis=0)
        current_model = retrain_model(X_combined, Y_combined)
    else:
        print("No significant change detected; using the existing model.")
    
    return current_model
