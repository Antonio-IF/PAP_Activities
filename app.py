from flask import Flask, request, jsonify
import pandas as pd
import pickle
from utils import make_prediction, handle_new_data, load_model

app = Flask(__name__)

# Load the initial model
model_path = 'models/best_models/Random_Classifier/random_forest_model.pkl'
current_model = load_model(model_path)

# Initialize variables for tracking new data
new_data_count = 0
collected_data = pd.DataFrame(columns=[f'X{i}' for i in range(1, 24)])

@app.route('/predict', methods=['POST'])
def predict():
    global current_model, new_data_count, collected_data
    
    # Get data from the request
    input_data = request.json
    data = pd.DataFrame(input_data, index=[0])
    
    # Make prediction using the current model
    prediction = make_prediction(current_model, data)
    
    # Add the new data to the collected data
    collected_data = pd.concat([collected_data, data], ignore_index=True)
    new_data_count += 1
    
    # Check if we have collected 1000 new data points
    if new_data_count >= 1000:
        current_model = handle_new_data(collected_data, current_model)
        collected_data = pd.DataFrame(columns=[f'X{i}' for i in range(1, 24)])  # Reset collected data
        new_data_count = 0
    
    return jsonify({'Y': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
