import pandas as pd
from flask import Flask, request, jsonify
from utils import make_prediction, handle_new_data, load_model

# --------------------------------------------------

# INITIALIZE FLASK APPLICATION
app = Flask(__name__)

# LOAD THE INITIAL MODEL FROM FILE USING PICKLE
model_path = 'best_models/Random_Classifier/random_forest_model.pkl'
current_model = load_model(model_path)

# INITIALIZE VARIABLES FOR TRACKING NEW DATA
new_data_count = 0
collected_data = pd.DataFrame(columns=[f'X{i}' for i in range(1, 24)])

# ------------------------------

# HOME ROUTE (DEFAULT ENDPOINT)
@app.route('/', methods=['GET'])
def home():

    return "Welcome to the Prediction API! Use the /predict endpoint to get predictions."

# ------------------------------

# PREDICT ROUTE TO HANDLE POST REQUESTS
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles incoming POST requests for predictions. Receives input data in JSON format, 
    makes a prediction using the loaded machine learning model, and returns the result.
    Additionally, it collects new data and checks if the model needs to be retrained 
    after a certain threshold of new data points is reached.
    
    Global Variables:
    - current_model: The model currently in use for predictions.
    - new_data_count: Counter to track the number of new data points received.
    - collected_data: DataFrame to accumulate new incoming data.
    
    Returns:
    JSON: The predicted class or value (Y).
    """

    global current_model, new_data_count, collected_data

    # EXTRACT INPUT DATA FROM THE REQUEST AND CONVERT IT TO A DATAFRAME
    input_data = request.json
    data = pd.DataFrame(input_data, index=[0])

    # MAKE A PREDICTION USING THE CURRENT MODEL
    prediction = make_prediction(current_model, data)

    # ADD THE NEW DATA TO THE COLLECTED DATAFRAME
    collected_data = pd.concat([collected_data, data], ignore_index=True)
    new_data_count += 1

    # IF 1000 NEW DATA POINTS HAVE BEEN COLLECTED, RETRAIN THE MODEL
    if new_data_count >= 1000:
        current_model = handle_new_data(collected_data, current_model)
        collected_data = pd.DataFrame(columns=[f'X{i}' for i in range(1, 24)])
        new_data_count = 0

    # RETURN THE PREDICTION AS A JSON RESPONSE
    return jsonify({'Y': int(prediction[0])})

# --------------------------------------------------

# THIS BLOCK OF CODE STARTS THE FLASK APPLICATION WHEN THE SCRIPT IS RUN DIRECTLY
if __name__ == '__main__':

    app.run(debug=True)