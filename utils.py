import pickle
import pandas as pd
from model_update import perform_statistical_tests, retrain_model

# --------------------------------------------------

def load_model(model_path):
    """
    Loads a pre-trained machine learning model from the given file path using pickle.
    
    Args:
    model_path (str): The path to the serialized model.
    
    Returns:
    model: The loaded machine learning model.
    """

    # OPEN THE MODEL FILE IN READ-BINARY MODE
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model

# ------------------------------

def make_prediction(model, data):
    """
    Uses the loaded model to make a prediction based on the provided input data.
    
    Args:
    model: The loaded machine learning model.
    data: The input data (usually a DataFrame) on which to make predictions.
    
    Returns:
    array: The predicted class or value from the model.
    """

    print("Data received for prediction:")
    print(data.head())

    try:
        prediction = model.predict(data)
        print("Prediction made successfully:", prediction)

    except Exception as e:
        print(f"Error during prediction: {e}")
        prediction = None

    return prediction

# ------------------------------

def handle_new_data(new_data, current_model):
    """
    Handles incoming new data and checks whether the model needs to be retrained
    based on statistical tests. If retraining is required, it combines the original
    training data with the new data and retrains the model.
    
    Args:
    new_data (DataFrame): The new incoming data that needs to be evaluated.
    current_model: The current machine learning model in use.
    
    Returns:
    model: The updated model (either retrained or the same one if no change is needed).
    """

    # LOAD ORIGINAL TRAINING DATA
    X_original = pd.read_csv('data/credit_train.csv').drop('Y', axis=1)
    Y_original = pd.read_csv('data/credit_train.csv')['Y']
    
    # PERFORM STATISTICAL TESTS TO DETECT SIGNIFICANT CHANGES IN THE NEW DATA
    if perform_statistical_tests(X_original, new_data):
        print("Retraining model...")

        # COMBINE THE ORIGINAL AND NEW DATA FOR RETRAINING
        X_combined = pd.concat([X_original, new_data.drop('Y', axis=1)], axis=0)
        Y_combined = pd.concat([Y_original, new_data['Y']], axis=0)

        # RETRAIN THE MODEL WITH THE UPDATED DATA
        current_model = retrain_model(X_combined, Y_combined)

    else:
        print("No significant change detected; using the existing model.")
    
    return current_model