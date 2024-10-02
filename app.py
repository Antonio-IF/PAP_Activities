import os
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils import make_prediction, handle_new_data, load_model

# --------------------------------------------------

# INITIALIZE FLASK APPLICATION
app = Flask(__name__)

# DEFINE UPLOAD FOLDER AND ALLOWED EXTENSIONS FOR MULTIPLE FILE TYPES
upload_folder = 'uploads/'
allowed_extensions = {'csv', 'xlsx', 'txt', 'json'}
app.config['UPLOAD_FOLDER'] = upload_folder

# ------------------------------

# FUNCTION TO CHECK ALLOWED FILE EXTENSIONS
def allowed_file(filename):

    # VALIDATE THAT THE FILE HAS AN ALLOWED EXTENSION
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# ------------------------------

# FUNCTION TO LOAD FILE BASED ON EXTENSION
def load_file(filepath, extension):
    """
    LOADS THE DATA FROM A FILE BASED ON ITS EXTENSION.
    
    ARGS:
    - filepath (str): PATH TO THE FILE.
    - extension (str): FILE EXTENSION (CSV, XLSX, TXT, JSON).
    
    RETURNS:
    DATAFRAME: LOADED DATA AS A PANDAS DATAFRAME.
    """

    if extension == 'csv':
        return pd.read_csv(filepath)

    elif extension == 'xlsx':
        return pd.read_excel(filepath)

    elif extension == 'txt':
        return pd.read_csv(filepath, delimiter='\t')

    elif extension == 'json':
        return pd.read_json(filepath)

    else:
        raise ValueError('Unsupported file extension')

# ------------------------------

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
    """
    HEALTH CHECK ENDPOINT TO VERIFY IF THE API IS RUNNING.
    
    RETURNS:
    JSON: A MESSAGE INDICATING THE API STATUS AND A STATUS CODE OF 200.
    """

    return jsonify({'status': 'API is running'}), 200

# ------------------------------

# PREDICT ROUTE TO HANDLE GET, POST REQUESTS, AND MULTIPLE FILE UPLOADS
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    HANDLES INCOMING GET, POST REQUESTS AND FILE UPLOADS FOR PREDICTIONS. 
    FOR POST REQUESTS, IT ALLOWS:
    - UPLOADING FILES WITH DIFFERENT EXTENSIONS (CSV, XLSX, TXT, JSON) CONTAINING INPUT DATA FOR PREDICTION, 
    - DIRECT JSON INPUT FOR PREDICTION.
    RETURNS PREDICTIONS IN BOTH CASES.
    """

    global current_model, new_data_count, collected_data

    # ----------

    # HANDLE GET REQUESTS WITH PARAMETERS FROM URL
    if request.method == 'GET':

        # EXTRACT QUERY PARAMETERS (FOR EXAMPLE: ?X1=50000&X2=1&X3=2...)
        params = request.args

        if params:

            # CONVERT PARAMETERS INTO A DATAFRAME
            input_data = {key: [float(value)] for key, value in params.items()}
            data = pd.DataFrame(input_data)

            # MAKE PREDICTION
            prediction = make_prediction(current_model, data)

            if prediction is not None:

                return jsonify({
                    'input_data': input_data,
                    'prediction': int(prediction[0])
                }), 200

            else:
                return jsonify({'error': 'Prediction failed.'}), 400

        else:

            # IF NO PARAMETERS ARE PROVIDED, SHOW AN EXAMPLE TO THE USER
            return jsonify({
                'message': 'TO MAKE PREDICTIONS, SEND A POST REQUEST WITH INPUT DATA IN JSON FORMAT, UPLOAD A FILE, OR PROVIDE QUERY PARAMETERS IN THE URL.',
                'example_usage': 'EXAMPLE OF HOW TO USE THE API WITH URL PARAMETERS: http://127.0.0.1:5000/predict?X1=50000&X2=1&X3=2&X4=3&X5=45&X6=0&X7=0&X8=0&X9=0&X10=0&X11=0&X12=0&X13=0&X14=0&X15=0&X16=0&X17=0&X18=0&X19=0&X20=0&X21=0&X22=0&X23=0'
            }), 200

    # ----------

    # HANDLE POST REQUESTS
    if request.method == 'POST':

        # CHECK IF THE POST REQUEST HAS THE FILE PART FOR UPLOAD
        if 'file' in request.files:
            file = request.files['file']

            # IF NO FILE IS SELECTED
            if file.filename == '':
                return jsonify({'error': 'NO FILE SELECTED.'}), 400

            # CHECK IF THE FILE IS ALLOWED (CSV, XLSX, TXT, JSON)
            if file and allowed_file(file.filename):

                # SECURE THE FILENAME AND SAVE IT
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # GET THE FILE EXTENSION
                file_extension = filename.rsplit('.', 1)[1].lower()

                try:
                    # LOAD THE FILE INTO A DATAFRAME BASED ON ITS EXTENSION
                    data = load_file(filepath, file_extension)

                    # MAKE PREDICTIONS USING THE CURRENT MODEL
                    predictions = make_prediction(current_model, data)

                    # RETURN THE PREDICTIONS IN JSON FORMAT
                    if predictions is not None:

                        return jsonify({
                            'message': f'Predictions made for file: {filename}',
                            'predictions': predictions.tolist()
                        }), 200

                    else:
                        return jsonify({'error': 'Prediction failed.'}), 400

                except ValueError as e:
                    return jsonify({'error': str(e)}), 400
            else:
                return jsonify({'error': 'FILE TYPE NOT ALLOWED. PLEASE UPLOAD A CSV, XLSX, TXT, OR JSON FILE.'}), 400

        # ----------

        # HANDLE JSON INPUT FOR PREDICTIONS (FOR EXAMPLE VIA USER.PY)
        else:

            # EXTRACT INPUT DATA FROM THE REQUEST (EXPECTING A LIST OF DICTIONARIES)
            input_data = request.json

            # CHECK IF THE INPUT IS A LIST (MULTIPLE ENTRIES)
            if isinstance(input_data, list):

                # CONVERT LIST OF DICTIONARIES TO DATAFRAME
                data = pd.DataFrame(input_data)

            else:

                # IF SINGLE DICTIONARY, CONVERT TO DATAFRAME DIRECTLY
                data = pd.DataFrame([input_data])

            # MAKE A PREDICTION USING THE CURRENT MODEL
            predictions = make_prediction(current_model, data)

            # ADD THE NEW DATA TO THE COLLECTED DATAFRAME
            collected_data = pd.concat([collected_data, data], ignore_index=True)
            new_data_count += len(data)

            # IF 1000 NEW DATA POINTS HAVE BEEN COLLECTED, RETRAIN THE MODEL
            if new_data_count >= 1000:
                current_model = handle_new_data(collected_data, current_model)
                collected_data = pd.DataFrame(columns=[f'X{i}' for i in range(1, 24)])
                new_data_count = 0

            # RETURN THE PREDICTION MADE FOR THE REQUESTED DATA
            if predictions is not None:

                return jsonify({'Y': predictions.tolist()}), 200

            else:
                return jsonify({'error': 'Prediction failed.'}), 400

# --------------------------------------------------

# THIS BLOCK OF CODE STARTS THE FLASK APPLICATION WHEN THE SCRIPT IS RUN DIRECTLY
if __name__ == '__main__':

    # CREATE UPLOAD FOLDER IF IT DOESN'T EXIST
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    app.run(debug=True)