import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve

# --------------------------------------------------

# LOAD THE INITIAL DATASET FROM CSV FILE
data = pd.read_csv('data/credit_train.csv')

# SEPARATE FEATURES AND TARGET VARIABLE
X_original = data.drop('Y', axis=1)
Y_original = data['Y']

# ------------------------------

def train_initial_model(X, Y):
    """
    TRAINS THE INITIAL XGBOOST MODEL WITH THE ORIGINAL DATA.
    
    Args:
    X (DataFrame): Feature data.
    Y (Series): Target variable.
    
    Returns:
    tuple: Best model, training features, testing features, training targets, testing targets.
    """
    
    # SPLIT THE DATASET INTO TRAINING AND TEST SETS
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
    
    # DEFINE THE PARAMETER GRID FOR XGBOOST
    param_grid = {
        'n_estimators': [300, 600, 900],
        'max_depth': [20, 40, 90],
        'learning_rate': [0.01, 0.1, 0.7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0.5, 1, 2]
    }

    # INITIALIZE THE XGBOOST CLASSIFIER MODEL
    model = XGBClassifier(random_state=0, eval_metric='logloss')

    # SET UP RANDOMIZEDSEARCHCV FOR HYPERPARAMETER TUNING
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        scoring='roc_auc',
        n_jobs=-1,
        cv=7,
        verbose=2,
        random_state=0
    )

    # FIT THE MODEL USING RANDOMIZEDSEARCHCV TO FIND THE BEST PARAMETERS
    random_search.fit(X_train, Y_train)
    best_model = random_search.best_estimator_

    # ----------
    
    # SAVE THE OPTIMIZED XGBOOST MODEL TO A PICKLE FILE
    with open('models/xgboost_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    return best_model, X_train, X_test, Y_train, Y_test

# ------------------------------

def perform_statistical_tests(X_old, X_new):
    """
    PERFORMS STATISTICAL TESTS TO DETECT SIGNIFICANT CHANGES BETWEEN OLD AND NEW DATA.
    
    Args:
    X_old (DataFrame): Original feature data.
    X_new (DataFrame): New feature data to compare.
    
    Returns:
    bool: True IF A SIGNIFICANT CHANGE IS DETECTED, OTHERWISE FALSE.
    """
    
    significant_change = False
    
    # ITERATE THROUGH EACH COLUMN IN THE OLD DATASET
    for column in X_old.columns:

        # CHECK THE DATA TYPE OF THE COLUMN
        if X_old[column].dtype in ['int64', 'float64']:
            # APPLY KOLMOGOROV-SMIRNOV TEST FOR NUMERICAL VARIABLES
            stat, p_value = ks_2samp(X_old[column], X_new[column])

        else:
            # CREATE A CONTINGENCY TABLE FOR CATEGORICAL VARIABLES AND APPLY CHI-SQUARED TEST
            contingency_table = pd.crosstab(X_old[column], X_new[column])
            stat, p_value, _, _ = chi2_contingency(contingency_table)

        # IF P_VALUE < 0.05, SIGNIFICANT DIFFERENCE DETECTED
        if p_value < 0.05:
            print(f"Significant difference found in column '{column}' with p-value {p_value:.4f}")
            significant_change = True

    return significant_change

# ------------------------------

def retrain_model_if_needed(model, X_old, Y_old, X_new, Y_new):
    """
    RETRAINS THE MODEL IF SIGNIFICANT CHANGES ARE DETECTED IN THE DATA DISTRIBUTION.
    
    Args:
    model (XGBClassifier): CURRENT MODEL.
    X_old (DataFrame): ORIGINAL FEATURE DATA.
    Y_old (Series): ORIGINAL TARGET DATA.
    X_new (DataFrame): NEW FEATURE DATA.
    Y_new (Series): NEW TARGET DATA.
    
    Returns:
    tuple: Updated model, combined feature data, combined target data.
    """
    
    # COMBINE OLD AND NEW FEATURE DATA
    X_combined = pd.concat([X_old, X_new], axis=0)
    # COMBINE OLD AND NEW TARGET DATA
    Y_combined = pd.concat([Y_old, Y_new], axis=0)

    # IF SIGNIFICANT CHANGES DETECTED, RETRAIN THE MODEL
    if perform_statistical_tests(X_old, X_new):
        print("Retraining the model due to significant changes in data distribution...")
        model = train_initial_model(X_combined, Y_combined)[0]

    # ELSE, USE THE EXISTING MODEL
    else:
        print("No significant changes detected. Using the existing model.")

    return model, X_combined, Y_combined

# ------------------------------

def handle_new_data(new_data, current_model, X_current, Y_current):
    """
    HANDLES NEW DATA BY CHECKING FOR DATA DISTRIBUTION CHANGES AND RETRAINING THE MODEL IF NECESSARY.
    
    Args:
    new_data (DataFrame): NEW DATA RECEIVED FOR PREDICTION.
    current_model (XGBClassifier): CURRENTLY TRAINED MODEL.
    X_current (DataFrame): CURRENT FEATURE DATA.
    Y_current (Series): CURRENT TARGET DATA.
    
    Returns:
    tuple: Predictions, updated model, combined feature data, combined target data.
    """
    
    # EXTRACT FEATURES AND TARGET FROM NEW DATA
    X_new = new_data.drop('Y', axis=1)
    Y_new = new_data['Y']

    # UPDATE THE MODEL BASED ON THE DISTRIBUTION CHECK
    current_model, X_combined, Y_combined = retrain_model_if_needed(current_model, X_current, Y_current, X_new, Y_new)
    
    # MAKE PREDICTIONS ON NEW DATA
    predictions = current_model.predict(X_new)

    return predictions, current_model, X_combined, Y_combined

# ------------------------------

def main():
    """
    MAIN FUNCTION TO SIMULATE RECEIVING NEW DATA AND MANAGING MODEL RETRAINING.
    """
    
    # TRAIN THE INITIAL MODEL WITH ORIGINAL DATA
    model, X_train, X_test, Y_train, Y_test = train_initial_model(X_original, Y_original)

    # SIMULATE RECEIVING NEW DATA POINTS
    new_data_counter = 0
    collected_data = pd.DataFrame(columns=X_original.columns.to_list() + ['Y'])

    while True:

        # SIMULATE RECEIVING A NEW DATA POINT (IN PRACTICE, REPLACE THIS WITH ACTUAL DATA COLLECTION)
        new_data_point = np.random.choice(len(X_test))
        new_sample = X_test.iloc[new_data_point].to_frame().T
        new_sample['Y'] = Y_test.iloc[new_data_point]

        # ADD THE NEW DATA POINT TO THE COLLECTED DATAFRAME
        collected_data = pd.concat([collected_data, new_sample], ignore_index=True)
        new_data_counter += 1

        # IF 1000 NEW DATA POINTS HAVE BEEN COLLECTED, CHECK FOR DISTRIBUTION CHANGES
        if new_data_counter >= 1000:

            # PRINT STATUS MESSAGE
            print("Collected 1000 new data points. Checking for distribution change...")
            predictions, model, X_train, Y_train = handle_new_data(collected_data, model, X_train, Y_train)

            # RESET THE COUNTER AND COLLECTED DATA
            new_data_counter = 0
            collected_data = pd.DataFrame(columns=X_original.columns.to_list() + ['Y'])

        # MAKE PREDICTIONS AS NEW DATA ARRIVES (SIMULATION PURPOSES)
        if new_data_counter % 100 == 0:
            print(f"Predictions at data point {new_data_counter}:", model.predict(X_test[:5]))

# ------------------------------

# THIS BLOCK OF CODE EXECUTES THE MAIN FUNCTION WHEN THE SCRIPT IS RUN DIRECTLY
if __name__ == '__main__':

    main()