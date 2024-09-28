import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
from scipy.stats import chi2_contingency, ks_2samp
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import numpy as np

# Load the initial dataset
data = pd.read_csv('data/credit_train.csv')
X_original = data.drop('Y', axis=1)
Y_original = data['Y']

# Train the model with the original data
def train_initial_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
    
    param_grid = {
        'n_estimators': [300, 600, 900],
        'max_depth': [20, 40, 90],
        'learning_rate': [0.01, 0.1, 0.7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0.5, 1, 2]
    }

    model = XGBClassifier(random_state=0, eval_metric='logloss')

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

    random_search.fit(X_train, Y_train)
    best_model = random_search.best_estimator_

    # Save the model
    with open('models/xgboost_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    return best_model, X_train, X_test, Y_train, Y_test

# Perform chi-square and KS tests
def perform_statistical_tests(X_old, X_new):
    significant_change = False
    
    for column in X_old.columns:
        if X_old[column].dtype in ['int64', 'float64']:  # Continuous variable
            stat, p_value = ks_2samp(X_old[column], X_new[column])
        else:  # Categorical variable
            contingency_table = pd.crosstab(X_old[column], X_new[column])
            stat, p_value, _, _ = chi2_contingency(contingency_table)

        # If p_value < 0.05, significant difference detected
        if p_value < 0.05:
            print(f"Significant difference found in column '{column}' with p-value {p_value:.4f}")
            significant_change = True

    return significant_change

# Retrain the model if needed
def retrain_model_if_needed(model, X_old, Y_old, X_new, Y_new):
    X_combined = pd.concat([X_old, X_new], axis=0)
    Y_combined = pd.concat([Y_old, Y_new], axis=0)

    if perform_statistical_tests(X_old, X_new):
        print("Retraining the model due to significant changes in data distribution...")
        model = train_initial_model(X_combined, Y_combined)[0]
    else:
        print("No significant changes detected. Using the existing model.")

    return model, X_combined, Y_combined

# Function to handle new incoming data
def handle_new_data(new_data, current_model, X_current, Y_current):
    # Extract features and target from new data
    X_new = new_data.drop('Y', axis=1)
    Y_new = new_data['Y']

    # Update the model based on the distribution check
    current_model, X_combined, Y_combined = retrain_model_if_needed(current_model, X_current, Y_current, X_new, Y_new)
    
    # Make predictions on new data
    predictions = current_model.predict(X_new)
    return predictions, current_model, X_combined, Y_combined

# Main function to initialize the training and handle new data
def main():
    # Train the initial model
    model, X_train, X_test, Y_train, Y_test = train_initial_model(X_original, Y_original)

    # Suppose new data comes in (you need to collect 1000 new samples before checking)
    new_data_counter = 0
    collected_data = pd.DataFrame(columns=X_original.columns.to_list() + ['Y'])

    while True:
        # Simulate receiving new data point-by-point (in practice, you'd replace this with actual new data)
        new_data_point = np.random.choice(len(X_test))  # Simulate selection of a new data point
        new_sample = X_test.iloc[new_data_point].to_frame().T
        new_sample['Y'] = Y_test.iloc[new_data_point]
        
        collected_data = pd.concat([collected_data, new_sample], ignore_index=True)
        new_data_counter += 1

        if new_data_counter >= 1000:
            print("Collected 1000 new data points. Checking for distribution change...")
            predictions, model, X_train, Y_train = handle_new_data(collected_data, model, X_train, Y_train)

            # Reset the counter and collected data
            new_data_counter = 0
            collected_data = pd.DataFrame(columns=X_original.columns.to_list() + ['Y'])

        # Use the model to make predictions as new data arrives (example for simulation purposes)
        if new_data_counter % 100 == 0:  # Display predictions every 100 data points
            print(f"Predictions at data point {new_data_counter}:", model.predict(X_test[:5]))

if __name__ == '__main__':
    main()
