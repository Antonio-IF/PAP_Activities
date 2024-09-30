import pickle
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------

def perform_statistical_tests(X_original, X_new):
    """
    Performs statistical tests to check for significant differences between
    the original training data and new incoming data. It uses the Kolmogorov-Smirnov
    test for numerical variables and Chi-squared test for categorical variables.
    
    Args:
    X_original (DataFrame): The original training data.
    X_new (DataFrame): The new data to compare against the original.
    
    Returns:
    bool: True if a significant change is detected, otherwise False.
    """

    significant_change = False
    
    # ITERATE THROUGH EACH COLUMN IN THE ORIGINAL DATASET
    for column in X_original.columns:
        if X_original[column].dtype in ['int64', 'float64']:

            # APPLY KOLMOGOROV-SMIRNOV TEST FOR NUMERICAL VARIABLES
            stat, p_value = ks_2samp(X_original[column], X_new[column])

        else:
            # CREATE A CONTINGENCY TABLE FOR CATEGORICAL VARIABLES AND APPLY CHI-SQUARED TEST
            contingency_table = pd.crosstab(X_original[column], X_new[column])
            stat, p_value, _, _ = chi2_contingency(contingency_table)
        
        # CHECK IF THERE IS A STATISTICALLY SIGNIFICANT CHANGE
        if p_value < 0.05:
            print(f"Significant change detected in column '{column}' (p-value: {p_value})")
            significant_change = True
            
    return significant_change

# ------------------------------

def retrain_model(X_combined, Y_combined):
    """
    Retrains a RandomForestClassifier model with the combined original and new data.
    Saves the retrained model as a pickle file for future use.
    
    Args:
    X_combined (DataFrame): The combined original and new feature data.
    Y_combined (DataFrame): The combined original and new target data.
    
    Returns:
    model: The retrained RandomForestClassifier model.
    """

    # INITIALIZE RANDOMFORESTCLASSIFIER WITH SPECIFIC PARAMETERS
    model = RandomForestClassifier(random_state=0, n_estimators=500, max_depth=20)

    # FIT THE MODEL TO THE COMBINED DATA
    model.fit(X_combined, Y_combined)
    
    # SAVE THE RETRAINED MODEL AS A PICKLE FILE
    with open('models/best_models/Random_Classifier/random_forest_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model