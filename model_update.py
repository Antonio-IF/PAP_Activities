import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from scipy.stats import chi2_contingency, ks_2samp

# Function to perform statistical tests
def perform_statistical_tests(X_original, X_new):
    significant_change = False
    
    for column in X_original.columns:
        if X_original[column].dtype in ['int64', 'float64']:
            stat, p_value = ks_2samp(X_original[column], X_new[column])
        else:
            contingency_table = pd.crosstab(X_original[column], X_new[column])
            stat, p_value, _, _ = chi2_contingency(contingency_table)
        
        if p_value < 0.05:
            print(f"Significant change detected in column '{column}' (p-value: {p_value})")
            significant_change = True
            
    return significant_change

# Function to retrain the model
def retrain_model(X_combined, Y_combined):
    model = RandomForestClassifier(random_state=0, n_estimators=500, max_depth=20)
    model.fit(X_combined, Y_combined)
    
    # Save the updated model
    with open('models/best_models/Random_Classifier/random_forest_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model
