import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve

# --------------------------------------------------

def main():
    """
    Main function to train and optimize a random forest classifier.
    """
    
    # LOAD THE DATASET FROM CSV FILE
    data = pd.read_csv('data/credit_train.csv')

    # SEPARATE FEATURES AND TARGET VARIABLE
    X = data.drop('Y', axis=1)
    Y = data['Y']

    # SPLIT THE DATASET INTO TRAINING AND TEST SETS
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
    
    # DEFINE THE PARAMETER GRID FOR RANDOM FOREST
    param_grid = {
        'n_estimators': [200, 500, 1000],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # INITIALIZE THE RANDOM FOREST CLASSIFIER MODEL
    model = RandomForestClassifier(random_state=0)

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

    # USE THE BEST MODEL FOUND BY RANDOMIZEDSEARCHCV
    best_model = random_search.best_estimator_

    # MAKE PREDICTIONS ON TRAINING AND TEST SETS
    Y_hat_train = best_model.predict(X_train)
    Y_hat_test = best_model.predict(X_test)

    # CALCULATE F1 SCORE AND ACCURACY FOR TRAINING AND TEST SETS
    f1_train = f1_score(Y_train, Y_hat_train)
    f1_test = f1_score(Y_test, Y_hat_test)
    accuracy_train = accuracy_score(Y_train, Y_hat_train)
    accuracy_test = accuracy_score(Y_test, Y_hat_test)

    # PRINT THE PERFORMANCE METRICS
    print('F1 score train:', f1_train)
    print('F1 score test:', f1_test)
    print('Accuracy train:', accuracy_train)
    print('Accuracy test:', accuracy_test)

    # CALCULATE AUC-ROC FOR TRAINING AND TEST SETS
    Y_prob_train = best_model.predict_proba(X_train)[:, 1]
    Y_prob_test = best_model.predict_proba(X_test)[:, 1]
    
    auc_train = roc_auc_score(Y_train, Y_prob_train)
    auc_test = roc_auc_score(Y_test, Y_prob_test)

    # PRINT THE AUC-ROC SCORES
    print('AUC-ROC train:', auc_train)
    print('AUC-ROC test:', auc_test)

    # PLOT THE ROC CURVE FOR THE TEST SET
    fpr, tpr, _ = roc_curve(Y_test, Y_prob_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC-ROC Test = {auc_test:.2f}', color='blue')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set) - Random Forest Model')
    plt.legend(loc='lower right')
    plt.show()

    # ----------
    
    # SAVE THE OPTIMIZED RANDOM FOREST MODEL TO A PICKLE FILE
    with open('models/random_forest_model.pkl',  'wb') as file:
        pickle.dump(best_model, file)

# ------------------------------

# THIS BLOCK OF CODE EXECUTES THE MAIN FUNCTION WHEN THE SCRIPT IS RUN DIRECTLY
if __name__ == '__main__':

    main()