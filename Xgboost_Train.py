import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

def main():
    # Load the dataset
    data = pd.read_csv('data/credit_train.csv')
    X = data.drop('Y', axis=1)
    Y = data['Y']

    # Split the dataset into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size=0.255555,
                                                        random_state=0)
    
    # Defining the parameter grid for XGBoost
    param_grid = {
        'n_estimators': [300, 600, 900],
        'max_depth': [15, 40, 75],
        'learning_rate': [0.01, 0.1, 0.7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],  # L1 regularization
        'reg_lambda': [0.5, 1, 2]  # L2 regularization
    }

    # Initialize the XGBoost model
    model = XGBClassifier(random_state=0, eval_metric='logloss')

    # Setting up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Number of different combinations to try
        scoring='roc_auc',
        n_jobs=-1,
        cv=3,  # 3-fold cross-validation
        verbose=1,
        random_state=0
    )

    # Fit the model using RandomizedSearchCV
    random_search.fit(X_train, Y_train)

    # Use the best model found by RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Make predictions
    Y_hat_train = best_model.predict(X_train)
    Y_hat_test = best_model.predict(X_test)

    # Calculate F1 score and accuracy
    f1_train = f1_score(Y_train, Y_hat_train)
    f1_test = f1_score(Y_test, Y_hat_test)
    accuracy_train = accuracy_score(Y_train, Y_hat_train)
    accuracy_test = accuracy_score(Y_test, Y_hat_test)

    print('F1 score train:', f1_train)
    print('F1 score test:', f1_test)
    print('Accuracy train:', accuracy_train)
    print('Accuracy test:', accuracy_test)

    # Calculate AUC-ROC
    Y_prob_train = best_model.predict_proba(X_train)[:, 1]
    Y_prob_test = best_model.predict_proba(X_test)[:, 1]
    
    auc_train = roc_auc_score(Y_train, Y_prob_train)
    auc_test = roc_auc_score(Y_test, Y_prob_test)

    print('AUC-ROC train:', auc_train)
    print('AUC-ROC test:', auc_test)

    # Plotting ROC Curve for test set
    fpr, tpr, _ = roc_curve(Y_test, Y_prob_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC-ROC Test = {auc_test:.2f}', color='blue')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set) - XGBoost Model')
    plt.legend(loc='lower right')
    plt.show()

    # Saving the optimized XGBoost model
    with open('models/xgboost_model.pkl',  'wb') as file:
        pickle.dump(best_model, file)

if __name__ == '__main__':
    main()
