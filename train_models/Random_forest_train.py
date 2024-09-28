import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load the dataset
    data = pd.read_csv('data/credit_train.csv')
    X = data.drop('Y', axis=1)
    Y = data['Y']

    # Split the dataset into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size=0.30,
                                                        random_state=0)
    
    # Defining the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [200, 500, 1000],  # Number of trees in the forest
        'max_features': ['sqrt', 'log2', None],  # Number of features to consider at each split
        'max_depth': [10, 20, 50, None],  # Maximum depth of the trees
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],    # Minimum samples required at each leaf node
        'bootstrap': [True, False]        # Method of selecting samples for training each tree
    }

    # Initialize the Random Forest model
    model = RandomForestClassifier(random_state=0)

    # Setting up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,  # Number of different combinations to try
        scoring='roc_auc',
        n_jobs=-1,
        cv=7,  # 7-fold cross-validation
        verbose=2,
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
    plt.title('ROC Curve (Test Set) - Random Forest Model')
    plt.legend(loc='lower right')
    plt.show()

    # Saving the optimized Random Forest model
    with open('models/random_forest_model.pkl',  'wb') as file:
        pickle.dump(best_model, file)

if __name__ == '__main__':
    main()
