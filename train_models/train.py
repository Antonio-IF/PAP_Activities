import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --------------------------------------------------

def main():
    """
    MAIN FUNCTION TO TRAIN A LOGISTIC REGRESSION MODEL.
    """
    
    # LOAD THE DATASET FROM CSV FILE
    data = pd.read_csv('data/credit_train.csv')

    # SEPARATE FEATURES AND TARGET VARIABLE
    X = data.drop('Y', axis=1)
    Y = data['Y']

    # SPLIT THE DATASET INTO TRAINING AND TEST SETS WITH SPECIFIC TEST SIZE
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.255555, random_state=0)

    # TRAIN THE LOGISTIC REGRESSION MODEL
    model = LogisticRegression().fit(X_train, Y_train)

    # MAKE PREDICTIONS ON TRAINING AND TEST SETS
    Y_hat_train = model.predict(X_train)
    Y_hat_test = model.predict(X_test)

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

    # ----------
    
    # SAVE THE TRAINED LOGISTIC REGRESSION MODEL TO A PICKLE FILE
    with open('models/logistic.pkl',  'wb') as file:
        pickle.dump(model, file)

# --------------------------------------------------

# THIS BLOCK OF CODE EXECUTES THE MAIN FUNCTION WHEN THE SCRIPT IS RUN DIRECTLY
if __name__ == '__main__':

    main()