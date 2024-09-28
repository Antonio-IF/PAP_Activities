# Tomar cada una de las filas de nuestro dataset y entrenar el modelo, queremos guardar la columna de Y que en este caso es binaria 
# subir nuestro archivo ya con la Y. 

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, f1_score


def main():
    data = pd.read_csv('data/credit_train.csv')
    X = data.drop('Y', axis=1)
    Y = data['Y']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size=0.255555,
                                                        random_state=0)

    model = LogisticRegression().fit(X_train,Y_train)

    Y_hat_train = model.predict(X_train)
    Y_hat_test = model.predict(X_test)

    f1_train = f1_score(Y_train, Y_hat_train)
    f1_test = f1_score(Y_test, Y_hat_test)

    accuracy_train = accuracy_score(Y_train, Y_hat_train)
    accuracy_test = accuracy_score(Y_test, Y_hat_test)

    print('F1 score train:', f1_train)
    print('F1 score test:', f1_test)
    print('Accuracy train:', accuracy_train)
    print('Accuracy test:', accuracy_test)

    # Saving our model
    with open('models/logistic.pkl',  'wb') as file:
        pickle.dump(model, file)



if __name__ == '__main__':
    main()