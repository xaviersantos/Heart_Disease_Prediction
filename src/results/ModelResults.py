import pandas as pd
from sklearn.metrics import accuracy_score

from ml.Models import logistic_regression, naive_bayes, k_nearest_neighbors, decision_tree, random_forest
from utils import log


def start_training(X_train, y_train, X_test, y_test):
    # list of algorithms with parameters
    models = [logistic_regression(X_train, y_train, X_test, y_test),
              naive_bayes(X_train, y_train, X_test, y_test),
              k_nearest_neighbors(X_train, y_train, X_test, y_test),
              decision_tree(X_train, y_train, X_test, y_test),
              random_forest(X_train, y_train, X_test, y_test)
              ]

    # loop through algorithms and append the score into the list
    for model in models:
        model.fit(X_train, y_train)

    logfile = open("report/tables/original_test_summary.tex", 'w')
    evaluate_models(models, X_test, y_test, logfile)
    logfile.close()

    return models


def start_prediction(models, data, logfile=None):
    X = data.drop("Diagnosis", axis=1)
    y = data["Diagnosis"]

    evaluate_models(models, X, y, logfile)


def evaluate_models(models, X, y, logfile=None):
    # initialize an empty list
    predictions = []
    accuracy = []

    # list of algorithms names
    classifiers = ['Logistic Regression', 'Naive Bayes', 'KNN', 'Decision Trees', 'Random Forests']

    # loop through algorithms and append the score into the list
    for model in models:
        y_pred = model.predict(X)
        predictions.append(y_pred)
        score = round(accuracy_score(y_pred, y) * 100, 2)
        accuracy.append(score)

    # create a dataframe with the model's predictions
    results = pd.DataFrame({'prediction': predictions}, index=classifiers)
    # create a dataframe from accuracy results
    summary = pd.DataFrame({'accuracy': accuracy}, index=classifiers)
    print(results)
    log(logfile, summary.sort_values(by='accuracy', ascending=False).to_latex())

