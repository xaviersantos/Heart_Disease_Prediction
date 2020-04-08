import pandas as pd

from ml.Models import logistic_regression, naive_bayes, k_nearest_neighbors, decision_tree, random_forest


def final_score(X_train, y_train, X_test, y_test):
    # initialize an empty list
    accuracy = []

    # list of algorithms names
    classifiers = ['Logistic Regression', 'Naive Bayes','KNN','Decision Trees','Random Forests']

    # list of algorithms with parameters
    models = [logistic_regression(X_train, y_train, X_test, y_test),
              naive_bayes(X_train, y_train, X_test, y_test),
              k_nearest_neighbors(X_train, y_train, X_test, y_test),
              decision_tree(X_train, y_train, X_test, y_test),
              random_forest(X_train, y_train, X_test, y_test)
              ]

    # loop through algorithms and append the score into the list
    for i in models:
        model = i
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        accuracy.append(score)

    # create a dataframe from accuracy results
    summary = pd.DataFrame({'accuracy': accuracy}, index=classifiers)
    print(summary)
