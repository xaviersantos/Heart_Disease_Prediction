from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml.Training import train_model, plot_learning_curve
from utils import log


def logistic_regression(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/log_reg.tex", "w")
    print("Logistic Regression\n")

    log_reg = train_model(X_train, y_train, X_test, y_test,
                          LogisticRegression, logfile, random_state=0, solver='liblinear')

    plot_learning_curve(log_reg, X_train, y_train)

    logfile.close()

    return log_reg


def naive_bayes(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/naive_bayes.tex", "w")
    print("Naive Bayes\n")

    print("\nGausian")
    gnb = train_model(X_train, y_train, X_test, y_test, GaussianNB)

    print("\nBernoulli")
    bnb = train_model(X_train, y_train, X_test, y_test, BernoulliNB, logfile)

    if gnb.score(X_test, y_test) > bnb.score(X_test, y_test):
        nb = gnb
    else:
        nb = bnb

    plot_learning_curve(nb, X_train, y_train)

    logfile.close()

    return nb


def k_nearest_neighbors(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/k_nearest_neighbors.tex", "w")
    print("K Nearest Neighbors\n")

    knn = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=1)

    n_neighbors = 1
    for i in range(2,10):
        tmp = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=i)
        if tmp.score(X_test, y_test) > knn.score(X_test, y_test):
            knn = tmp
            n_neighbors = i

    log(logfile, "N neighbors = " + str(n_neighbors) + "\\\\")
    train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, logfile, n_neighbors=n_neighbors)

    plot_learning_curve(knn, X_train, y_train)

    logfile.close()

    return knn


def decision_tree(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/decision_tree.tex", "w")
    print("Decision Tree\n")

    print("Max depth = 1")
    dt = train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, max_depth=1, random_state=0)

    max_depth = 1
    for i in range(2,10):
        tmp = train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, max_depth=i, random_state=0)
        if tmp.score(X_test, y_test) > dt.score(X_test, y_test):
            dt = tmp
            max_depth = i

    log(logfile, "Max depth = " + str(max_depth) + "\\\\")
    train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, logfile, max_depth=max_depth, random_state=0)

    plot_learning_curve(dt, X_train, y_train)

    logfile.close()

    return dt


def random_forest(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/random_forest.tex", "w")
    print("Random Forest\n")

    # Random forest with 100 trees
    rf = train_model(X_train, y_train, X_test, y_test,
                     RandomForestClassifier, max_depth=1, n_estimators=100, random_state=0, n_jobs=-1)

    max_depth = 1
    for i in range(2, 10):
        tmp = train_model(X_train, y_train, X_test, y_test, RandomForestClassifier,
                          max_depth=i, n_estimators=100, random_state=0, n_jobs=-1)

        if tmp.score(X_test, y_test) > rf.score(X_test, y_test):
            rf = tmp
            max_depth = i
        # elif tmp.score(X_test, y_test) != rf.score(X_test, y_test):
        #     break

    log(logfile, "Max depth = " + str(max_depth) + "\\\\")
    train_model(X_train, y_train, X_test, y_test, RandomForestClassifier, logfile,
                max_depth=max_depth, n_estimators=100, random_state=0, n_jobs=-1)

    plot_learning_curve(rf, X_train, y_train)

    logfile.close()

    return rf
