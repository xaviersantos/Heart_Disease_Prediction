from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml.Training import train_model, plot_learning_curve
from utils import log


def logistic_regression(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/log_reg.txt", "w")
    log(logfile, "Logistic Regression\n")

    log_reg = train_model(X_train, y_train, X_test, y_test,
                          LogisticRegression, logfile, random_state=0, solver='newton-cg', multi_class='multinomial')

    plot_learning_curve(log_reg, X_train, y_train)

    logfile.close()

    return log_reg


def naive_bayes(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/native_bayes.txt", "w")
    log(logfile, "Naive bayes\n")

    log(logfile, "GaussianNB\n")
    gnb = train_model(X_train, y_train, X_test, y_test, GaussianNB, logfile)

    log(logfile, "BernoulliNB\n")
    bnb = train_model(X_train, y_train, X_test, y_test, BernoulliNB, logfile)

    if gnb.score(X_test, y_test) > bnb.score(X_test, y_test):
        nb = gnb
    else:
        nb = bnb

    plot_learning_curve(nb, X_train, y_train)

    logfile.close()

    return nb


def k_nearest_neighbors(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/k_nearest_neighbors.txt", "w")
    log(logfile, "K Nearest Neighbors\n")

    log(logfile, "n_neighbors = 1")
    knn = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, logfile, n_neighbors=1)

    log(logfile, "\nSeek optimal 'n_neighbours' parameter:")
    for i in range(2,10):
        log(logfile, "\nN neighbors = " + str(i))
        tmp = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, logfile, n_neighbors=i)
        if tmp.score(X_test, y_test) > knn.score(X_test, y_test):
            knn = tmp
        else:
            break

    plot_learning_curve(knn, X_train, y_train)

    logfile.close()

    return knn


def decision_tree(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/decision_tree.txt", "w")
    log(logfile, "Decision Tree\n")

    log(logfile, "max_depth = 1")
    dt = train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, logfile, max_depth=1, random_state=0)

    log(logfile, "\nSeek optimal 'max_depth' parameter:")
    for max_depth in range(2,10):
        log(logfile, "\nmax_depth = " + str(max_depth))
        tmp = train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, logfile, max_depth=max_depth, random_state=0)
        if tmp.score(X_test, y_test) > dt.score(X_test, y_test):
            dt = tmp
        else:
            break

    plot_learning_curve(dt, X_train, y_train)

    logfile.close()

    return dt


def random_forest(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/random_forest.txt", "w")
    log(logfile, "Random Forest\n")

    # Random forest with 100 trees
    rf = train_model(X_train, y_train, X_test, y_test,
                     RandomForestClassifier, logfile, max_depth=1, n_estimators=100, random_state=0, n_jobs=-1)

    log(logfile, "\nSeek optimal 'max_depth' parameter:")
    for max_depth in range(2, 10):
        log(logfile, "\nmax_depth = " + str(max_depth))
        tmp = train_model(X_train, y_train, X_test, y_test,
                          RandomForestClassifier, logfile, max_depth=max_depth, n_estimators=100, random_state=0,
                          n_jobs=-1)

        if tmp.score(X_test, y_test) > rf.score(X_test, y_test):
            rf = tmp
        elif tmp.score(X_test, y_test) != rf.score(X_test, y_test):
            break

    plot_learning_curve(rf, X_train, y_train)

    logfile.close()

    return rf
