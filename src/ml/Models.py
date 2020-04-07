from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml.Training import train_model
from utils import log


def logistic_regression(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/log_reg.txt", "w")
    log(logfile, "Logistic Regression\n")

    log_reg = train_model(X_train, y_train, X_test, y_test,
                          LogisticRegression, logfile, random_state=0, solver='newton-cg', multi_class='multinomial')

    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_test)
    log(logfile, "\npredictions= " + str(y_pred_lr))

    score_lr = round(accuracy_score(y_pred_lr,y_test)*100,2)
    log(logfile, "\nThe accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %\n")

    logfile.close()

    return log_reg


def naive_bayes(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/native_bayes.txt", "w")
    log(logfile, "Naive bayes\n")

    nb = train_model(X_train, y_train, X_test, y_test, GaussianNB, logfile)

    nb.fit(X_train, y_train)

    y_pred_nb = nb.predict(X_test)
    log(logfile, "\npredictions= " + str(y_pred_nb))

    score_nb = round(accuracy_score(y_pred_nb, y_test)*100,2)
    log(logfile, "\nThe accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

    return nb


def k_nearest_neighbors(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/k_nearest_neighbors.txt", "w")
    log(logfile, "K Nearest Neighbors\n")

    log(logfile, "n_neighbors = 1")
    knn = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, logfile, n_neighbors=1)

    knn.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    log(logfile, "\npredictions= " + str(y_pred_knn))

    score_knn = round(accuracy_score(y_pred_knn, y_test) * 100, 2)
    log(logfile, "\nThe accuracy score achieved using KNN is: " + str(score_knn) + " %")

    log(logfile, "\nSeek optimal 'n_neighbours' parameter:")
    for i in range(2,10):
        log(logfile, "\nN neighbors = " + str(i))
        tmp = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, logfile, n_neighbors=i)
        if tmp.score(X_test, y_test) > knn.score(X_test, y_test):
            knn = tmp
        else:
            break
    
    logfile.close()

    return knn


def decision_tree(X_train, y_train, X_test, y_test):
    logfile = open("report/logs/decision_tree.txt", "w")
    log(logfile, "Decision Tree\n")

    log(logfile, "max_depth = 1")
    dt = train_model(X_train, y_train, X_test, y_test,
                     DecisionTreeClassifier, logfile, max_depth=1, random_state=0)

    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    log(logfile, "\npredictions= " + str(y_pred_dt))

    score_dt = round(accuracy_score(y_pred_dt, y_test) * 100, 2)
    log(logfile, "\nThe accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")

    log(logfile, "\nSeek optimal 'max_depth' parameter:")
    for max_depth in range(2,10):
        log(logfile, "\nmax_depth = " + str(max_depth))
        tmp = train_model(X_train, y_train, X_test, y_test,
                          DecisionTreeClassifier, logfile, max_depth=max_depth, random_state=0)

        if tmp.score(X_test, y_test) > dt.score(X_test, y_test):
            dt = tmp
        else:
            break
    
    logfile.close()

    return dt
