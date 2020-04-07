from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ML.Training import train_model


def logistic_regression(X_train, y_train, X_test, y_test):
    log_reg = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')

    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_test)
    print(y_pred_lr)

    score_lr = round(accuracy_score(y_pred_lr,y_test)*100,2)

    print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

    model = train_model(X_train, y_train, X_test, y_test, LogisticRegression)

    return model


def naive_bayes(X_train, y_train, X_test, y_test):
    nb = train_model(X_train, y_train, X_test, y_test, GaussianNB)

    nb.fit(X_train, y_train)

    y_pred_nb = nb.predict(X_test)
    print(y_pred_nb)

    score_nb = round(accuracy_score(y_pred_nb, y_test)*100,2)
    print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
    
    model = train_model(X_train, y_train, X_test, y_test, GaussianNB)

    return model


def k_nearest_neighbors(X_train, y_train, X_test, y_test):
    knn = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=8)

    knn.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    print(y_pred_knn)

    score_knn = round(accuracy_score(y_pred_knn, y_test) * 100, 2)
    print("The accuracy score achieved using KNN is: " + str(score_knn) + " %")

    model = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier)

    # Seek optimal 'n_neighbours' parameter
    for i in range(1,10):
        print("N neighbors = " + str(i))
        tmp = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=i)
        if tmp.score(X_test, y_test) > model.score(X_test, y_test):
            model = tmp

    return model


def decision_tree(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier(max_depth=3, random_state=0)

    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    print(y_pred_dt)

    score_dt = round(accuracy_score(y_pred_dt, y_test) * 100, 2)

    print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")

    model = DecisionTreeClassifier(max_depth=1, random_state=0)
    model.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))

    for max_depth in range(2,10):
        tmp = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        tmp.fit(X_train, y_train)
        print("\nmax_depth = " + str(max_depth))
        print("Accuracy on training set: {:.3f}".format(tmp.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(tmp.score(X_test, y_test)))

        if tmp.score(X_test, y_test) > model.score(X_test, y_test):
            model = tmp
        else:
            return model

    return model
