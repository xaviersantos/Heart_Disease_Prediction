import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

from utils import log


def train_model(X_train, y_train, X_test, y_test, classifier, logfile=None, **kwargs):
    # instantiate model
    model = classifier(**kwargs)

    # train model
    model.fit(X_train, y_train)

    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    log(logfile, f"Train accuracy: {fit_accuracy:0.2%}")
    log(logfile, f"Test accuracy: {test_accuracy:0.2%}")

    y_pred = model.predict(X_test)
    log(logfile, "\npredictions= " + str(y_pred))

    score = round(accuracy_score(y_pred, y_test) * 100, 2)
    log(logfile,"\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    log(logfile, "The accuracy score achieved is: " + str(score) + " %")

    log(logfile, "-" * 90)

    return model


def plot_learning_curve(model, X_train, y_train):
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            X_train,
                                                            y_train,
                                                            # Number of folds in cross-validation
                                                            cv=10,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=-1,
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(0.015, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    plt.savefig('report/images/' + type(model).__name__ + '_lc.pdf', bbox_inches='tight')

