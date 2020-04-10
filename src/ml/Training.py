import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

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

    result_analysis(y_test, y_pred, logfile)

    log(logfile, "-" * 90)

    return model


def plot_learning_curve(model, X_train, y_train):
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            X_train,
                                                            y_train,
                                                            # shuffle training data before taking prefixes
                                                            shuffle=True,
                                                            # Number of folds in cross-validation
                                                            cv=10,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=-1,
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(0.05, 1.0, 50))

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
    plt.savefig('report/images/' + type(model).__name__ + '_lc.pdf', bbox_inches='tight')
    plt.close()


def result_analysis(y_test, y_pred, logfile):
    matrix = confusion_matrix(y_test, y_pred)
    log(logfile, "Confusion Matrix: \n" + str(matrix))
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.savefig('report/images/' + os.path.basename(logfile.name) + '_confmtarix.pdf', bbox_inches='tight')
    plt.close()

    report = classification_report(y_test, y_pred)
    log(logfile,'\nClassification report on full data set:\n' + str(report))

    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TP = matrix[1, 1]

    fnr = FN * 100 / (FN + TP)
    log(logfile, "False Negative Rate: " + str(fnr))

    fpr = FP * 100 / (FP + TN)
    log(logfile, "False Positive Rate: " + str(fpr))
