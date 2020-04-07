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

    return model
