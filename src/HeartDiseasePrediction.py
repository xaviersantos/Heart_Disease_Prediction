from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from data_analysis.PreAnalysis import *
from data_analysis.ViewData import *
from results.ModelResults import start_training, start_prediction


def load_dataset(path):
    names = ['Age', 'Sex', 'ChestPainType', 'RestingBloodPressure', 'Cholesterol', 'FastingBloodSugar', 'RestingECG',
             'MaxHeartRate', 'ExerciseInducedAngina', 'ST_depression', 'ST_slope', 'NumMajorVessels',
             'ThalliumStressTest', 'Diagnosis']

    # separate the numerical from the categorical values
    numerical = [
        'Age', 'RestingBloodPressure', 'Cholesterol', 'MaxHeartRate', 'ST_depression'
    ]
    categorical = [
        'Sex', 'ChestPainType', 'FastingBloodSugar', 'RestingECG', 'ExerciseInducedAngina', 'ST_slope',
        'NumMajorVessels', 'ThalliumStressTest', 'Diagnosis'
    ]

    # the result(num) is changed to binary (0=no heart problems; 1=heart problems) and the '?' is considered NaN
    data = pd.read_csv(path, names=names, na_values='?', converters={'Diagnosis': lambda x: int(int(x) > 0)})

    # replace the rows with NaN results with the median
    data = fill(data, numerical, categorical)

    return data, data[numerical], data[categorical]


def fill(data, numerical, categorical):
    for feature in categorical:
        data.fillna(data[feature].value_counts(), inplace=True)

    for feature in numerical:
        data.fillna(data[feature].mean(), inplace=True)

    return data


def split_dataset(data):
    predictors = data.drop("Diagnosis", axis=1)
    target = data["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
    print("Training features have {0} records and Testing features have {1} records.".format(y_train.shape[0],
                                                                                             y_test.shape[0]))

    return X_train, X_test, y_train, y_test


def main():
    data, data_numerical, data_categorical = load_dataset("dataset/processed.cleveland.data")

    data_pd = pretty_data(data.copy())
    data_categorical = pretty_data(data_categorical.copy())

    outliers_data = outliers(data)
    outliers_data.to_csv("dataset/outliers.csv")

    show_data(data)
    histogram(data_numerical)
    view_categorical(data_categorical)
    box_diagrams(data_pd)
    percentage_general(data_pd)
    percentage_by_sex(data_pd)
    age_distribution(data_pd)
    scatter_pairs(data_numerical.join(data_pd['Diagnosis']))
    correlation(data_pd)

    X_train, X_test, y_train, y_test = split_dataset(data.copy())

    models = start_training(X_train, y_train, X_test, y_test)

    # show the results of the models applied to the original dataset
    logfile = open("report/logs/original_summary.txt", 'w')
    log(logfile, "ORIGINAL (\\W TRAINING SET) DATASET SUMMARY\n")
    start_prediction(models, data, logfile)
    logfile.close()

    # show the results of the models applied to the swiss dataset
    logfile = open("report/logs/switzerland_summary.txt", 'w')
    log(logfile, "SWISS DATASET SUMMARY\n")
    switzerland_data = load_dataset("dataset/processed.switzerland.data")
    start_prediction(models, switzerland_data[0], logfile)
    logfile.close()

    # show the results of the models applied to the hungarian dataset
    logfile = open("report/logs/hungary_summary.txt", 'w')
    log(logfile, "HUNGARIAN DATASET SUMMARY\n")
    hungary_data = load_dataset("dataset/processed.hungarian.data")
    start_prediction(models, hungary_data[0], logfile)
    logfile.close()

    # show the results of the models applied to the hungarian dataset
    logfile = open("report/logs/va_summary.txt", 'w')
    log(logfile, "V.A. MEDICAL CENTER DATASET SUMMARY\n")
    va_data = load_dataset("dataset/processed.va.data")
    start_prediction(models, va_data[0], logfile)
    logfile.close()


if __name__ == '__main__':
    main()
