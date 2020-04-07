from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ML.Models import logistic_regression, naive_bayes, k_nearest_neighbors, decision_tree
from visualization.ViewData import *
from visualization.PreAnalysis import *


def load_dataset():
    path = "Dataset/processed.cleveland.data"

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
    data = pd.read_csv(path, names=names, na_values=["?"], converters={'Diagnosis': lambda x: int(int(x) > 0)})

    # drop the rows with NaN results
    data.dropna(inplace=True)

    return data, data[numerical], data[categorical]


def split_dataset(data):
    predictors = data.drop("Diagnosis", axis=1)
    target = data["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
    print("Training features have {0} records and Testing features have {1} records.".format(y_train.shape[0], y_test.shape[0]))

    return X_train, X_test, y_train, y_test


def scaling_data(data):
    data = pd.get_dummies(data, columns=['Sex', 'ChestPainType', 'FastingBloodSugar', 'RestingECG',
                                         'ExerciseInducedAngina', 'ST_slope', 'NumMajorVessels', 'ThalliumStressTest'])
    standardScaler = StandardScaler()
    columns_to_scale = ['Age', 'RestingBloodPressure', 'Cholesterol', 'MaxHeartRate', 'ST_depression']
    data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])

    return data

def final_score(X_train, y_train, X_test, y_test):
    # initialize an empty list
    accuracy = []

    # list of algorithms names
    classifiers = ['Logistic Regression', 'Naive Bayes','KNN','Decision Trees',
                   #'Random Forests'
                   ]

    # list of algorithms with parameters
    models = [logistic_regression(X_train, y_train, X_test, y_test),
              naive_bayes(X_train, y_train, X_test, y_test),
              k_nearest_neighbors(X_train, y_train, X_test, y_test),
              decision_tree(X_train, y_train, X_test, y_test),
              #random_forest(X_train, y_train, X_test, y_test)
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



def main():
    data, data_numerical, data_categorical = load_dataset()

    data_pd = pretty_data(data.copy())
    data_categorical = pretty_data(data_categorical.copy())

    # outliers_data = outliers(data)
    # outliers_data.to_csv("Dataset/outliers.csv")

    # show_data(data)
    # histogram(data_numerical)
    # view_categorical(data_categorical)
    # box_diagrams(data_pd)
    # percentage_general(data_pd)
    # percentage_by_sex(data_pd)
    # age_distribution(data_pd)
    # scatter_pairs(data_numerical.join(data_pd['Diagnosis']))
    # correlation(data_pd)

    data_processed = scaling_data(data.copy())
    X_train, X_test, y_train, y_test = split_dataset(data_processed)

    final_score(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
