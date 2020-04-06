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


def main():
    data, data_numerical, data_categorical = load_dataset()

    data_pd = pretty_data(data.copy())
    data_categorical = pretty_data(data_categorical.copy())

    show_data(data)
    histogram(data_numerical)
    view_categorical(data_categorical)
    box_diagrams(data_pd)
    percentage_general(data_pd)
    percentage_by_sex(data_pd)
    age_distribution(data_pd)
    scatter_pairs(data_numerical.join(data_pd['Diagnosis']))
    correlation(data_pd)


if __name__ == '__main__':
    main()
