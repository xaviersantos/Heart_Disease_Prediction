
def show_data(data):
    # shape
    print(data.shape)

    # head
    print(data.head(10))
    with open('report/tables/data_head.tex', 'w') as tf:
        tf.write(data.head(10).to_latex())

    # descriptions
    print(data.describe())
    with open('report/tables/data_describe.tex', 'w') as tf:
        tf.write(data.describe().to_latex())

    # info
    print(data.info())

    # class distribution
    print(data.groupby('Diagnosis').size())
    with open('report/tables/data_class_distribution.tex', 'w') as tf:
        tf.write(data.groupby('Diagnosis').size().to_latex())

    # data correlation
    print(data.corr()["Diagnosis"].abs().sort_values(ascending=False))
    with open('report/tables/data_correlation.tex', 'w') as tf:
        tf.write((data.corr()["Diagnosis"].abs().sort_values(ascending=False).to_latex()))


# shows the categorical data as name instead of value
def pretty_data(df):
    df.loc[df.Sex == 1, 'Sex'] = "Male"
    df.loc[df.Sex == 0, 'Sex'] = "Female"

    df.loc[df.ChestPainType == 1, 'ChestPainType'] = "Typical angina"
    df.loc[df.ChestPainType == 2, 'ChestPainType'] = "Atypical angina"
    df.loc[df.ChestPainType == 3, 'ChestPainType'] = "Non-angina"
    df.loc[df.ChestPainType == 4, 'ChestPainType'] = "Asymptomatic"

    df.loc[df.FastingBloodSugar == 0, 'FastingBloodSugar'] = "< 120 mg/dl"
    df.loc[df.FastingBloodSugar == 1, 'FastingBloodSugar'] = "> 120 mg/dl"

    df.loc[df.RestingECG == 0, 'RestingECG'] = "Normal"
    df.loc[df.RestingECG == 1, 'RestingECG'] = "ST-wave abnorm."
    df.loc[df.RestingECG == 2, 'RestingECG'] = "left ventr. hypertrophy"

    df.loc[df.ExerciseInducedAngina == 0, 'ExerciseInducedAngina'] = "No"
    df.loc[df.ExerciseInducedAngina == 1, 'ExerciseInducedAngina'] = "Yes"

    df.loc[df.ST_slope == 1, 'ST_slope'] = "Upsloping"
    df.loc[df.ST_slope == 2, 'ST_slope'] = "Flat"
    df.loc[df.ST_slope == 3, 'ST_slope'] = "Downsloping"

    df.loc[df.ThalliumStressTest == 3, 'ThalliumStressTest'] = "Normal"
    df.loc[df.ThalliumStressTest == 6, 'ThalliumStressTest'] = "Fixed defect"
    df.loc[df.ThalliumStressTest == 7, 'ThalliumStressTest'] = "Reversible defect"

    df.loc[df.Diagnosis == 0, 'Diagnosis'] = "No disease"
    df.loc[df.Diagnosis == 1, 'Diagnosis'] = "Disease"

    return df
