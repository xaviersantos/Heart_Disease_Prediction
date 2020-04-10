from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np


class CustomImputer(TransformerMixin):
    def __init__(self, cols=None, strategy='mean'):
        self.cols = cols
        self.strategy = strategy

    def transform(self, df):
        X = df.copy()
        impute = SimpleImputer(strategy=self.strategy)
        if self.cols is None:
            self.cols = list(X.columns)
        for col in self.cols:
            if X[col].dtype == np.dtype('O'):
                X[col].fillna(X[col].value_counts().index[0], inplace=True)
            else:
                X[col] = impute.fit_transform(X[[col]])

        return X

    def fit(self, *_):
        return self


# prints to a text file and console
def log(file, text):
    if file:
        file.write(str(text) + "\n")
    print(text)
