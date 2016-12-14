import numpy as np
import pandas as pd

class FeatureExtractorClf(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_df):
        self.X_df_fit = X_df
        self.y_df_fit = y_df
        pass
    
    def transform(self, X_df):
        valueArray = X_df['spectra'].values
        for i in range(len(valueArray)):
            valueArray[i] = [valueArray[i][ind] for ind in range(len(valueArray[i])) if ind % 20 == 0]
        if X_df is self.X_df_fit:
            return zip(valueArray, self.y_df_fit.values)
        return zip(valueArray, [None] * len(valueArray))