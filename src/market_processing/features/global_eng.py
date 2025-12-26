import pandas as pd

class FeatureEngineer:
    def __init__(self, method='pca'):
        self.method = method
    
    def transform(self, joint_df: pd.DataFrame):
        if self.method == 'correlation_matrix':
            # return rolling correlation matrix
            pass
        elif self.method == 'wavelet':
            # apply wavelet transform to all columns
            pass