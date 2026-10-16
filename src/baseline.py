import numpy as np
import pandas as pd

class DepartmentMedianBaseline:
    def __init__(self):
        self.overall_median = 0
        self.dept_medians = {}

    def fit(self, train_df):
        self.overall_median = train_df['wait_minutes'].median()
        self.dept_medians = train_df.groupby('department_id')['wait_minutes'].median().to_dict()

    def predict(self, df):
        return df['department_id'].map(self.dept_medians).fillna(self.overall_median).values
