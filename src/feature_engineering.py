import pandas as pd
import numpy as np

class FeaturePipeline:
    def __init__(self):
        self.scaler_mean = None
        self.scaler_std = None
        self.cat_columns = ['visit_type', 'triage_level', 'sex', 'insurance_type', 'department_id']
        self.num_columns = ['age', 'staff_count', 'bed_count', 'daily_capacity', 'arrival_hour', 'day_of_week', 'recent_1h_dept_arrivals']
        self.feature_names = []
        self.one_hot_columns = []

    def extract_raw_features(self, df):
        df = df.copy()
        df['arrival_hour'] = df['arrival_time'].dt.hour
        df['day_of_week'] = df['arrival_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Fast vectorized 1-hour rolling arrival count per department
        # Uses numpy searchsorted (O(n log n)) instead of nested loops (O(n²))
        if 'recent_1h_dept_arrivals' not in df.columns:
            recent_arrivals = np.zeros(len(df), dtype=float)
            dept_ids = df['department_id'].values
            arrival_times = df['arrival_time'].values.astype('datetime64[ns]')
            one_hour_ns = np.timedelta64(1, 'h').astype('timedelta64[ns]')

            for dept in np.unique(dept_ids):
                mask = dept_ids == dept
                dept_indices = np.where(mask)[0]
                dept_times = arrival_times[mask]
                # dept_times is already sorted (df sorted by arrival_time)
                for i, idx in enumerate(dept_indices):
                    t = dept_times[i]
                    t_start = t - one_hour_ns
                    lo = np.searchsorted(dept_times, t_start, side='left')
                    hi = np.searchsorted(dept_times, t, side='left')
                    recent_arrivals[idx] = hi - lo

            df['recent_1h_dept_arrivals'] = recent_arrivals
        return df

    def fit_transform(self, train_df):
        train_df = self.extract_raw_features(train_df)
        
        encoded_cats = pd.get_dummies(train_df[self.cat_columns], drop_first=True)
        self.one_hot_columns = list(encoded_cats.columns)
        
        X_num = train_df[self.num_columns + ['has_chronic_condition', 'is_weekend']].values.astype(float)
        
        self.scaler_mean = np.mean(X_num, axis=0)
        self.scaler_std = np.std(X_num, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        
        X_num_scaled = (X_num - self.scaler_mean) / self.scaler_std
        X_cat = encoded_cats.values.astype(float)
        
        X = np.hstack([np.ones((len(train_df), 1)), X_num_scaled, X_cat])
        self.feature_names = ['intercept'] + self.num_columns + ['has_chronic_condition', 'is_weekend'] + self.one_hot_columns
        
        y = train_df['wait_minutes'].values.astype(float)
        return X, y

    def transform(self, df):
        df = self.extract_raw_features(df)
        
        encoded_cats = pd.get_dummies(df[self.cat_columns]).reindex(columns=self.one_hot_columns, fill_value=0)
        
        X_num = df[self.num_columns + ['has_chronic_condition', 'is_weekend']].values.astype(float)
        X_num_scaled = (X_num - self.scaler_mean) / self.scaler_std
        X_cat = encoded_cats.values.astype(float)
        
        X = np.hstack([np.ones((len(df), 1)), X_num_scaled, X_cat])
        y = df['wait_minutes'].values.astype(float)
        return X, y
