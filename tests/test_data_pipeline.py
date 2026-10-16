import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_raw_data, build_merged_dataset
from feature_engineering import FeaturePipeline

class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    def test_raw_files_exist_and_counts(self):
        patients, encounters, departments, diagnostic_tests = load_raw_data(self.data_dir)
        self.assertEqual(len(patients), 2000, "patients.csv must have 2,000 records")
        self.assertEqual(len(encounters), 8000, "encounters.csv must have 8,000 records")
        self.assertEqual(len(departments), 12, "departments.csv must have 12 records")
        self.assertEqual(len(diagnostic_tests), 5000, "diagnostic_tests.csv must have 5,000 records")

    def test_no_row_multiplication_in_merged_dataset(self):
        """Diagnostic tests (5000 rows across encounters) must not cause row duplication."""
        df = build_merged_dataset(self.data_dir)
        self.assertEqual(len(df), 8000, "Merged dataset must preserve exact encounter grain of 8,000 records")

    def test_wait_minutes_consistency(self):
        df = build_merged_dataset(self.data_dir)
        calc_wait = ((df['service_start_time'] - df['arrival_time']).dt.total_seconds() / 60.0).round()
        diff = (calc_wait - df['wait_minutes']).abs().max()
        self.assertEqual(diff, 0.0, "wait_minutes in data must precisely match service_start - arrival")

    def test_feature_pipeline_leakage_free(self):
        df = build_merged_dataset(self.data_dir)
        train_df = df[df['arrival_time'] < '2025-10-01'].copy()
        test_df = df[df['arrival_time'] >= '2025-12-01'].copy()

        pipeline = FeaturePipeline()
        X_train, y_train = pipeline.fit_transform(train_df)
        X_test, y_test = pipeline.transform(test_df)

        self.assertEqual(X_train.shape[0], len(train_df))
        self.assertEqual(X_test.shape[0], len(test_df))
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        
        # Ensure post-arrival timestamps are NOT features
        forbidden = ['service_start_time', 'service_end_time', 'discharge_time', 'outcome', 'service_minutes']
        for col in forbidden:
            self.assertNotIn(col, pipeline.feature_names)

if __name__ == '__main__':
    unittest.main()
