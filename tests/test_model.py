import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import build_merged_dataset
from feature_engineering import FeaturePipeline
from model_from_scratch import LinearRegressionScratch
from baseline import DepartmentMedianBaseline
from evaluation import calculate_metrics

class TestModelFromScratch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        df = build_merged_dataset(cls.data_dir)
        cls.train_df = df[df['arrival_time'] < '2025-10-01'].copy()
        cls.test_df = df[df['arrival_time'] >= '2025-12-01'].copy()

        cls.pipeline = FeaturePipeline()
        cls.X_train, cls.y_train = cls.pipeline.fit_transform(cls.train_df)
        cls.X_test, cls.y_test = cls.pipeline.transform(cls.test_df)

    def test_gradient_descent_loss_reduction(self):
        """Gradient descent must monotonically or significantly reduce training MSE loss."""
        model = LinearRegressionScratch(learning_rate=0.05, iterations=200)
        model.fit(self.X_train, self.y_train)
        self.assertTrue(len(model.loss_history) == 200)
        self.assertLess(model.loss_history[-1], model.loss_history[0], "Final loss must be strictly lower than initial loss")

    def test_model_beats_baseline(self):
        """Custom NumPy model must outperform naive department median baseline on held-out test."""
        model = LinearRegressionScratch(learning_rate=0.05, iterations=1500)
        model.fit(self.X_train, self.y_train)
        test_preds = model.predict(self.X_test)
        
        baseline = DepartmentMedianBaseline()
        baseline.fit(self.train_df)
        base_preds = baseline.predict(self.test_df)
        
        m_metrics = calculate_metrics(self.y_test, test_preds)
        b_metrics = calculate_metrics(self.y_test, base_preds)
        
        self.assertLess(m_metrics['MAE'], b_metrics['MAE'], f"Model MAE ({m_metrics['MAE']:.2f}) must beat baseline ({b_metrics['MAE']:.2f})")
        self.assertLess(m_metrics['RMSE'], b_metrics['RMSE'], f"Model RMSE ({m_metrics['RMSE']:.2f}) must beat baseline ({b_metrics['RMSE']:.2f})")

    def test_metrics_no_nan(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 29.0, 42.0])
        metrics = calculate_metrics(y_true, y_pred)
        for k, v in metrics.items():
            self.assertFalse(np.isnan(v), f"Metric {k} must not be NaN")

if __name__ == '__main__':
    unittest.main()
