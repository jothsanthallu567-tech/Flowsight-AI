import numpy as np

class LinearRegressionScratch:
    """
    Pure NumPy Multiple Linear Regression using Gradient Descent.
    Model Equation: y_hat = X @ w
    Loss Function (MSE): (1/N) * sum((y_hat - y)^2)
    Gradient: (2/N) * X.T @ (X @ error)
    Weight Update: w = w - learning_rate * gradient
    """
    def __init__(self, learning_rate=0.05, iterations=1500):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.loss_history = []
        
        for i in range(self.iterations):
            y_pred = X @ self.weights
            error = y_pred - y
            mse = np.mean(error ** 2)
            self.loss_history.append(mse)
            
            gradient = (2 / n_samples) * (X.T @ error)
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        return X @ self.weights
