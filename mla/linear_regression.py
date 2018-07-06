import numpy as np

class LinearRegression:
    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

        return self.beta
    
    def predict(self, x):
        return np.dot(self.beta, np.r_[1, x])
