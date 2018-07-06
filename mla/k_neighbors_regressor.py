import numpy as np

class KNeighborsRegressor:
    def __init__(self, k):
        self._k = k

    def fit(self, X, y):
        self._X = X
        self._y = y
    
    def predict(self, x):
        X, y, k = self._X, self._y, self._k
        distances = ((X - x) ** 2).sum(axis=1)
      
        return np.mean(y[distances.argpartition(k)[:k]])
