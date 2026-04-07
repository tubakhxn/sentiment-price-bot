import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

class MarketRegimeModel:
    def __init__(self, method='kmeans', n_clusters=3):
        self.method = method
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError('Unknown method')

    def fit(self, X, y=None):
        if self.method == 'kmeans':
            self.model.fit(X)
            return self.model.labels_
        elif self.method == 'rf':
            self.model.fit(X, y)
            return self.model.predict(X)

    def predict(self, X):
        return self.model.predict(X)
