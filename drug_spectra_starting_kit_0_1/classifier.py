from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.svm import SVC

class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 10
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)), 
            ('svc', SVC(C=8.3, kernel='poly', degree=10, gamma=0.05, coef0=3.4, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None))
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)