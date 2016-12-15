from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import numpy as np
from scipy.stats import multivariate_normal

class Classifier(BaseEstimator):
    def __init__(self):
        self.group_m = []
        self.group_c = []
        self.group_mean = []
        self.group_cov = []
        self.group_pdf = []
        self.group_size = []
        self.group_n = 0
        self.group_total_size = 0;
        pass

    def fit(self, X, y):
        epsilon = 3.5 * 1e-7
        n = len(X)
        self.group_total_size = n
        X_values = [x[0] for x in X]
        X_molecules = [x[1][0] for x in X]
        X_concentrations = [x[1][1] for x in X]
        molecules = np.unique(X_molecules)
        concentrations = np.unique(X_concentrations)
        for m in molecules:
            for c in concentrations:
                mc_values = [X_values[i] for i in range(n) if X_molecules[i] == m and X_concentrations[i] == c]
                if len(mc_values) > 0:
                    self.group_m.append(m)
                    self.group_c.append(c)
                    self.group_mean.append([0] * len(mc_values[0]))
                    for i in range(len(mc_values[0])):
                        self.group_mean[self.group_n][i] = np.mean([x[i] for x in mc_values])
                    self.group_cov.append(np.cov(np.transpose(mc_values)))
                    for i in range(len(mc_values[0])):
                        for j in range(len(mc_values[0])):
                            self.group_cov[self.group_n][i][j] += epsilon
                        self.group_cov[self.group_n][i][i] += epsilon
                    self.group_pdf.append(multivariate_normal(mean=self.group_mean[self.group_n], cov=self.group_cov[self.group_n]))
                    self.group_size.append(len(mc_values))
                    self.group_n += 1
                    

    def predict(self, X):
        pass

    def predict_proba(self, X):
        mol = ['A', 'B', 'Q', 'R']
        result = []
        for i in range(len(X)):
            result.append([0,0,0,0])
            m = -1
            for j in range(self.group_n):
                for k in range(len(mol)):
                    if self.group_m[j] == mol[k]:
                        m = k
                result[i][m] += self.group_pdf[j].pdf(X[i][0]) * self.group_size[j] / float(self.group_total_size)
            prob_sum = sum(result[i])
            for j in range(4):
                result[i][j] /= prob_sum
            
        return np.array(result)