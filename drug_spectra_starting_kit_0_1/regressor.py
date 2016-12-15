from sklearn.ensemble import GradientBoostingRegressor                           
from sklearn.decomposition import PCA                                            
from sklearn.pipeline import Pipeline                                            
from sklearn.base import BaseEstimator                                           
import numpy as np                                                               
                                                                                 
                                                                                 
class Regressor(BaseEstimator):                                                  
    def __init__(self):                                                          
        self.n_components = 10                                        
        self.n_estimators = 140                                              
        self.learning_rate = 0.2
        self.loss = 'ls'
        self.subsample = 0.9
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = 0.0
        self.max_depth = 3
        self.min_impurity_split = 1e-07
        self.alpha = 0.9
        
        self.list_molecule = ['A', 'B', 'Q', 'R']                                
        self.dict_reg = {}                                                       
        for mol in self.list_molecule:                                           
            self.dict_reg[mol] = Pipeline([                                      
                ('pca', PCA(n_components=self.n_components)),                    
                ('reg', GradientBoostingRegressor(
                    n_estimators=self.n_estimators,                              
                    learning_rate=self.learning_rate, 
                    loss=self.loss, # 'ls'
                    subsample=self.subsample, #1.0
                    criterion='friedman_mse',
                    min_samples_split=self.min_samples_split, # 2
                    min_samples_leaf=self.min_samples_leaf, # 1
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf, # 0.0
                    max_depth=self.max_depth, # 3
                    min_impurity_split=self.min_impurity_split, # 1e-07
                    init=None,
                    max_features=None,
                    alpha=self.alpha, # 0.9
                    max_leaf_nodes=None,
                    warm_start=False,
                    random_state=42))                                            
            ])                                                                   
                                                                                 
    def fit(self, X, y):                                                         
        for i, mol in enumerate(self.list_molecule):                             
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol]                                                  
            y_mol = y[ind_mol].astype(float)                                     
            self.dict_reg[mol].fit(XX_mol, np.log(y_mol))                        
                                                                                 
    def predict(self, X):                                                        
        y_pred = np.zeros(X.shape[0])                                            
        for i, mol in enumerate(self.list_molecule):                             
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol].astype(float)                                    
            y_pred[ind_mol] = np.exp(self.dict_reg[mol].predict(XX_mol))         
        return y_pred      