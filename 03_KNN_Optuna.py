
import pandas as pd
import numpy as np
import sklearn
data = pd.read_csv('train_prepared.csv', index_col=0)

X_train = data.drop(['Transported'],  axis =1)
y_train = data['Transported']

roc_auc_scores = pd.read_csv('roc_auc_scores.csv', index_col=0)

SEED = 123

# In Terminal: pip install optuna
import optuna

from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier with points weighted by distance
knn = KNeighborsClassifier()
param_distributions = {
    'leaf_size': optuna.distributions.IntDistribution(1, 1000),
    'n_neighbors': optuna.distributions.IntDistribution(2, 400),
    'algorithm': optuna.distributions.CategoricalDistribution(['ball_tree', 'kd_tree']),
    #'metric': optuna.distributions.CategoricalDistribution(['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'dice', 'euclidean', 'hamming', 'infinity', 'jaccard', 'kulsinski', 'l1', 'l2', 'manhattan', 'matching', 'minkowski', 'p', 'pyfunc', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath']),
    'metric': optuna.distributions.CategoricalDistribution(['chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p']),
    'weights': optuna.distributions.CategoricalDistribution(['uniform', 'distance']),
}
print(sorted(sklearn.neighbors.VALID_METRICS['kd_tree']))
optuna_search = optuna.integration.OptunaSearchCV(knn, param_distributions, \
                                                  scoring='roc_auc', \
                                                  timeout = 180, \
                                                  n_trials = 200, \
                                                  #max_iter = 1200, \
                                                  random_state = SEED, \
                                                  n_jobs=-1)
optuna_search.fit(X_train, y_train)
print(optuna_search.best_params_)
print(optuna_search.best_score_)

roc_auc_scores['KNN_Optuna'] = [optuna_search.best_score_]
print(roc_auc_scores)

roc_auc_scores.to_csv('roc_auc_scores.csv')

