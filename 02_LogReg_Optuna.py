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

model_name = 'LinReg_Optuna'
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(warm_start=True)
param_distributions = {
    'penalty': optuna.distributions.CategoricalDistribution(['l2']),
    "C": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True),
    'solver': optuna.distributions.CategoricalDistribution(['liblinear', 'newton-cholesky', 'sag', 'lbfgs']),


}

optuna_search = optuna.integration.OptunaSearchCV(model, param_distributions, \
                                                  scoring='roc_auc', \
                                                  timeout = 30, \
                                                  n_trials = 200, \
                                                  #max_iter = 1200, \
                                                  random_state = SEED, \
                                                  n_jobs=-1)
optuna_search.fit(X_train, y_train)
print(optuna_search.best_params_)
print(optuna_search.best_score_)

roc_auc_scores[model_name] = [optuna_search.best_score_]
print(roc_auc_scores)

roc_auc_scores.to_csv('roc_auc_scores.csv')

