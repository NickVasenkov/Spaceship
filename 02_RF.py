# Slow

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

model_name = 'RF_Optuna'
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=SEED)
param_distributions = {
    'n_estimators': optuna.distributions.IntDistribution(800, 2000),
    'min_samples_leaf': optuna.distributions.IntDistribution(1, 20),
    'max_features': optuna.distributions.CategoricalDistribution(['log2', 'sqrt', 0.8, 1.0])
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

