# Slow

import pandas as pd
# In Terminal: pip install optuna
import optuna


data = pd.read_csv('04_train_prepared.csv', index_col=0)

X_train = data.drop(['Transported'], axis=1)
y_train = data['Transported']

roc_auc_scores = pd.read_csv('roc_auc_scores.csv', index_col=0)

SEED = 123

model_name = '04_RF'
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=SEED,
                               # n_estimators=516,
                               n_estimators=100,
                               criterion='log_loss',
                               max_depth=17,
                               max_features=0.7,
                               max_leaf_nodes=123,
                               min_impurity_decrease=0.00020380822483963789,
                               min_samples_leaf=2,
                               max_samples=0.9999360987512214,
                               n_jobs=-1
                               )
param_distributions = {
    # 'n_estimators': optuna.distributions.IntDistribution(100, 1000),
    # 'criterion': optuna.distributions.CategoricalDistribution(['log_loss', 'entropy']),
    'criterion': optuna.distributions.CategoricalDistribution(['log_loss', 'gini']),
    'max_depth': optuna.distributions.IntDistribution(2, 20),
    'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 0.7, 1]),
    'max_leaf_nodes': optuna.distributions.IntDistribution(20, 80),
    "min_impurity_decrease": optuna.distributions.FloatDistribution(1e-7, 5e-7, log=True),
    'min_samples_leaf': optuna.distributions.IntDistribution(2, 20),
    'ccp_alpha': optuna.distributions.FloatDistribution(0, 0.4),
    'max_samples': optuna.distributions.FloatDistribution(0.5, 1)
}

optuna_search = optuna.integration.OptunaSearchCV(model, param_distributions,
                                                  scoring='roc_auc',
                                                  timeout=120,
                                                  n_trials=None,
                                                  # max_iter = 1200, \
                                                  random_state=SEED,
                                                  n_jobs=-1)
optuna_search.fit(X_train, y_train)
print(optuna_search.best_params_)
print(optuna_search.best_score_)

roc_auc_scores[model_name] = [optuna_search.best_score_]
print(roc_auc_scores)

roc_auc_scores.to_csv('roc_auc_scores.csv')
