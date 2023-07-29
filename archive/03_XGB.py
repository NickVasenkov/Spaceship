import pandas as pd
data = pd.read_csv('train_prepared.csv', index_col=0)

X_train = data.drop(['Transported'],  axis =1)
y_train = data['Transported']

roc_auc_scores = pd.read_csv('roc_auc_scores.csv', index_col=0)

SEED = 123

# In Terminal: pip install optuna
import optuna

model_name = 'XGB_Optuna'
# In Terminal: pip install xgboost
# Create the DMatrix
import xgboost as xgb
dmatrix = xgb.DMatrix(data=X_train, label=y_train)

# Instantiate the regressor
model = xgb.XGBClassifier(random_state=SEED)

# # specify parameters via map
# param = {'booster': 'dart',
#          'max_depth': 5, 'learning_rate': 0.1,
#          'objective': 'binary:logistic',
#          'sample_type': 'uniform',
#          'normalize_type': 'tree',
#          'rate_drop': 0.1,
#          'skip_drop': 0.5}
#



param_distributions = {
    'n_estimators': optuna.distributions.IntDistribution(40, 100, step=5),
    'max_depth': optuna.distributions.IntDistribution(20, 40),
    'max_leaves': optuna.distributions.IntDistribution(0, 150, step=10),
    'grow_policy': optuna.distributions.CategoricalDistribution(['depthwise', 'lossguide']),
    'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.2, step=0.01),
    'booster': optuna.distributions.CategoricalDistribution(['gbtree', 'dart']),
    'tree_method': optuna.distributions.CategoricalDistribution(['approx', 'hist']), #We may use 'exact' method for the best params (it is slow),
    'gamma': optuna.distributions.FloatDistribution(1e-2, 1e2, log=True),
    'min_child_weight': optuna.distributions.FloatDistribution(1e-2, 1e2, log=True),
    'subsample': optuna.distributions.FloatDistribution(0.7, 1.00, step=0.05),
    'colsample_bytree': optuna.distributions.FloatDistribution(0.7, 1.00, step=0.05),
    #'num_parallel_tree': optuna.distributions.IntDistribution(1, 5)




}

optuna_search = optuna.integration.OptunaSearchCV(model, param_distributions, \
                                                  scoring='roc_auc', \
                                                  timeout = 600, \
                                                  n_trials = None, \
                                                  #max_iter = 1200, \
                                                  random_state = SEED, \
                                                  n_jobs=-1)
optuna_search.fit(X_train, y_train)
print(optuna_search.best_params_)
print(optuna_search.best_score_)

roc_auc_scores[model_name] = [optuna_search.best_score_]
print(roc_auc_scores)

roc_auc_scores.to_csv('roc_auc_scores.csv')

#{'n_estimators': 65, 'max_depth': 35, 'max_leaves': 70, 'grow_policy': 'lossguide', 'learning_rate': 0.03, 'booster': 'dart', 'tree_method': 'hist', 'gamma': 0.03679141517068278, 'min_child_weight': 0.14230098619447734, 'subsample': 0.75, 'colsample_bytree': 0.8999999999999999}
#0.8763826326797327