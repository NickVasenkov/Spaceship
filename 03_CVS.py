# Works slowly. Score 0.85

import pandas as pd
data = pd.read_csv('train_prepared.csv', index_col=0)

X_train = data.drop('Transported', axis =1)
y_train = data['Transported']

SEED = 123

# In Terminal: pip install optuna
import optuna

from sklearn.svm import SVC

clf = SVC()
param_distributions = {
    "C": optuna.distributions.FloatDistribution(1e2, 1e10, log=True)
}
optuna_search = optuna.integration.OptunaSearchCV(clf, param_distributions, \
                                                  scoring='roc_auc', \
                                                  timeout = 30, \
                                                  max_iter = 1200, \
                                                  random_state = SEED, \
                                                  n_jobs=-1)
optuna_search.fit(X_train, y_train)

#0.73
