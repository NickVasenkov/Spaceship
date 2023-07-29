import pandas as pd
import numpy as np
data = pd.read_csv('train_prepared.csv', index_col=0)

X_train = data.drop('Transported', axis =1)
y_train = data['Transported']

SEED = 123

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Create the model
reg = LogisticRegression(max_iter =150)

# Import the modules for cross-validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Create a KFold object
kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=SEED)

# Perform cross-validation
scores = cross_val_score(reg, X_train, y_train, cv=kf, scoring="roc_auc")

# Calculate average  ROC AUC score
print("Average ROC AUC score for Logistic Regression: {}".format(np.mean(scores)))

roc_auc_scores = pd.DataFrame({'Logistic Regression': [np.mean(scores)]})
print(roc_auc_scores)

reg.fit(X_train, y_train)
feature_effect = pd.Series(data=reg.coef_[0], index=X_train.columns)
print(feature_effect)

# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier with points weighted by distance
knn = KNeighborsClassifier(weights = 'distance')

# Parameters for grid search
params_knn = {'algorithm': ['ball_tree', 'kd_tree'], \
             'leaf_size': [1, 2, 10, 30], \
             'metric': ['l1', 'l2', 'cosine', 'nan_euclidean'], \
             'n_neighbors': [20, 30, 35, 40, 45, 50, 60, 70]}

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid = GridSearchCV(estimator=knn,
                       param_grid=params_knn,
                       scoring='roc_auc',
                       cv=kf,
                       verbose=0,
                       n_jobs=-1)

# Train the models
grid.fit(X_train, y_train)

grid.best_params_

roc_auc_scores['KNeighborsClassifier'] = [grid.best_score_]
print(roc_auc_scores)

roc_auc_scores.to_csv('roc_auc_scores.csv')


