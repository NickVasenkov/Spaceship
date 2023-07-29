def get_score(train, test, model, n_splits):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd
    '''
    This function takes train and test sets, as well as a model for cross validation and
    a number of cross-validation splits.

    It returns:
        1) Average training Score.
        2) Average cross-validation Score
    '''

    seed_file = pd.read_csv('seed.csv', index_col=0)
    SEED = seed_file.iloc[0, 0]

    # Import random state number
    # Create a StratifiedKFold object (n_splits splits with equal proportion of positive target values)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Empty lists for collecting scores
    train_scores = []
    cv_scores = []

    # Iterate through folds
    for train_index, cv_index in skf.split(train.drop('Transported', axis=1), train['Transported']):
        # Obtain training and testing folds
        cv_train, cv_test = train.iloc[train_index], train.iloc[cv_index]

        # Fit the model
        model.fit(cv_train.drop('Transported', axis=1), cv_train['Transported'])

        # Calculate scores and append to the scores lists
        train_pred_proba = model.predict_proba(cv_train.drop('Transported', axis=1))[:, 1]
        train_scores.append(roc_auc_score(cv_train['Transported'], train_pred_proba))
        cv_pred_proba = model.predict_proba(cv_test.drop('Transported', axis=1))[:, 1]
        cv_scores.append(roc_auc_score(cv_test['Transported'], cv_pred_proba))

    return np.mean(train_scores) - np.std(train_scores), np.mean(cv_scores) - np.std(cv_scores), np.std(cv_scores)