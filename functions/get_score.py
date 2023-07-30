def get_score(global_variables, train, test=None, model=None, scores_df=None,
              update=True, comment='',
              prepare_submission=True,
              n_splits=3, global_n_splits=True,
              random_state=123, global_random_state=True):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd
    '''
    This function takes the global variables DataFrame, 
    the processed train and test sets,
    an estimator for cross validation,
    a number of cross-validation splits and
    the scores dataframe.
    
    If 'update' is True, then the scores dataframe is being updated eith new scores and 'comment'.
    
    If 'prepare_submission' is True, a submission DataFrame is returned
    
    'n_splits' can be chosen for StratifiedKFold In this case,'global_n_splits' has to be set to False
    
    'random_state' can be chosen for StratifiedKFold In this case,'global_random_state' has to be set to False

    It returns:
        1) Average training Score.
        2) Average cross-validation Score
        3) Standard Deviation of cross-validation scores
        4) A submission DataFrame (if 'prepare_submission' is True)
        
    (Score is described in Spaceship.ipynb -> 00. Baseline)
    '''

    # Import global n_splits
    if global_n_splits:
        n_splits = global_variables.loc[0, 'N_SPLITS']
    # Import a global random state number
    if global_random_state:
        random_state = global_variables.loc[0, 'SEED']

    # Create a StratifiedKFold object ('n_splits' splits with equal proportion of positive target values)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

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

    # Calculate Scores
    train_score = np.mean(train_scores) - np.std(train_scores)
    cross_score = np.mean(cv_scores) - np.std(cv_scores)

    # Update the scores DataFrame
    if update:
        scores_df.loc[len(scores_df)] = [comment, train_score, cross_score, np.nan]

    submission = "prepare_submission=False"

    if prepare_submission:
        # Fit the model to the whole training set:
        model.fit(train.drop('Transported', axis=1), train['Transported'])
        # Prepare the submission DataFrame
        test_Ids = pd.read_csv('test_Ids.csv', index_col=0).reset_index(drop=True)
        test_pred = model.predict(test)
        test_pred = [True if i == 1 else False for i in test_pred]
        test_pred = pd.DataFrame(test_pred, columns=['Transported'])
        submission = pd.concat([test_Ids, test_pred], axis=1)

    return train_score, cross_score, np.std(cv_scores), submission