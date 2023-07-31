## This file can be re-run over and over to continue searching.

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 10
SECONDS = 0

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 10000

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

# CHOOSE THE STUDY
STUDY_NAME = '08'

# Import packages
import joblib
import optuna
import optuna.visualization as vis
import pandas as pd

# CHOOSE THE DATASET
train = pd.read_csv('../new_datasets/train_07.csv', index_col=0)

# CHOOSE THE NUMBER OF PROCESSORS (will be multiplied by 2)
N_JOBS = 2

# Load study
study = joblib.load("{}.pkl".format(STUDY_NAME))
total_seconds = pd.read_csv('{}_seconds.csv'.format(STUDY_NAME), index_col=0)
total_hours = round(total_seconds.iloc[0, 0] / 3600, 3)

# Load the global_variables
global_variables = pd.read_csv('../global_variables.csv', index_col=0)

SEED = global_variables.loc[0, 'SEED']


# The function to maximize
def train_evaluate(params):

    import xgboost as xgb

    # Instantiate the classifier
    model = xgb.XGBClassifier(random_state=SEED, n_jobs=N_JOBS, **params)

    # Calculate the cross-validation Score
    from functions.get_score import get_score

    train_score, cross_score, std, sub = get_score(global_variables, train, model=model, update=False, prepare_submission=False)

    return cross_score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {
        # 'n_estimators': trial.suggest_int(40, 100, step=20),
        'max_depth': trial.suggest_int('max_depth', 2, 50),
        'max_leaves': trial.suggest_int('max_leaves', 20, 500),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, step=0.01),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        # We may use 'exact' method for the best params (it is slow),
        'gamma': trial.suggest_float('gamma', 1e-2, 1e2, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.00, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.00, step=0.05)
        # 'num_parallel_tree': optuna.distributions.IntDistribution(1, 5)

    }
    return train_evaluate(params)


# Run the session
study.optimize(objective, timeout=RUNNING_TIME, n_trials=N_TRIALS, n_jobs=N_JOBS)
total_seconds.iloc[0, 0] = total_seconds.iloc[0, 0] + RUNNING_TIME

# Save back to the file
joblib.dump(study, "{}.pkl".format(STUDY_NAME))
total_seconds.to_csv('{}_seconds.csv'.format(STUDY_NAME))

print("Best trial:", study.best_trial.number)
print("Best average cross-validation Score:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)
total_hours = round(total_seconds.iloc[0, 0] / 3600, 3)
print("Total running time (hours):", total_hours)