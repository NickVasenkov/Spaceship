## This file can be re-run over and over to continue searching.

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 2
SECONDS = 0

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 100

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

# CHOOSE THE STUDY
STUDY_NAME = '09'

# Import packages
import joblib
import optuna
import optuna.visualization as vis
import pandas as pd

# CHOOSE THE DATASET
train = pd.read_csv('../new_datasets/train_07.csv', index_col=0)

# CHOOSE THE NUMBER OF PROCESSORS (will be multiplied by 2)
N_JOBS = 4

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

    # LOAD PREVIOUS BEST PARAMETERS
    best_params_from_08 = {'max_depth': 13, 'max_leaves': 29, 'grow_policy': 'lossguide', 'learning_rate': 0.08, 'booster': 'dart', 'tree_method': 'approx', 'gamma': 1.834861192156488, 'min_child_weight': 0.02502298800121074, 'subsample': 0.8500000000000001, 'colsample_bytree': 0.9}

    # Instantiate the classifier
    model = xgb.XGBClassifier(random_state=SEED, n_jobs=N_JOBS, **best_params_from_08)

    model.set_params(**params)

    # Calculate the cross-validation Score
    from functions.get_score import get_score

    train_score, cross_score, std, sub = get_score(global_variables, train, model=model, update=False, prepare_submission=False)

    return cross_score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'tree_method': trial.suggest_categorical('tree_method', ['exact', 'approx', 'hist'])

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