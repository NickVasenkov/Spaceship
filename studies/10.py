## This file can be re-run over and over to continue searching.

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 0
SECONDS = 5

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 10000

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

# CHOOSE THE STUDY
STUDY_NAME = '10'

# Import packages
import joblib
import optuna
import optuna.visualization as vis
import pandas as pd

# CHOOSE THE DATASET
train = pd.read_csv('../new_datasets/train_07.csv', index_col=0)

# CHOOSE THE NUMBER OF PROCESSORS (will be multiplied by 2)
N_JOBS = -1

features_n = len(train.columns) - 1

# Load study
study = joblib.load("{}.pkl".format(STUDY_NAME))
total_seconds = pd.read_csv('{}_seconds.csv'.format(STUDY_NAME), index_col=0)
total_hours = round(total_seconds.iloc[0, 0] / 3600, 3)

# Load the global_variables
global_variables = pd.read_csv('../global_variables.csv', index_col=0)

SEED = global_variables.loc[0, 'SEED']


# The function to maximize
def train_evaluate(params):


    # Instantiate the classifier
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=SEED,
                                   n_estimators=90,
                                   n_jobs=N_JOBS,
                                   **params
                                   )

    # Calculate the cross-validation Score
    from functions.get_score import get_score

    train_score, cross_score, std, sub = get_score(global_variables, train, model=model, update=False, prepare_submission=False)

    return cross_score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {
        # 'n_estimators': optuna.distributions.IntDistribution(100, 1000),
        # 'criterion': optuna.distributions.CategoricalDistribution(['log_loss', 'entropy']),
        'criterion': trial.suggest_categorical('criterion', ['log_loss', 'gini']),
        'max_depth': trial.suggest_int('max_depth', 2, 70),
        'max_features': trial.suggest_int('max_features', 1, features_n),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 500),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-9, 1e-1, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 30),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-7, 4e-1, log=True),
        'max_samples': trial.suggest_float('max_samples', 0.3, 1)

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