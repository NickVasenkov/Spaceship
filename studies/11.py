## This file can be re-run over and over to continue searching.

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 1
SECONDS = 0

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 10000

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

# CHOOSE THE STUDY
STUDY_NAME = '11'

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


    # LOAD PREVIOUS BEST PARAMETERS
    best_params_from_10 = {'criterion': 'log_loss', 'max_depth': 44, 'max_features': 6, 'max_leaf_nodes': 443, 'min_impurity_decrease': 1.0589275371663915e-09, 'min_samples_leaf': 3, 'ccp_alpha': 0.0010619054629917488, 'max_samples': 0.9846315931336822}

    # Instantiate the classifier
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=SEED,
                                   n_jobs=N_JOBS,
                                   **best_params_from_10
                                   )
    model.set_params(**params)

    # Calculate the cross-validation Score
    from functions.get_score import get_score

    train_score, cross_score, std, sub = get_score(global_variables, train, model=model, update=False, prepare_submission=False)

    return cross_score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'criterion': trial.suggest_categorical('criterion', ['log_loss', 'gini', 'entropy']),

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