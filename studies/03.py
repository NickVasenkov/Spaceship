## This file can be re-run over and over to continue searching.

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 0
SECONDS = 10

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 100

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

STUDY_NAME = '03'

# Import packages
import joblib
import optuna
import optuna.visualization as vis
import pandas as pd

# Load study
study = joblib.load("{}.pkl".format(STUDY_NAME))
total_seconds = pd.read_csv('{}_seconds.csv'.format(STUDY_NAME), index_col=0)
total_hours = round(total_seconds.iloc[0, 0] / 3600, 3)

# Load the dataset
train_full = pd.read_csv('../new_datasets/train_03.csv')

# Load the global_variables
global_variables = pd.read_csv('../global_variables.csv', index_col=0)

SEED = global_variables.loc[0, 'SEED']


# The function to maximize
def train_evaluate(params):
    # Choose variables to include

    features = []
    features_number = 0
    for key, value in params.items():
        if value:
            features.append(key)
            features_number +=1

    # Create the train set
    train = train_full[features]
    train = pd.concat([train, train_full['Transported']], axis=1)

    # UNCOMMENT TO INSTALL XGBOOST
    # !pip install xgboost
    import xgboost as xgb

    # Instantiate the regressor
    model = xgb.XGBClassifier(random_state=SEED, n_jobs=-1)

    # Calculate the cross-validation Score
    from functions.get_score import get_score

    if features_number > 0:
        train_score, cross_score, std, sub = get_score(global_variables, train, model=model, update=False, prepare_submission=False)
    else:
        cross_score = 0

    return cross_score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {
        # Variables inclusion
        'Age': trial.suggest_categorical('Age', [True, False]),
        'RoomService': trial.suggest_categorical('RoomService', [True, False]),
        'FoodCourt': trial.suggest_categorical('FoodCourt', [True, False]),
        'ShoppingMall': trial.suggest_categorical('ShoppingMall', [True, False]),
        'Spa': trial.suggest_categorical('Spa', [True, False]),
        'VRDeck': trial.suggest_categorical('VRDeck', [True, False]),
        'GroupSize': trial.suggest_categorical('GroupSize', [True, False]),

    }
    return train_evaluate(params)


# Run the session
study.optimize(objective, timeout=RUNNING_TIME, n_trials=N_TRIALS, n_jobs=-1)
total_seconds.iloc[0, 0] = total_seconds.iloc[0, 0] + RUNNING_TIME

# Save back to the file
joblib.dump(study, "{}.pkl".format(STUDY_NAME))
total_seconds.to_csv('{}_seconds.csv'.format(STUDY_NAME))

print("Best trial:", study.best_trial.number)
print("Best average cross-validation ROC AUC:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)
total_hours = round(total_seconds.iloc[0, 0] / 3600, 3)
print("Total running time (hours):", total_hours)