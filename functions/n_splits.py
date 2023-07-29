import pandas as pd

# Load files
train = pd.read_csv('../new_datasets/train_01.csv', index_col=0)
test =  pd.read_csv('../new_datasets/test_01.csv', index_col=0)
scores_df = pd.read_csv('../scores_df.csv', index_col=0)
global_variables = pd.read_csv('../global_variables.csv', index_col=0)

# Import a global random state number
SEED = global_variables.loc[0, 'SEED']


import xgboost as xgb
# Create the DMatrix
dmatrix = xgb.DMatrix(train.drop(['Transported'], axis=1), label=train['Transported'])

# Instantiate the regressor
model = xgb.XGBClassifier(random_state=SEED, n_jobs=-1)

# Create array of random states for testing number of splits
import random
random_states = random.sample(range(1, 100000), 30)

# Calculate Scores and stds vs. number of splits
from functions.get_score import get_score

tradeoff = pd.DataFrame({'random_state': [], 'n_splits': [], 'Train Score': [],
                                   'Cross-Val Score': [], 'Cross-val std': []})

for random_state in random_states:
    for n_splits in range(2, 8):
        index = len(tradeoff)
        tradeoff.loc[index, 'random_state'] = random_state
        tradeoff.loc[index, 'n_splits'] = n_splits
        train_score, cross_score, cross_scores_std, subm = get_score(train, test, model, scores_df,
                                                                     update=False,
                                                                     prepare_submission=False,
                                                                     n_splits=n_splits,
                                                                     global_n_splits=False,
                                                                     random_state=random_state,
                                                                     global_random_state=False)
        tradeoff.loc[index, 'Train Score'] = train_score
        tradeoff.loc[index, 'Cross-Val Score'] = cross_score
        tradeoff.loc[index, 'Cross-val std'] = cross_scores_std


print(tradeoff)



import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(data=tradeoff, x='n_splits', y='Train Score')
plt.show()
sns.lineplot(data=tradeoff, x='n_splits', y='Cross-Val Score')
plt.show()
sns.lineplot(data=tradeoff, x='n_splits', y='Cross-val std')
plt.show()

