#Create new virtual environment
#pip install pandas
#pip install pycaret

import pandas as pd
import pycaret
from pycaret.classification import *

set_config('seed', 999)

data = pd.read_csv('train_pycaret.csv')

s = setup(data=data, target="Transported",  fold_shuffle=True, session_id=999, silent=True)
compare_models()