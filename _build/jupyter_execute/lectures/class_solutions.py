# Exercise Solutions

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

## Lecture 1

# Define X and y
candy_df = pd.read_csv('data/candybars.csv')

X = candy_df.loc[:, 'chocolate':'multi']
y = candy_df['availability']

# Creating a model
for min_samples_split in [2, 5, 10]:
    hyper_tree = DecisionTreeClassifier(random_state=1, max_depth=8, min_samples_split=min_samples_split)
    hyper_tree.fit(X,y)
    print("For max_depth= ",min_samples_split, "accuracy=", hyper_tree.score(X, y).round(2))

4. a) Which `min_samples_split` value would you choose to predict this data? <br>

> Not necessarily the one with the greatest accuracy.
   
4. b) Would you choose the same `min_samples_split` value to predict new data?

>  No and we will explain this next lecture. 

5. Do you think most of the computational effort for a decision tree takes place in the `.fit()` stage or `.predict()` stage?

>  Most of the computational effort takes places in the .fit() stage, when we create the model.