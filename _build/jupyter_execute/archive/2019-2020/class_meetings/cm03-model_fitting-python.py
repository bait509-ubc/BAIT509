# The Model-Fitting Paradigm in Python

To fit a machine learning model in python, we typically follow a common workflow. Though, you should always consult the documentation to be sure.

Here's an example using kNN classification with a fruit dataset, following [Susan Li's Medium post titled "Solving A Simple Classification Problem with Python — Fruits Lovers’ Edition"](https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2).

Import libraries and load iris data:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
## Import kNN tools:
from sklearn.neighbors import KNeighborsClassifier
## Import accuracy calculator:
from sklearn.metrics import accuracy_score
## Import train-test-split tool:
from sklearn.model_selection import train_test_split
## Import data:
fruit = pd.read_table('https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt')


Take a peak at the fruit data:

fruit.head()

First, extract the response as a list, and the predictors as an array. We'll choose fruit_name as the response, and mass, width, height, and color_score as our predictors.

y = fruit["fruit_name"]
X = fruit[["mass", "width", "height", "color_score"]]

Now, split the data into training and test data using `train_test_split(x)`:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

Second, "initiate" a model by calling the method's function. For kNN classification, it's `KNeighborsClassifier()`.

model = KNeighborsClassifier()

Now, fit the model by applying the `.fit()` method on our initiated model. This modifies the `model` object!

model.fit(X_train, y_train)

Now we can go ahead and make predictions and evaluate error by appending methods onto `model`. These _do not_ modify the `model` object!

print(model.predict(X_test))

model.score(X_test, y_test)

