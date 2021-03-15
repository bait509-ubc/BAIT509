import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import altair as alt # altair is a plotting library
# alt.renderers.enable('notebook') # this helps render altair figures in a jupyter notebook
import sys
sys.path.append('code/')
from model_plotting import plot_model, plot_regression_model, plot_tree_grid # these are some custom plotting scripts I made

## Lecture 1 - Introduction to Machine Learning, the decision tree algorithm

### Question

df = pd.read_csv('data/cities_USA.csv', index_col=0)

Your tasks:

1. How many features are in this dataset?
2. How many observations are in this dataset?
3. Using sklearn, create 3 different decision tree classifiers using 3 different `max_depth` values based on this data
4. What is the accuracy of each classifier on the training data?
5. Visualise each classifier using the `plot_model()` code (or some other method)
    1. Which `max_depth` value would you choose to predict this data?
    2. Would you choose the same `max-depth` value to predict new data?
6. Do you think most of the computational effort for a decision tree takes place in the `.fit()` stage or `.predict()` stage?

### Solution

# 1
print(f"There are {df.shape[1]-1} features and 1 target.")
# 2
print(f"There are {df.shape[0]} observations.")
# 3/4/5
X = df.drop(columns='vote')
y = df[['vote']]
for max_depth in [1, 5, 10]:
    model = DecisionTreeClassifier(max_depth=max_depth).fit(X, y)
    print(f"For max_depth={max_depth}, accuracy={model.score(X, y):.2f}.")
    display(plot_model(X, y, model))
# 6
# Most of the computational effort takes places in the .fit() stage, when we create the model.

## Lecture 2 - Fundamentals of learning, train/test error

### Question

The workflow described below is really the fundamental approach to developing machine learning models: there is typically three stages - training, optimization, and testing.

Your tasks:

1. Split the cities dataset into 3 parts using `train_test_split()`: 40% training, 40% validation, 20% testing
2. How many observations are in each data set?
3. Using only the training set, fit 3 different decision tree classifiers (each with a different `max_depth`)
4. Obtain the error of each classifier on the validation data. Which model does the best? Are their big differences between your models?
5. Using the `max_depth` that gave you the lowest error, fit a new model using both the training and validation sets (80% of your original data)
6. Use this model to predict the test data. Is the error on the test data the same as the validation data? Is your result surprising?

### Solution

df = pd.read_csv('data/cities_USA.csv', index_col=0)
X = df.drop(columns='vote')
y = df[['vote']]
# 1
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, random_state=123)
# 2
print(f"There are {X_train.shape[0]} observations in the training set.")
print(f"There are {X_val.shape[0]} observations in the validation set.")
print(f"There are {X_test.shape[0]} observations in the testing set.")
# 3/4
for max_depth in [1, 5, 10, 15, 20]:
    model = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
    print(f"For max_depth={max_depth}, validation error = {1 - model.score(X_val, y_val):.2f}.")
# 5/6
model = DecisionTreeClassifier(max_depth=10).fit(X_trainval, y_trainval)
print(f"Optimum model has test error = {1 - model.score(X_test, y_test):.2f}.")
# We get a lower error here than the validation error. This is not surprising because we used
# significantly more data to build our model before testing.

## Lecture 3 - Cross-validation, KNN, loess

### Question

Using cross-validation is the standard way to optimize hyperparameters in ML model. We will practice that methodology here.

Your tasks:

1. Split the cities dataset into 2 parts using `train_test_split()`: 80% training, 20% testing.
2. Fit 5 different kNN classifiers to the training data (each with a different `k`).
3. Use 5-fold cross validation to get an estimate of validation error for each model.
4. Choose your best `k` value and fit a new model using the whole training data set.
5. Use this model to predict the test data. Is the error on the test data similar to the validation data?

### Solution

# load data
df = pd.read_csv('data/cities_USA.csv', index_col=0)
X = df.drop(columns=['vote'])
y = df[['vote']].to_numpy().ravel()
# 1
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=123)
# 2/3
print("Hyperparameter optimization")
print("***************************")
for k in np.array([1, 3, 6, 9, 12]):
    model = KNeighborsClassifier(n_neighbors=k)
    print(f"k = {k}, cross-val error = {1 - cross_validate(model, X_train, y_train, cv=5)['test_score'].mean():.2f}")
# 4/5
print("")
print("Test score")
print("**********")
model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
print(f"k = 1, test error = {1 - model.score(X_test, y_test):.2f}")

## Lecture 4 - Cross-validation, KNN, loess

### Question

In this class exercise we will practice using the pre-processing techniques we've learned in this lecture. We are going to use a real binary classification dataset of breast-cancer (read more [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)). The code below will load the dataset for you and split it into features (X) and the target (y). The features describe characteristics of cell nuclei in images of breast tissue - the features names canbe accessed using `dataset.feature_names` and you can read more about them [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). The target here is binary and composed of 0 (no breast cancer) and 1 (breast cancer).

Your tasks:

1. Split the dataset into 2 parts using `train_test_split()`: 80% training, 20% testing.
2. Fit a kNN classifier using the training data (using a `k` of your choice).
3. Calculate the error of your model on the test data.
4. Now, use `StandardScaler` to standardize your feature data (note that all attributes are numeric).
5. Refit your model and calculate the error once more. Did your result change?
6. (Bonus) repeat the above but using a DecisionTreeClassifier (use a `max_depth` of your choosing but specify `random_state=123` to negate the effect of randomness in the tree). Does scaling affect your result now? Is this surprising?

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target.astype(int)

### Solution

# 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
# 2/3
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
dt = DecisionTreeClassifier(random_state=123).fit(X_train, y_train)
print(f'No scaling test score for kNN = {1 - knn.score(X_test, y_test):.2f}')
print(f'No scaling test score for dt = {1 - dt.score(X_test, y_test):.2f}')
# 4/5
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
# 6
dt = DecisionTreeClassifier(random_state=123).fit(X_train, y_train)
print(f'Scaling test score = {1 - knn.score(X_test, y_test):.2f}')
print(f'Scaling test score for dt = {1 - dt.score(X_test, y_test):.2f}')

## Lecture 6 - Model and feature selection

### Question

1. Load the data and vectorize it using the `CountVectorizer` function.
2. Split the data into 2 parts: 80% training, 20% testing.
3. Use the `SelectKBest` function with a `chi2` metric to select the best **30** features from the dataset;
4. Now, using `GridSearchCV` for parameter tuning and 5-fold cross-validation, develop four optimum models:
    1. KNNClassifier
    2. DecisionTreeClassifier
    3. LogisitcRegression
    4. MultinomialNaiveBayes
5. Select your best model and test it on the your test data.

### Solution

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.ensemble import VotingClassifier

# Question 1 and 2
df = pd.read_csv('data/twitter-airline-sentiment.csv')
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(df['tweet'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

# Question 3
selector = SelectKBest(chi2, k=30)
X_train_30 = selector.fit_transform(X_train, y_train)
X_test_30 = selector.transform(X_test)

# Question 4
# I will first define a dictionary of the different models I want to test
models = {
    'KNN': GridSearchCV(KNeighborsClassifier(),
                        param_grid = {'n_neighbors': np.arange(1, 20, 2)},
                        cv=5),
    'DT': GridSearchCV(DecisionTreeClassifier(),
                       param_grid = {'max_depth': np.arange(1, 20, 2)},
                       cv=5),
    'LR': GridSearchCV(LogisticRegression(solver='lbfgs'),
                       param_grid = {'C': [0.01, 0.1, 1.0]},
                       cv=5),
    'NB': GridSearchCV(MultinomialNB(),
                       param_grid = {'alpha': [0.01, 0.1, 1, 10]},
                       cv=5)}
# I will now loop over each model in my dictionary and find the score
print("*** Hyperparameter tuning ***")
for name, model in models.items():
    model.fit(X_train_30, y_train)
    print(f"{name} best hyperparams = {model.best_params_}.")
    print(f"{name} error: {1 - model.best_score_:.2f}")
    
# Question 5
# Naive Bayes is the best model (although they are all similar)
print("")
print("*** Best model ***")
best_model = MultinomialNB(alpha=0.01).fit(X_train_30, y_train)
print(f"Error on test data: {1 - best_model.score(X_test_30, y_test):.2f}")

# Bonus material
# All our classifiers did well so why not use all of them to make predictions?
# We can do this with the VotingClassifier (which we'll learn about in a later lecture)
# Turns out that this ensemble approach doesn't really do better than our single Naive Bayes model
print("")
print("*** Voting classifier ***")
voter = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=7)),
                                     ('DT', DecisionTreeClassifier(max_depth=15)),
                                     ('LR', LogisticRegression(solver='lbfgs', C=1)),
                                     ('NB', MultinomialNB(alpha=0.01))],
                         voting='soft')
voter.fit(X_train_30, y_train)
print(f"Error on test data: {1 - voter.score(X_test_30, y_test):.2f}")