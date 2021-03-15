# BAIT 509: Business Applications of Machine Learning
## Lecture 3 - Cross-validation, KNN, loess
Tomas Beuzen, 13th January 2020

# Lecture outline
- [0. Recap (5 mins)](#0)
- [1. Lecture learning objectives](#1)
- [2. Splitting your data and cross-validation (20 mins)](#2)
- [3. A conceptual introduction to kNN (5 mins)](#3)
- [4. Class Exercise: A toy example of kNN (10 mins)](#4)
- [5. kNN on a real dataset (10 mins)](#5)
- [--- Break --- (10 mins)](#break)
- [6. kNN regression (10 mins)](#6)
- [7. Loess/Lowess scatter smoothing (10 mins)](#7)
- [8. Class Exercise: hyperparameter optimization (20 mins)](#8)
- [9. What we've learned today (5 mins)](#9)
- [10. Summary questions to ponder](#10)

# Announcements

- Assignment 1 has been released (access it on Canvas)
- It is due 11:59pm next Monday 20th January
- You will submit **both** a jupyter notebook and html file
- Use the following code in your terminal to convert your jupyter notebook to html:
> `jupyter nbconvert --to html_embed assignment1.ipynb`
- Remember to use Piazza for Q&A
- Due to a conflict I've had to change my office hours to 1-2pm Monday
- You can also contact the TA to arrange an office hour at any time (I'll post an announcement on Canvas on how to contact them)

# 0. The story so far...  (5 mins) <a id=0></a>

- **Machine learning** is a way of making predictive models
- We've learned just one algorithm so far (**decision trees**)
- We can control the way our algorithms learn a model by tuning **hyperparameters**
- We usually use 3 datasets in ML development:
    - **Training dataset**: used to fit the model/learn parameters
    - **Validation dataset**: used to optimize model hyperparameters
    - **Testing dataset**: reserved data used to give us our "gold standard" estimate of model performance

# 1. Lecture learning objectives  <a id=1></a>

- Describe cross-validation and use `.cross_validate()` to calculate cross-validation error
- Broadly describe how the kNN algorithm works.
- Discuss the effect of using a small/large value of the hyperparameter *k* when using the *k*NN algorithm.
- Discuss the difference between parametric and non-parametric machine learning models.

# 2. Splitting your data and cross-validation (15 mins) <a id=2></a>
- Last lecture we dicussed how there are usually 3 datasets involved in machine learning:
    1. **Training set**: used to learn the model
    2. **Validation set**: used to optimize the model (e.g., choose the best hyperparameters)
    3. **Testing set**: used to test the model (lock it in a "vault" until you're ready to test)
- But how does one choose these splits?
- Well, if you have a lots of data (1000/10,000/1,000,000 observations??? - "lots" depends on your problem) simple % splits are often good enough
- Some common examples are shown below:

<img src='./img/splits.png' width="400">

- A few rules of thumb:
    - We want a large enough training set to be able to develop a good model
    - We want a large enough validation/test set to get a good approximation of model performance on unseen data

### Problems with having a single train/test split
- Only using a portion of your full data set for training/testing (data is our most precious resource!!!)
- If your dataset is small you might end up with a tiny training/testing set
- Might be unlucky with your splits such that they don't well represent your data (shuffling data, as is done in `train_test_split()`, is not immune to being unlucky!)

### A solution: k-fold cross-validation
- $k$-fold cross-validation (CV) is one of the most common techniques used in practice
- CV helps us combine the training/validation steps and use more data!
- Get a more "robust" measure of error on unseen data
- How it works:
    1. Split your data into k-folds ($k>2$, often $k=10$)
    2. Each "fold" gets a turn at being the validation set, while the other folds are used for fitting the model
    3. By this approach, we get $k$ prediction error estimates
    4. We then (typically) take the average of these $k$ estimates as our validation error

<img src='./img/cv.png' width="600">

- Cross-validation allows us to use more of our data during training/validation
- Typically we will split off a test set at the very start, and then use cross-validation to optimize our model

<img src='./img/cv2.png' width="600">

- We can use sklearn's `cross_validate()` to do cross-validation for us
- It is imported from the `model_selection` module
- We'll also import other libraries we need for this lecture while we're at it

from sklearn.model_selection import cross_validate # this is the cross-validation function

import numpy as np
import pandas as pd
import altair as alt
import sys
sys.path.append('code/')
from model_plotting import plot_model, plot_knn_grid, plot_lowess, plot_lowess_grid
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

- Let's load the cities data and split it into a train and test portion

df = pd.read_csv('data/cities_USA.csv', index_col=0)
X = df.drop(columns=['vote'])
y = df[['vote']]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=123)

X

y

- Now let's create a Decision Tree model with a `max_depth=3`

model = DecisionTreeClassifier(max_depth=3)

- Now we can use `cross_validate()` to do cross-validation
- Note that we don't have to call `.fit()` when using the function, it does it automatically for us
- We will do 5-fold cross-validation (i.e., $k$ = 5) using **only the training data**
- We should therefore get 5 results
- The `cv` argument can be used to specify how many folds of CV you want to do in `cross_validate()`
- The argument `return_train_score` is also useful, it will tell the function to also return the training error
- The output of `cross_validate()` is a dictionary of useful information
- Note that the output is in terms of accuracy, and they call it `test_score` - although it is really `validation_score`

cv_score = cross_validate(model,
                          X_train,
                          y_train,
                          cv=5,
                          return_train_score=True)
cv_score

- Typically we average the results of cross-validation to get an estimate of error
- This is our validation error

print(f"The mean cross-validation error is: {1 - cv_score['test_score'].mean():.2f}")

- Typically we will test different values of our hyperparameter(s)
- Then choose the one that gives the best mean cross-validation score
- But for now, let's just test our model on the test data

model.fit(X_train, y_train) # first fit the model
print(f"The test error is: {1 - model.score(X_test, y_test):.2f}")

- It's not unusual for out test error to be higher than our validation error
- We specifically chose the hyperparameter(s) that gave the lowest error on our training/validation data
- This doesn't mean they will give the lowest error on the test data (but we hope they do!)

# 3. A conceptual introduction to kNN (5 mins) <a id=3></a>

- Okay, it's time to learn a new algorithm!
- The k Nearest Neighbors (kNN) algorithm
- This is a fairly simple algorithm that is best understood by example
- Here is some toy data showing 2 features and 1 target (with 2 classes: <font color="blue">blue</font> and <font color="orange">orange</font>)
- I want to predict the point in grey

<img src='./img/scatter.png' width="400">

- An intuitive way to do this is predict the grey point using the same label as the next "closest" point (*k* = 1)
- We would predict a target of <font color="orange">**orange**</font> in this case

<img src='./img/scatter_k1.png' width="400">

- We could also use the 3 closest points (*k* = 3)...
- Of the 3 closest points, 1 is <font color="orange">**orange**</font>, 2 are <font color="blue">**blue**</font>
- We would therefore predict a target of <font color="blue">**blue**</font> in this case

<img src='./img/scatter_k3.png' width="400">

## 4. Class Exercise: A toy example of kNN (10 mins) <a id=4></a>

Consider this toy dataset:

$$ X = \begin{bmatrix}5 & 2\\4 & 3\\  2 & 2\\ 10 & 10\\ 9 & -1\\ 9& 9\end{bmatrix}, \quad y = \begin{bmatrix}0\\0\\1\\1\\1\\2\end{bmatrix}.$$

1. If $k=1$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$?
2. If $k=3$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$?
3. If $k=3$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$ if we were doing regression rather than classification?

- Now let's validate our answers using sklearn's kNN implementation
- The code below imports necessary libraries for this lecture
- It also creates the `X` and `y` data shown above

X = pd.DataFrame({'feature1': [5, 4, 2, 10, 9, 9],
                  'feature2': [2, 3, 2, 10, -1, 9]})
y = pd.DataFrame({'target': [0, 0, 1, 1, 1, 1]})

X

y

- Now let's create a kNN classifier
- We can import the classifier from the `sklearn.neighbors` module

from sklearn.neighbors import KNeighborsClassifier

- And now we can create a model the same we've learned previously

knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)

- An aside...
- What is that warning? It has to do with the data type (a dataframe) we are passing the the model
- The warning is telling us that sklearn is converting the dataframe to a 1D array (a vector)
- For the purpose of this lecture, I'm going to turn that warning off with the following code

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

- So let's do question 1 above
- Predict `x = [0, 0]`

knn.predict(np.atleast_2d([0, 0]))

- How about for question 2 above
- Predict `x = [0, 0]` but this time with `k=3`

knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
knn.predict(np.atleast_2d([0, 0]))

- Finally, question 3 requires to import the kNN regressor model
- We'll talk more about kNN regression later

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=3).fit(X, y)
knn.predict(np.atleast_2d([0, 0]))

# 5. kNN on a real dataset (10 mins) <a id=5></a>

- Let's reload the cities dataset to see how kNN performs on a real dataset

df = pd.read_csv('data/cities_USA.csv', index_col=0)
X = df.drop(columns=['vote'])
y = df[['vote']]

- Let's use our code to plot up some different kNN models (with different k values)
- First we will do `k=1`

knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)
plot_model(X, y, knn)

1 - knn.score(X, y)

- How about a larger `k`

knn = KNeighborsClassifier(n_neighbors=20).fit(X, y)
plot_model(X, y, knn)

#### How does kNN relate to decision trees?
- Large *k* is "simple" like a decision stump
- It's not actually simple because we have to compare to a large number of observations!
- Small *k* is like a deep tree

# -------- Break (10 mins) -------- <a id="break"></a>

# 6. kNN regression (10 mins) <a id=6></a>
- In kNN regression we take the average of the *k* nearest neighbours
- Note: regression plots more naturally in 1D, classification in 2D, but of course we can do either for any $d$
- The code below creates some synthetic data and plots to help us visualise kNN regression
- We see that kNN regression is a way of average (smoothing) data to make predictions
- The higher the *k* the "smoother" the predictions

X = np.atleast_2d(np.linspace(-7, 6.5, 60)).T
y = (np.sin(X.T**2/5) + np.random.randn(60)*0.1).T
plot_knn_grid(X, y, k=[1, 5, 10, 20])

# 7. Loess/Lowess scatter smoothing (10 mins) <a id=7></a>
- Another approach for fitting a smooth curve through a set of data points
- Kind of like a combination of kNN and least squares regression
- But this [video](https://www.youtube.com/watch?v=Vf7oJ6z2LCc) provides an excellent explanation

<img src='./img/loess.png' width="500">

Source: [StatQuest](https://www.youtube.com/watch?v=Vf7oJ6z2LCc)

- We can import the lowess model from the statsmodel library as shown below

from statsmodels.nonparametric.smoothers_lowess import lowess

- Let's see how the model works for different $k$ values

X = np.linspace(-7, 6.5, 60)
y = np.sin(X**2/5) + np.random.randn(60)*0.1
plot_lowess_grid(X, y, k=[1, 5, 10, 20])

# 8. Class Exercise: hyperparameter optimization (20 mins) <a id=8></a>

Using cross-validation is the standard way to optimize hyperparameters in ML model. We will practice that methodology here.

Your tasks:

1. Split the cities dataset into 2 parts using `train_test_split()`: 80% training, 20% testing.
2. Fit 5 different kNN classifiers to the training data (each with a different `k`).
3. Use 5-fold cross validation to get an estimate of validation error for each model.
4. Choose your best `k` value and fit a new model using the whole training data set.
5. Use this model to predict the test data. Is the error on the test data similar to the validation data?

# load data
df = pd.read_csv('data/cities_USA.csv', index_col=0)
X = df.drop(columns=['vote'])
y = df[['vote']].to_numpy().ravel()
# Question 1
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=123)
# Questions 2/3
print("Hyperparameter optimization")
print("***************************")
for k in [1, 3, 6, 9, 12]:
    model = KNeighborsClassifier(n_neighbors=k)
    print(f"k = {k}, cross-val error = {1 - cross_validate(model, X_train, y_train, cv=5)['test_score'].mean():.2f}")
# Questions 4/5
print("")
print("Test score")
print("**********")
model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
print(f"k = 1, test error = {1 - model.score(X_test, y_test):.2f}")

# 9. What we've learned today <a id="9"></a>

- Cross-validation as a way of efficiently combining training/validation and choosing hyperparameters
- The kNN algorithm for classification and regression
- Lowess/loess smoothing 

# 10. Questions to ponder <a id="10"></a>

1. What happens if our data is of different scales?
2. How do we handle categorical data?