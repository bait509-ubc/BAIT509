# Lecture 1 - Introduction to Machine Learning & The Decision Tree Algorithm

*Hayley Boyce, Monday April 19th 2021*

## Welcome

Welcome to Bait 509 - Business Application of Machine Learning.

```{note}
Buckle up because there are going to be a lot of new concepts here but in the lyrics of Trooper "We're here for a good time, 
Not a long time". 
```

### Course Learning Objectives

1.	Describe fundamental machine learning concepts such as: supervised and unsupervised learning, regression and classification, overfitting, training/validation/testing error, parameters and hyperparameters, and the golden rule.
2.	Broadly explain how common machine learning algorithms work, including: naïve Bayes, k-nearest neighbors, decision trees, support vector machines, and logistic regression.
3.	Identify when and why to apply data pre-processing techniques such as scaling and one-hot encoding.
4.	Use Python and the scikit-learn package to develop an end-to-end supervised machine learning pipeline.
5.	Apply and interpret machine learning methods to carry out supervised learning projects and to answer business objectives.

### Course Structure
- 2 lectures per week (Synchonus lecture + class activity)
- My office hours: 1-2pm Thursday on Zoom
- TA office hours: TBD
- Camera on policy!
- Additional resources here: Canvas course URL
- We will be using [Piazza](https://piazza.com/configure-classes/winterterm22020/bait509ba1) for discussions and questions.
- Assessments:

| Assessment       | Weight       | Due                            |
|     :---:        | :---:        | :---:                          |
| 3 Assignments    | 60%(20% each)| April 28th, May 10th, May 19th |
| 1 Quiz           | 10%          | May 5th (24 hours to complete) |
| Final Project    | 30%          | May 29th                       |


All assessments will be submitted via Canvas 

### Who I am

![](imgs/hi.png)


- I am have an undergraduate degree in Applied Mathematics from the University of Western Ontario
- I have a master's degree in Data Science 
- I have experience in Python, R, Tableau, some Latex, Some HTML and CSS and dabble in a few other things - Jack of all trades, master of none. 

### Python, Jupyter, Visualizations 

- In this course we be using Python and Jupyter notebooks for lectures as well as assignments. 
- You are free to use [Anaconda distribution](https://www.anaconda.com/distribution/) to install and manage your Python package installations
- If you are using anaconda, you can install a few key packages we will be using in the course, by typing the following at the command line:
    > `conda install pandas numpy scikit-learn matplotlib jupyter altair seaborn python-graphviz`

- Or you can simpling use pip and type in at the command line: 
    > `pip install pandas numpy scikit-learn matplotlib jupyter altair seaborn graphviz`


- We will be making visualizations for this course and I give the option of plotting using any Python library but I strongly recommend getting familar with [`altair`](https://altair-viz.github.io/index.html). I have 2 very quick slide decks that teach you a bit about how to plot using `altair`. 
From the course [Programming in Python for Data Science](https://prog-learn.mds.ubc.ca/en/)
 - Module 1, exercise 31, 32, 33
 - Module 2, exercise 29, 30
 
And if you want to dive further there is a whole course dedicated to visualizing plots using `altair` called [Data Visualization](https://viz-learn.mds.ubc.ca/en/).


### Lecture Learning Objectives 

- Explain motivation to study machine learning.
- Differentiate between supervised and unsupervised learning.
- Differentiate between classification and regression problems.
- Explain machine learning terminology such as features, targets, training, and error.
- Explain the `.fit()` and `.predict()` paradigm and use `.score()` method of ML models.
- Broadly describe how decision trees make predictions.
- Use `DecisionTreeClassifier()` and `DecisionTreeRegressor()` to build decision trees using scikit-learn.
- Explain the difference between parameters and hyperparameters.
- Explain how decision boundaries change with `max_depth`.

## What is Machine Learning (ML)?

# Let's import our libraries
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


Machine learning is all around us. You can find it in things like: 


![](imgs/examples.png)


- Voice assistance
- Google news
- Recommender systems
- Face recognition
- Auto completion
- Stock market predictions
- Character recognition
- Self-driving cars
- Cancer diagnosis
- Drug discovery

Machine Learning has no clear consensus in how it is define but essentially it is a different way to think about problem solving. 

-  As the [ISLR](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf) book puts it, ML is a “*vast set of tools for understanding data*” 
- Usually we think logically and mathematically. In the Machine Learning paradiagm we instead take data and some output and our Machine Learning algorithm will return a program which can be used to make predictions and give an output for data it has yet to see. 

### Examples of Machine Learning 

*In all the the upcoming examples, Don't worry about the code. Just focus on the input and output in each example.*


#### Example 1: Predict Housing Prices
**Data Attribution:** <a href="https://www.kaggle.com/harlfoxem/housesalesprediction" target="_blank">house sales prediction dataset.</a>

df = pd.read_csv("data/kc_house_data.csv")
df = df.drop(columns=["id", "date"])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=4)
train_df.head()

X_train = train_df.drop(columns=["price"])
y_train = train_df["price"]
X_test = test_df.drop(columns=["price"])
y_test = test_df["price"]

from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train);

pred_df = pd.DataFrame({"Predicted price": model.predict(X_test[0:4]).tolist(), "Actual price": y_test[0:4].tolist()})
df_concat = pd.concat([X_test[0:4].reset_index(drop=True), pred_df], axis=1)
df_concat.head()

#### Example 2: Predict Creditcard Default 

**Data Attribution:** <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud" target="_blank">credit card fraud detection data set</a>

cc_df = pd.read_csv("data/creditcard_sample.csv")

train_df, test_df = train_test_split(cc_df, test_size=4, random_state=603)
train_df.head(5)

from xgboost import XGBClassifier
X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Class'])
y_test = test_df['Class']
model = XGBClassifier()
model.fit(X_train, y_train);

pred_df = pd.DataFrame({"predicted_label": model.predict(X_test).tolist()})
df_concat = pd.concat([test_df.reset_index(drop=True), pred_df], axis=1)
df_concat

model.score(X_test, y_test)

#### Example 3: Image Classification

import pandas as pd
import numpy as np
from PIL import Image
import sys
sys.path.append('code/')
from toy_classifier import classify_image


img = Image.open("imgs/apple.jpg")
img

classify_image(img, 5)

## Types of Machine Learning

- **Supervised learning** (this course)

- Unsupervised learning

### Supervised Learning: 

> Example: Labelling emails as spam or not

- In supervised machine learning, we have a set of observations usually denoted with an uppercase `X`.

- We also have a set of corresponding targets usually denoted with a lowercase `y`. 

- Our goal is to define a function that relates `X` to `y`. 

- We then use this function to predict the targets of new examples. 



<img src='imgs/sup-learning.png'  width = "75%" alt="404 image" />


### Supervised Learning: (not going into detail here)  

> Example: Categorizing Google News articles.


- In unsupervised learning, we are not given targets and are only given observations `X`. 

- We apply some clustering algorithms to create a model that finds patterns in our data and groups together similar characteristics from our data.



<img src='imgs/unsup-learning.png'  width = "75%" alt="404 image" />

## Types of Supervised Learning: Classification vs Regressiion

- Classification
- Regression

### Classifications 
- **Classification** predicting among two or more categories, also known as classes.
    > - *Example1*: Predict whether a customer will default on their credit card or not. 
    > - *Example2*: Predict whether the letter grade of a student (A,B,C,D or F)
    
<br>

- **Regression** predicting a continuous (in other words, a number) value.
    > - Example1: Predict housing prices
    > - Example2: Predict a student's score in this course's quiz2
    
    
<img src="imgs/classification-vs-regression2.png" width = "90%" alt="404 image" />


## Let's Practice! 


Are the following supervised or unsupervised problems?

1. Finding groups of similar properties in a real estate data set.
2. Predicting real estate prices based on house features like number of rooms, learning from past sales as examples.
3. Idenitfying groups of animals given features such as "number of legs", "wings/no wings", "fur/no fur", etc.
4. Detecting heart disease in patients based on different test results and history.
5. Grouping articles on different topics from different news sources (something like Google News app).

Are the following classification or regression problems?

1. Predicting the price of a house based on features such as number of rooms and the year built.
2. Predicting if a house will sell or not based on features like the price of the house, number of rooms, etc.
3. Predicting your grade in BAIT 509 based on past grades.
4. Predicting what product a consumer is most likely to purchase based on their internet browsing patterns.
5. Predicting a cereal’s manufacturer given the nutritional information.

## Tabular Data and Terminology

Basic terminology used in ML:

- **examples/observations** = rows 
- **features/variables** = inputs (columns)
- **targets** = outputs (one special column)
- **training** = learning = fitting

<img src="imgs/sup-ml-terminology2.png" width = "90%" alt="404 image" />


### Example:

- This [dataset](http://simplemaps.com/static/demos/resources/us-cities/cities.csv) contains longtitude and latitude data for 400 cities in the US.
- Each city is labelled as `red` or `blue` depending on how they voted in the 2012 election.

df = pd.read_csv('data/cities_USA.csv', index_col=0).sample(20, random_state=77)
df

df.shape

In this dataset, we have:
- 2 **features**, (3 columns = 2 **features** + 1 target) and,
- 20 **examples**.

Our **target** column is `vote` since that is what we are interesting in predicting. 

### Splitting the data

Before we build any model (we are getting to that so hang tight), we need to make sure we have the right "parts" aka inputs and outputs. 

That means we need to split up our tabular data into the features and the target, also known as $X$ and $y$.

$X$ is all of our features in our data, which we also call our ***feature table***. <br>
$y$ is our target, which is what we are predicting.





X = df.drop(columns=["vote"])
y = df["vote"]

X.head()

y.head()

## Decision Tree Algorithm

### A conceptual introduction to Decision Trees

- Shown below is some hypothetical data with 2 features (x and y axes) and 1 target (with 2 classes).
- The supervised learning problem here is to predict whether a particular observaton belongs to the <font color='blue'>**BLUE**</font> or <font color='orange'>**ORANGE**</font> class.
- A fairly intuitive way to do this is to simply use thresholds to split the data up.

<img src='imgs/scatter_dt1.png'  width = "40%" alt="404 image" />

- For example, we can **split** the data at `Feature_1 = 0.47`.
- Everything **less than** the split we can classify as <font color='orange'>**ORANGE**</font>
- Everything **greater than** the split we can classify as <font color='blue'>**BLUE**</font>
- By this method, we can successfully classify 7 / 9 observations.

<img src='imgs/scatter_dt2.png'  width = "40%" alt="404 image" />

- But we don't have to stop there, we can make another split!
- Let's now split the section that is greater than `Feature_1 = 0.47`, using `Feature_2 = 0.52`.
- We now have the following conditions:
    - If `Feature_1 > 0.47` and `Feature_2 < 0.52` classify as <font color='blue'>**BLUE**</font>
    - If `Feature_1 > 0.47` and `Feature_2 > 0.52` classify as <font color='orange'>**ORANGE**</font>
- Using these rules, we now successfully classify 8 / 9 observations.

<img src='imgs/scatter_dt3.png'  width = "40%" alt="404 image" />

- Okay, let's add one more threshhold.
- Let's make a final split of the section that is less than `Feature_1 = 0.47`, using `Feature_2 = 0.6`.
- By this methodology we have successfully classified all of our data.

<img src='imgs/scatter_dt4.png'  width = "40%" alt="404 image" />

- What we've really done here is create a group of `if` statements:
    - If `Feature_1 < 0.47` and `Feature_2 < 0.6` classify as <font color='orange'>**ORANGE**</font>
    - If `Feature_1 < 0.47` and `Feature_2 > 0.6` classify as <font color='blue'>**BLUE**</font>
    - If `Feature_1 > 0.47` and `Feature_2 < 0.52` classify as <font color='blue'>**BLUE**</font>
    - If `Feature_1 > 0.47` and `Feature_2 > 0.52` classify as <font color='orange'>**ORANGE**</font>
- This is easier to visualize as a tree:

<img src='imgs/toy_tree.png'  width = "40%" alt="404 image" />

- We just made our first decision tree!

Before we go forward with learning about decision tree classifiers and aggressors we need to understand the structure of a decision tree.
Here is the key terminology that you will have to know: 

- **Root**: Where we start making our conditions.
- **Branch**:  A branch connects to the next node (statement). Each branch represents either true or false.
- **Internal node**: conditions within the tree.  
- **Leaf**: the value predicted from the conditions. 
- **Tree depth**: The longest path from the root to a leaf.

With the decision tree algorithm in machine learning, the tree can have at most two nodes resulting from it, also known as children.

If a tree only has a depth of 1, we call that a **decision stump**.

<img src="imgs/lingo_tree.png"  width = "55%" alt="404 image">

This tree  and the one in our example above, both have a depth of 2.

Trees do not need to be balanced. (You'll see this shortly)

### Implimentation with Scikit-learn

- There are several machine learning libraries available to use but for this course, we will be using the  Scikit-learn (hereafter, referred to as sklearn) library, which is a popular (41.6k stars on Github) Machine Learning library for Python.


- We generally import a particular ML algorithm using the following syntax:
> `from sklearn.module import algorithm`
- The decision tree classification algorithm (`DecisionTreeClassifier`) sits within the `tree` module.
- (Note there is also a Decision Tree Regression algorithm in this module which we'll come to later...)
- Let's import the classifier using the following code:

from sklearn.tree import DecisionTreeClassifier

- We can begin creating a model by instantiating an instance of the algorithm class.

- Here we are naming our decision tree model `model`:

model = DecisionTreeClassifier()
model

- At this point we just have the framework of a model
- We can't do anything with our algorithm yet, because it hasn't seen any data! 
- We need to give our algorithm some data to learn/train/fit a model
- Let's use the election data we imported previously to make a decision tree model

df.head()

Remember that we split this data into our features table and target values or our `x` and  `y` object. 

X.head()

y.head()

- We can now use the `.fit()` method to train our model using the `X` and `y` data. 
- When we call fit on our model object, the actual learning happens. 

model.fit(X, y)

- Now we've used data to learn a model, let's take a look at the model we made!
- The code below prints out our model structure for us (like the tree we made ourselves earlier)

*Note: This `display_tree`, function was adapted from the `graphviz` library with some amendments to make the trees easier to understand. You can find the code in the `script` file on Canvas.* 

sys.path.append('code/')
from display_tree import display_tree
display_tree(X.columns, model, "imgs/decision_tree")

- We can better visualize what's going on by actually plotting our data and the model's  **decision boundaries**.

*Note: This `plot_classifier` made by Mike Gelbart, function is available for installation [here](https://github.com/mgelbart/plot-classifier) or using:*
>`pip install git+git://github.com/mgelbart/plot-classifier.git`

or with conda

> `conda install git` <br>
> `conda install pip`<br>
> `pip install git+git://github.com/mgelbart/plot-classifier.git`

from plot_classifier import plot_classifier

plot_classifier(X, y, model, ticks=True)
plt.xticks(fontsize= 20);
plt.yticks(fontsize= 20);
plt.xlabel('lon', fontsize=20);
plt.ylabel('lat', fontsize=20);  

- In this plot the shaded regions show what our model predicts for different feature values
- The scatter points are our actual 20 observations
- From the above plot, we can see that our model is classifying all our observations correctly 
- But there's an easier way to find out how our model is doing

We can predict the target of examples by calling `.predict()` on the classifier object.

Let’s see what it predicts for a single randomly new observation first:

new_ex = [-87.4, 59]
new_example = pd.DataFrame(data= [new_ex], columns = ["lon", "lat"])
new_example

model.predict(new_example)

we get a prediction of `red` for this example!

We can also predict on our whole feature table - Here, we are predicting on all of X.

model.predict(X)

pd.DataFrame({'true_values' : y.to_numpy(), 'predicted' : model.predict(X)})

Or if we just want to know how many we got right, in the classification setting, we can use  `score()` which gives the accuracy of the model, i.e., the proportion of correctly predicted examples.

Sometimes we will also see people reporting **error**, which is usually 1 - accuracy. 

Our model has an accurary of 100% (or 0% error)!

model.score(X,y)

### How does `.predict()` work?

For us to see how our algorithm predicts for each example, all we have to do is return to our Decision Tree. 

display_tree(X.columns, model, "imgs/decision_tree")

Let's use our `new_example` object for this example.

new_example

First we start at the root. <br>
Is `lon` < -102.165? False, so we go down the right branch. <br>
Is `lon` < -81.529? True , so we go down the left branch . <br>
We arrive at another node. Is `lat` < 41.113? True , so we go down the left branch and arrive at a prediction of `red`! <br>

Let's check this using predict again. 

model.predict(new_example)

Nice!

### How does `.fit()` work?

Or "How does do Decision Trees decide what values to split on?"


We will not go into detail here, but there the important thing to note here is: 

- We evaluate the utility of a split using a mathematical formula (see [here](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)) where we minimize impurity at each question/node (gives you the least hetegeneous splits)

- Common criteria to minimize impurity
    - Gini Index
    - Information gain
    - Cross entropy 


## Let's Practice! 

Using the data `candybars.csv` from the datafolder to aswer the following questions:

1. how many features are there?
2. How many observations are there? 
3. What would be suitable target for a classification problem?

candy_df = pd.read_csv('data/candybars.csv')
candy_df.head()

candy_df.shape

***Answer as either `fit`  or `predict`***
1. Is called first (before the other one).
2. Only takes X as an argument.
3. In scikit-learn, we can ignore its output.In scikit-learn, we can ignore its output.

***Quick Questions***
1. What is the top node in a decision tree called? 
2. What Python structure/syntax are the nodes in a decision tree similar to? 

## Parameters and Hyperparameters

- ***Parameters***:  Derived during training
- ***Hyperparameters***: Adjustable parameters that can be set before training. 



### Parameters 

When you call `fit` (the training stage of building your model), **parameters** get set, like the split variables and split thresholds. 


<img src='imgs/parameters.png'  width = "30%" alt="404 image" />
 

### Hyperparameters

But even before calling `fit` on a specific data set, we can set some some "knobs" which that control the learning which are called **hyperparameters**. 

In scikit-learn, hyperparameters are set in the constructor.

`max_depth`is a hyperparameter (of many) that lets us decide and set how "deep" we allow our tree to grow.

Let's practice by making a decision stump (A tree with a depth of 1). Our last model was made where we set the depth to "unlimited" so we need to initial a new model and train a new where where we set the `max_depth` hyperparameter. 

model_1 = DecisionTreeClassifier(max_depth=1).fit(X, y)
model_1.fit(X, y)

Let's see what the tree looks like now.  

display_tree(X.columns, model_1, "imgs/decision_stump")

We see that it's a depth of one and split on `lon` at -102.165. 

- The hyperparameter `max_depth`  is being set by us at 1.
- The parameter `lon` is set by the algorithm at -102.165. 

We can see the decision boundary at `lon`= -102.165 with the vertical line in the plot below. 

plot_classifier(X, y, model_1, ticks=True)
plt.xticks(fontsize= 20);
plt.yticks(fontsize= 20);
plt.xlabel('lon', fontsize=20);
plt.ylabel('lat', fontsize=20);  

- Looking  at the score of this model, we get an accuracy of 75%.

model_1.score(X, y)

Let's try growing a more complex tree model and now set `max_depth = 3`

model_3 = DecisionTreeClassifier(max_depth=3).fit(X, y)

display_tree(X.columns, model_3, "imgs/dt_2")

This has 4 splits in the tree so we expect 4 decision boundaries (2 on `lon` and 2 on `lat`). 

plot_classifier(X, y, model_3, ticks=True)
plt.xticks(fontsize= 20);
plt.yticks(fontsize= 20);
plt.xlabel('lon', fontsize=20);
plt.ylabel('lat', fontsize=20);  

- Looking at the score of this model now get an accuracy of 90%! 

model_3.score(X, y)

Let's do one more and set `max_depth = 5`.

model_5 = DecisionTreeClassifier(max_depth=5).fit(X, y)

display_tree(X.columns, model_5, "imgs/dt_5")

This has 7 splits in the tree. How many decision boundaries should there be?

plot_classifier(X, y, model_5, ticks=True)
plt.xticks(fontsize= 20);
plt.yticks(fontsize= 20);
plt.xlabel('lon', fontsize=20);
plt.ylabel('lat', fontsize=20);  

And if we check the score of this model we get 100% accuracy! 

model_5.score(X, y)

We see here that as `max_depth` increases, the accuracy of the training data does as well.

Doing this isn’t always the best idea and we’ll explain this a little bit later on.

- This is just one of many other hyperparameters for decision trees that you can explore -> link <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html" target="_blank">here</a> There are many other hyperparameters for decision trees that you can explore at the link <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html" target="_blank">here</a>.

To summarize this section:
- **parameters** are automatically learned by an algorithm during training
- **hyperparameters** are specified before training

## Decision Tree Regressor 

We saw that we can use decision trees for classification problems but we can also use this decision tree algorithm for regression problems.  

Instead of using Gini impurity (which we briefly mentioned this above), we can use <a href="https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation" target="_blank">some other criteria</a> for splitting. 

(A common one is mean squared error (MSE) which we will discuss shortly)

`scikit-learn` supports regression using decision trees with `DecisionTreeRegressor()` and the `.fit()` and `.predict()` paradigm that is similar to classification.

Let's do an example using the `kc_house_data` we saw in example 1. 

df = pd.read_csv("data/kc_house_data.csv")
df = df.drop(columns=["id", "date"])
df.head()

X = df.drop(columns=["price"])
X.head()

y = df["price"]
y.head()

We can see that instead of predicting a categorical column like we did with `vote` before, our target column is now numeric. 

Instead of importing `DecisionTreeClassifier`, we import `DecisionTreeRegressor`.

We follow the same steps as before and can even set hyperparameters as we did in classification. 

Here when we build our model, we are specifying a `max_depth` of 4. 

This means our decision tree is going to be constrained to a depth of 4.

from sklearn.tree import DecisionTreeRegressor

depth = 3
reg_model = DecisionTreeRegressor(max_depth=depth)
reg_model.fit(X, y)

Let's look at the tree it produces Our leaves used to contain a categorical value for prediction, but this time we see our leaves are predicting numerical values.

display_tree(X.columns, reg_model, "imgs/dt_reg")

Let's see what our model predicts for a single example. 

X.loc[[0]]

reg_model.predict(X.loc[[0]])

Our model predicts a housing price of $269848.39 

Should we see what the true value is? 

y.loc[[0]]

The true value is $221900.0, but how well did it score? 

With regression problems we can't use accuracy for a scoring method so instead when we use `.score()`  it returns somethings called an <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score" target="_blank"> 𝑅2 </a>.

reg_model.score(X,y)

The maximum 𝑅2 is 1 for perfect predictions. 

It can be negative which is very bad (worse than DummyRegressor). 

## Let's Practice 

Using the data `candybars.csv` from the datafolder for the following:
1. Define two objects named `X` and `y` which contain the features and target column respectively.
2. Using sklearn, create 3 different decision tree classifiers using 3 different `min_samples_split` values based on this data.
3. What is the accuracy of each classifier on the training data?
4. a) Which `min_samples_split` value would you choose to predict this data? <br>
   b) Would you choose the same `min_samples_split` value to predict new data?
5. Do you think most of the computational effort for a decision tree takes place in the `.fit()` stage or `.predict()` stage?

candy_df = pd.read_csv('data/candybars.csv')
candy_df.head()

## What We've Learned Today<a id="9"></a>

- What is machine learning (supervised/unsupervised, classification/regression)
- Machine learning terminology
- What is the decision tree algorithm and how does it work
- The scikit-learn library
- Parameters and hyperparameters

