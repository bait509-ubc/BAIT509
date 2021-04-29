<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Lecture-Learning-Objectives" data-toc-modified-id="Lecture-Learning-Objectives-0"><span class="toc-item-num">0&nbsp;&nbsp;</span>Lecture Learning Objectives</a></span></li><li><span><a href="#Five-Minute-Recap/-Lightning-Questions" data-toc-modified-id="Five-Minute-Recap/-Lightning-Questions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Five Minute Recap/ Lightning Questions</a></span><ul class="toc-item"><li><span><a href="#Some-lingering-questions" data-toc-modified-id="Some-lingering-questions-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Some lingering questions</a></span></li></ul></li><li><span><a href="#Generalization" data-toc-modified-id="Generalization-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Generalization</a></span><ul class="toc-item"><li><span><a href="#Visualizing-model-complexity-using-decision-boundaries" data-toc-modified-id="Visualizing-model-complexity-using-decision-boundaries-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Visualizing model complexity using decision boundaries</a></span></li><li><span><a href="#The-Fundamental-goal-of-machine-learning" data-toc-modified-id="The-Fundamental-goal-of-machine-learning-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>The Fundamental goal of machine learning</a></span></li><li><span><a href="#Generalizing-to-unseen-data" data-toc-modified-id="Generalizing-to-unseen-data-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Generalizing to unseen data</a></span></li><li><span><a href="#Training-score-versus-Generalization-score-(or-Error)" data-toc-modified-id="Training-score-versus-Generalization-score-(or-Error)-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Training score versus Generalization score (or Error)</a></span></li></ul></li><li><span><a href="#Splitting" data-toc-modified-id="Splitting-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Splitting</a></span><ul class="toc-item"><li><span><a href="#Simple-train-and-test-split" data-toc-modified-id="Simple-train-and-test-split-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Simple train and test split</a></span></li><li><span><a href="#How-do-we-do-this?" data-toc-modified-id="How-do-we-do-this?-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>How do we do this?</a></span><ul class="toc-item"><li><span><a href="#First-way" data-toc-modified-id="First-way-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>First way</a></span></li><li><span><a href="#Second-way" data-toc-modified-id="Second-way-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Second way</a></span></li></ul></li><li><span><a href="#Applications-with-Splitting" data-toc-modified-id="Applications-with-Splitting-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Applications with Splitting</a></span></li><li><span><a href="#Parameters-in-.train_test_split()" data-toc-modified-id="Parameters-in-.train_test_split()-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Parameters in <code>.train_test_split()</code></a></span></li></ul></li><li><span><a href="#Train/validation/test-split" data-toc-modified-id="Train/validation/test-split-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Train/validation/test split</a></span></li><li><span><a href="#&quot;Deployment&quot;-data" data-toc-modified-id="&quot;Deployment&quot;-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>"Deployment" data</a></span></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#Cross-validation" data-toc-modified-id="Cross-validation-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Cross-validation</a></span></li><li><span><a href="#Cross-validation-using-sk-learn" data-toc-modified-id="Cross-validation-using-sk-learn-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Cross-validation using <code>sk-learn</code></a></span><ul class="toc-item"><li><span><a href="#cross_val_score" data-toc-modified-id="cross_val_score-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span><code>cross_val_score</code></a></span></li><li><span><a href="#cross_validate" data-toc-modified-id="cross_validate-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span><code>cross_validate</code></a></span></li></ul></li><li><span><a href="#Our-typical-supervised-learning-set-up-is-as-follows:" data-toc-modified-id="Our-typical-supervised-learning-set-up-is-as-follows:-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Our typical supervised learning set up is as follows:</a></span></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#Overfitting-and-Underfitting" data-toc-modified-id="Overfitting-and-Underfitting-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Overfitting and Underfitting</a></span><ul class="toc-item"><li><span><a href="#Types-of-scores" data-toc-modified-id="Types-of-scores-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>Types of scores</a></span></li><li><span><a href="#Overfitting" data-toc-modified-id="Overfitting-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>Overfitting</a></span></li><li><span><a href="#Underfitting" data-toc-modified-id="Underfitting-11.3"><span class="toc-item-num">11.3&nbsp;&nbsp;</span>Underfitting</a></span></li><li><span><a href="#How-can-we-figure-this-out?" data-toc-modified-id="How-can-we-figure-this-out?-11.4"><span class="toc-item-num">11.4&nbsp;&nbsp;</span>How can we figure this out?</a></span></li></ul></li><li><span><a href="#The-&quot;Fundamental-Tradeoff&quot;-of-Supervised-Learning" data-toc-modified-id="The-&quot;Fundamental-Tradeoff&quot;-of-Supervised-Learning-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>The "Fundamental Tradeoff" of Supervised Learning</a></span><ul class="toc-item"><li><span><a href="#How-to-pick-a-model-that-would-generalize-better?" data-toc-modified-id="How-to-pick-a-model-that-would-generalize-better?-12.1"><span class="toc-item-num">12.1&nbsp;&nbsp;</span>How to pick a model that would generalize better?</a></span></li></ul></li><li><span><a href="#The-Golden-Rule" data-toc-modified-id="The-Golden-Rule-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>The Golden Rule</a></span></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#What-We've-Learned-Today" data-toc-modified-id="What-We've-Learned-Today-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>What We've Learned Today<a id="9"></a></a></span></li></ul></div>

# Lecture 2 - Splitting and Cross-validation

*Hayley Boyce, Wednesday, April 21st 2021*

#import sys
#!{sys.executable} -m pip install numpy pandas sklearn graphviz

# Importing our libraries

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.model_selection import train_test_split
import sys
sys.path.append('code/')
from display_tree import display_tree

from plot_classifier import plot_classifier
import matplotlib.pyplot as plt

## Lecture Learning Objectives 

- Explain the concept of generalization.
- Split a dataset into train and test sets using `train_test_split` function.
- Explain the difference between train, validation, test, and "deployment" data.
- Identify the difference between training error, validation error, and test error.
- Explain cross-validation and use `cross_val_score()` and `cross_validate()` to calculate cross-validation error.
- Explain overfitting, underfitting, and the fundamental tradeoff.
- State the golden rule and identify the scenarios when it's violated.

## Five Minute Recap/ Lightning Questions 

- What is an example of machine learning? 
- Which type of machine learning does not have labels?
- What is an example of a Regression problem? 
- In a dataframe, what is an observation? 
- What is the first node of a Decision Tree called?
- Where/who determines the parameter values?
- What library will we be using for machine learning? 

# load data
voting_df = pd.read_csv('data/cities_USA.csv',index_col=0)  

voting_df.head()

# feature table
X = voting_df.drop(columns='vote') 

# the target variable
y = voting_df[['vote']] 

# initiate model framework with a decision tree of max_depth 2
model = DecisionTreeClassifier(max_depth=2) 

# training the model
model.fit(X, y)  

# new longitude and latitude coordinates

lon_lat_coords = np.array([[-75, 56]])

# predict on coordinates
prediction = model.predict(lon_lat_coords)  


print('With a latitude of', lon_lat_coords[0,1],
      'and a longitude of', lon_lat_coords[0,0], 
      'the model predicted', prediction[0])

### Some lingering questions

1. How do we choose a value of `max_depth` (or other hyperparameters)?
2. Why not just use large `max_depth` for every supervised learning problem and get super high accuracy?
3. Is model performance on the training data a good indication of how it will perform on new data?


We will be answering these questions in this lecture. 

## Generalization

### Visualizing model complexity using decision boundaries

In the last lecture, we saw that we could visualize the splitting of decision trees using these boundaries. 

So let's look back at our cities dataset.

voting_df = pd.read_csv('data/cities_USA.csv',index_col=0)  
voting_df.head()

We then create our feature table and our target objects (`X` and `y`) 

X = voting_df.drop(columns=["vote"])
X.head()

y = voting_df["vote"]
y.head()

Let's build our model now. 

We're going to build a decision tree classifier and set the `max_depth` hyperparameter to 1 to create a decision stump. 



depth = 1
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X, y);
model.score(X, y)

display_tree(X.columns, model, "imgs/dt_md_1")

In this decision tree, there is only 1 split.  

model.score(X, y)

When we score it on data that it’s already seen, we get an accuracy of 74.75%.

Plotting the values, we see the decision boundary separating the 2 regions where lat=37.682.

plot_classifier(X, y, model, ticks=True);
plt.xticks(fontsize= 12);
plt.yticks(fontsize= 12);
plt.xlabel('lon', fontsize=14);
plt.ylabel('lat', fontsize=14);
plt.title("Decision tree with depth = %d" % (depth), fontsize=18);

Ok, now let's see what happens to our decision boundaries when we change our maximum tree depth to 2. 

depth = 2
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X, y);
model.score(X, y)

display_tree(X.columns, model, "imgs/dt_md_2")

The decision boundaries are created by asking 3 questions (only possible to ask 2 questions per observation) and we can see 3 splits now. 

Our score here has increased from 74.75% to 82.75%.

When we graph it, we can now see 3 boundaries. 2 where `lon` equals -114.063 and -96.061 and another where `lat` equals 37.682.  

plot_classifier(X, y, model, ticks=True);
plt.xticks(fontsize= 12);
plt.yticks(fontsize= 12);
plt.xlabel('lon', fontsize=14);
plt.ylabel('lat', fontsize=14);
plt.title("Decision tree with depth = %d" % (depth), fontsize=18);

Now let’s continue on making a new model with a hyperparameter `max_depth` equal to 4. 

depth = 4
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X, y);
model.score(X, y)

display_tree(X.columns, model, "imgs/dt_md_4")

Our score now has shot up to 90.25% and we have way more splits now.

We can see our boundaries are getting more complex. 

plot_classifier(X, y, model, ticks=True);
plt.xticks(fontsize= 12);
plt.yticks(fontsize= 12);
plt.xlabel('lon', fontsize=14);
plt.ylabel('lat', fontsize=14);
plt.title('Decision tree with depth = %d' % (depth), fontsize=18);

What happens if we give the model an unlimited `max_depth`?

model = DecisionTreeClassifier()
model.fit(X, y);
model.score(X, y)

display_tree(X.columns, model, "imgs/dt_md_unlimited")

plot_classifier(X, y, model, ticks=True);
plt.xticks(fontsize= 12);
plt.yticks(fontsize= 12);
plt.xlabel('lon', fontsize=14);
plt.ylabel('lat', fontsize=14);
plt.title('Decision tree with unlimited max depth', fontsize=18);

model.get_depth()

Our score is now 100%.  We can see that with this model we are perfectly fitting every observation.

The model is now more specific and sensitive to the training data. 

For our decision tree model, we see that score increases as we increase `max_depth`.
Since we are creating a more complex tree (higher `max_depth`) we can fit all the peculiarities of our data to eventually get 100% accuracy.

Do you think that's going to be helpful for us to have a model with a perfect score on all the data?

### The Fundamental goal of machine learning

Goal: **to generalize beyond what we see in the training examples**. <br>
We are only given a sample of the data and do not have the full distribution. <br>
Using the training data, we want to come up with a reasonable model that will perform well on some unseen examples. <br>

At the end of the day, we want to deploy models that make reasonable predictions on unseen data


<img src='imgs/generalization-train.png'  width = "30%" alt="404 image" />

### Generalizing to unseen data

<center><img src="imgs/generalization-predict.png" width = "100%" alt="404 image" /></center>

Would you expect her to be able to correctly identify each image?

The point here is that we want this learning to be able to generalize beyond what it sees here and be able to predict and predict labels for the new examples.

These new examples should be representative of the training data.

### Training score versus Generalization score (or Error)

- Although we can get 100% accuracy on our model, do we trust it? 

What if we used the model we saw that gives 100% accuracy, Would you expect this model to perform equally well on unseen examples? Probably not. 

Given a model in machine learning, people usually talk about two kinds of accuracies (scores):

1. Accuracy on the training data
    
2. **Accuracy on the entire distribution of data**


We are really interested in the score on the entire distribution because at the end of the day we want our model to perform well on unseen examples. 

But the problem is that we do not have access to the distribution and only the limited training data that is given to us. 

So, what do we do? 

## Splitting

We can approximate generalization accuracy by splitting our data!


<img src="imgs/splitted.png"  width = "60%" alt="404 image" />


- Keep a randomly selected portion of our data aside that we call that the testing data. 

- fit (train) a model on the training portion only.
- score (assess) the trained model on this set-aside **Testing** data to get a sense of how well the model would be able to generalize.


###  Simple train and test split


<center><img src="imgs/train-test-split.png"  width = "100%" alt="404 image" /></center>




- First, the data needs to be shuffled.
- Then, we split the rows of the data into 2 sections -> **train** and **test**. 

- The lock and key icon on the test set symbolizes that we don't want to touch the test data until the very end (more on this soon).


### How do we do this? 

In our trusty Scikit Learn package, we have a function for that! 
- `train_test_split`

Let's try it out using a similar yet slightly different dataset. Here we still have `latitude` and `longitude` coordinates but this time our target variable is if the city with these coordinates lies in Canada or the USA. 

#### First way

cities_df = pd.read_csv('data/canada_usa_cities.csv')  

cities_df.head()

X = cities_df.drop(columns=["country"])
X.head()

y = cities_df["country"]
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123) 

# Split the dataset into 80% train and 20% test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train.head()

X_test.head()

y_train.head()

y_test.head()

# Print shapes (ignore the code here)
shape_dict = {"Data portion": ["X", "y", "X_train", "y_train", "X_test", "y_test"],
    "Shape": [X.shape, y.shape,
              X_train.shape, y_train.shape,
              X_test.shape, y_test.shape]}

shape_df = pd.DataFrame(shape_dict)
shape_df

#### Second way

Instead of splitting our `X` and `y` objects. We can split the whole dataset first into train and test splits. 

The earlier to split the data the better!

cities_df

train_df, test_df = train_test_split(cities_df, test_size = 0.2, random_state = 123)

X_train = train_df.drop(columns=["country"])
y_train = train_df["country"]

X_test = test_df.drop(columns=["country"])
y_test = test_df["country"]

train_df.head()

Sometimes we may want to keep the target in the train split for EDA or for visualization.

import altair as alt
chart_votes = alt.Chart(train_df).mark_circle(size=20, opacity=0.6).encode(
    alt.X('longitude:Q', scale=alt.Scale(domain=[-140, -50])),
    alt.Y('latitude:Q', scale=alt.Scale(domain=[25, 60])),
    alt.Color('country:N', scale=alt.Scale(domain=['Canada', 'USA'],
                                           range=['red', 'blue'])))
chart_votes

### Applications with Splitting

Now let's compare the decision boundaries using our data. 

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
display_tree(X_train.columns, model)

Let's see how on model scores on the training data and the test data.

print("Train score: " + str(round(model.score(X_train, y_train), 2)))
print("Test score: " + str(round(model.score(X_test, y_test), 2)))

For this tree, the training score is 1.0 and the test score is only 0.74.  

The model does not perform quite as well on data that it has not seen. 

Let's look at the training and testing data with the decision boundaries made by the model. 

model.fit(X_train, y_train);
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1);
plt.title("Decision tree model on the training data");
plot_classifier(X_train, y_train, model, ticks=True, ax=plt.gca(), lims=(-140,-50,25,60))
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.subplot(1, 2, 2);
plt.title("Decision tree model on the test data")
plot_classifier(X_test, y_test, model, ticks=True, ax=plt.gca(), lims=(-140,-50,25,60))
plt.xlabel("longitude")
plt.ylabel("latitude");

On the left and the right, we have the same boundaries But different data being shown.

The model is getting 100 percent accuracy on the training and for that to happen, the model ends up being extremely specific.

The model got over complicated on the training data and this doesn’t generalize to the test data well.

In the plot on the right, we can see some red triangles in the blue area. That is the model making mistakes which explains the 71% accuracy.

### Parameters in `.train_test_split()`

- `test_size` - test split size (0-1)
- `train_size` - train split size (0-1) (only need to specify one if these
- `random_state` - randomizes the split

train_df, test_df = train_test_split(cities_df, test_size = 0.2, random_state = 123)

train_df, test_df = train_test_split(cities_df, train_size = 0.8, random_state = 123)

train_df, test_df = train_test_split(cities_df, test_size = 0.2, train_size = 0.8, random_state = 123)

There is no hard and fast rule on the split sizes should we use. Some common splits are 90/10, 80/20, 70/30 (training/test).

In the above example, we used an 80/20 split.

But there is a trade-off:
- More training -> More information for our model. 
- More test -> Better assessment of our model.

Now let's look at the random_state argument:

The random_state argument controls this shuffling and without this argument set, each time we split our data, it will be split in a different way.

train_df, test_df = train_test_split(cities_df, test_size = 0.2)
train_df.head()

We set this to add a component of reproducibility to our code and if we set it with a `random_state` when we run our code again it will produce the same result. 

train_df_rs5, test_df_rs5 = train_test_split(cities_df, test_size = 0.2, random_state = 5)
train_df_rs5.head()

train_df_rs7, test_df_rs7 = train_test_split(cities_df, test_size = 0.2, random_state = 7)
train_df_rs7.head()

## Train/validation/test split

Remember hyperparameters?     
What dataset do we use if we want to find which hyperparameters produce the best generalized model (also called **hyperparameter optimization**)?      
    
It's a good idea to have separate data for tuning the hyperparameters of a model that is not the test set.

Enter, the ***validation*** set. 

So we actually want to split our dataset into 3 splits: train, validation, and test.

<img src='imgs/train-valid-test-split.png' width="100%" />


***Note: There isn't a good consensus on the terminology of what is validation and what is test data.***

We use: 
- **validation data**: data where we have access to the target values, but unlike the training data, we only use this for hyperparameter tuning and model assessment; we don't pass these into `fit`.  

- **test data**: Data where we have access to the target values, but in this case, unlike training and validation data, we neither use it in training nor hyperparameter optimization and only use it **once** to evaluate the performance of the best performing model on the validation set. We lock it in a "vault" until we're ready to evaluate. 

## "Deployment" data

What's the point of making models? 
 > We want to predict something which we do not know the answer to, so we do not have the target values and we only have the features.

After we build and finalize a model, we deploy it, and then the model is used with data in the wild.

We will use **deployment data** to refer to data, where we do not have access to the target values.

Deployment score is the thing we really care about.

We use validation and test scores as proxies for the deployment score, and we hope they are similar.

So, if our model does well on the validation and test data, we hope it will do well on deployment data.

## Let's Practice 

1. When is the most optimal time to split our data? 
2. Why do we split our data?
3. Fill in the table below:

| datasets   | `.fit()` | `.score()` | `.predict()` |
|------------|:--------:|:----------:|:------------:|
| Train      |    ✔️     |   ✔️      |   ✔️          |
| Validation |          |   ✔️       |     ✔️       |
| Test       |          |    Once    |   Once       |
| Deployment |          |            |      ✔️      |


## Cross-validation


<img src='imgs/train-valid-test-split.png' width="100%" />

Problems with having a single train/test split:

- Only using a portion of your full data set for training/testing (data is our most precious resource!!!)
- If your dataset is small you might end up with a tiny training/testing set
- Might be unlucky with your splits such that they don't well represent your data (shuffling data, as is done in `train_test_split()`, is not immune to being unlucky!)


There must be a better way! 

<center><img src="https://media.giphy.com/media/i4gLlAUz2IVIk/giphy.gif"  width = "50%" alt="404 image" /></center>

There is! The answer to our problem is called.....

**Cross-validation** or **𝑘-fold cross-validation**.


<img src='imgs/cross-validation.png' width="100%">



- We still have the test set here at the bottom locked away that we will not touch until the end.
- But, we split the training data into $k$ folds ($k>2$, often $k=10$). In the graphic above $k=4$.
- Each "fold" gets a turn at being the validation set.
- Each round will produce a score so after 𝑘-fold cross-validation, it will produce 𝑘 scores. We usually average over the 𝑘 results.
- Note that cross-validation doesn't shuffle the data; it's done in `train_test_split`.
- We can get now a more “robust” score on unseen data since we can see the variation in the scores across folds.  


***Question: What's the disadvantage then? Why not use larger values for $k$?***

> ....


## Cross-validation using `sk-learn`

There are 2 ways we can do cross-validation with `sk-learn`:
- `.cross_val_score()`
- `.cross_validate()`

Before doing cross-validation we still need to split our data into our training set and our test set and separate the features from the targets. 
So using our `X` and `y` from our Canadian/United States cities data we split it into train/test splits. 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

### `cross_val_score`


from sklearn.model_selection import cross_val_score

model = DecisionTreeClassifier(max_depth=4)
cv_score = cross_val_score(model, X_train, y_train, cv=5)
cv_score

Once, we've imported `cross_val_score` we can make our model and call our model, the feature object and target object as arguments. 

- `cv` determines the cross-validation splitting strategy or how many "folds" there are.

-For each fold, the model is fitted on the training portion and scores on the validation portion.

- The output of `cross_val_score()` is the validation score for each fold. 

cv_score.mean()

cv_score = cross_val_score(model, X_train, y_train, cv=10)
cv_score

cv_score.mean()

### `cross_validate`

- Similar to `cross_val_score` but more informative.
- Lets us access training ***and*** validation scores using the parameter `return_train_score`.
- Note: in the dictionary output `test_score` and `test_time` refers to *validation score* and *validation time*

from sklearn.model_selection import cross_validate

scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
scores

pd.DataFrame(scores)

scores_df = pd.DataFrame(cross_validate(model, X_train, y_train, cv=10, return_train_score=True))
scores_df

scores_df.mean()

scores_df.std()

## Our typical supervised learning set up is as follows: 

1. Given training data with `X` and `y`.
2. We split our data into `X_train, y_train, X_test, y_test`.
3. Hyperparameter optimization using cross-validation on `X_train` and `y_train`. 
4. We assess the best model using  `X_test` and `y_test`.
5. The **test score** tells us how well our model generalizes.
6. If the **test score** is reasonable, we deploy the model.

## Let's Practice 

1. We carry out cross-validation to avoid reusing the same validation set again and again. Let’s say you do 10-fold cross-validation on 1000 examples. For each fold, how many examples do you train on?
2. With 10-fold cross-validation, you split 1000 examples into 10-folds. For each fold, when you are done, you add up the accuracies from each fold and divide by what?

True/False:
- 𝑘-fold cross-validation calls fit 𝑘 times and predict 𝑘 times.


## Overfitting and Underfitting

### Types of scores 
We've talked about the different types of splits, now we are going to talk about their scores. 

- **Training score**: The score that our model gets on the same data that it was trained on. (seen data - training data) 
- **Validation score**: The mean validation score from cross-validation).
- **Test score**: This is the score from the data that we locked away. 

### Overfitting

- Overfitting occurs when our model is overly specified to the particular training data and often leads to bad results.
- Training score is high but the validation score is much lower.  
- The gap between train and validation scores is large.
- It's usually common to have a bit of overfitting (only a bit!) 
- This produces more severe results when the training data is minimal or when the model’s complexity is high.

model = DecisionTreeClassifier()
scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)

pd.DataFrame(scores)

print("Train score: " + str(round(scores["train_score"].mean(), 2)))
print("Validation score: " + str(round(scores["test_score"].mean(), 2)))

model.fit(X_train, y_train);
plot_classifier(X_train, y_train, model);
plt.title("Decision tree with no max_depth");

### Underfitting 

- Underfitting is somewhat the opposite of overfitting in the sense that it occurs when the model is not complex enough. 
- Underfitting is when our model is too simple (`DecisionTreeClassifier` with max_depth=1). 
- The model doesn't capture the patterns in the training data and the training score is not that high.
- Both train and validation scores are low and the gap between train and validation scores is low as well.

model = DecisionTreeClassifier(max_depth=1)

scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
print("Train score: " + str(round(scores["train_score"].mean(), 2)))
print("Validation score: " + str(round(scores["test_score"].mean(), 2)))

Standard question to ask ourselves: 
***Which of these scenarios am I in?***

### How can we figure this out?

- If the training and validation scores are very far apart → more likely **overfitting**.     
    - Try decreasing model complexity.

- If the training and validation scores are very close together → more likely **underfitting**.  
    - Try increasing model complexity.

## The "Fundamental Tradeoff" of Supervised Learning

As model complexity increases:

$\text{Training score}$  ↑ and ($\text{Training score} − \text{Validation score}$) tend to also  ↑


If our model is too simple (underfitting) then we won't really learn any "specific patterns" of the training set. 

**BUT** 

If our model is too complex then we will learn unreliable patterns that get every single training example correct, and there will be a large gap between training error and validation error.

The trade-off is there is tension between these two concepts. 

When we underfit less, we overfit more. 

How do we know how much overfitting is too much and how much is not enough? 

### How to pick a model that would generalize better?

results_dict = {"depth": list(), "mean_train_score": list(), "mean_cv_score": list()}

for depth in range(1,20):
    model = DecisionTreeClassifier(max_depth=depth)
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
    results_dict["depth"].append(depth)
    results_dict["mean_cv_score"].append(scores["test_score"].mean())
    results_dict["mean_train_score"].append(scores["train_score"].mean())

results_df = pd.DataFrame(results_dict)
results_df

source = results_df.melt(id_vars=['depth'] , 
                              value_vars=['mean_train_score', 'mean_cv_score'], 
                              var_name='score_type', value_name='accuracy')
chart1 = alt.Chart(source).mark_line().encode(
    alt.X('depth:Q', axis=alt.Axis(title="Tree Depth")),
    alt.Y('accuracy:Q'),
    alt.Color('score_type:N', scale=alt.Scale(domain=['mean_train_score', 'mean_cv_score'],
                                           range=['teal', 'gold'])))
chart1

chart1.encode(alt.Y('accuracy:Q', scale=alt.Scale(zero=False)))

- As we increase our depth (increase our complexity) our training data increases. 
- As we increase our depth, we overfit more, and the gap between the train score and validation score also increases... except  ... 

- There is a spot where the gap between the validation score and test score is the smallest while still producing a decent validation score.
- In the plot, this would be around `max_depth` is 5. 
- Commonly, we look at the cross-validation score and pick the hyperparameter with the highest cross-validation score. 

results_df.sort_values('mean_cv_score', ascending=False)

Now that we know the best value to use for `max_depth`, we can build a new classifier setting `max_depth=5`, train it and now (only now) do we score our model on the test set.

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train);
print("Score on test set: " + str(round(model.score(X_test, y_test), 2)))

- Is the test error comparable with the cross-validation error?
- Do we feel confident that this model would give a similar performance when deployed? 

## The Golden Rule 

- Even though we care the most about test error **THE TEST DATA CANNOT INFLUENCE THE TRAINING PHASE IN ANY WAY**. 
- We have to be very careful not to violate it while developing our ML pipeline. 
- Why? When this happens, the test data influences our training and the test data is no longer unseen data and so the test score will be too optimistic.
- Even experts end up breaking it sometimes which leads to misleading results and lack of generalization on the real data. 
    - https://www.theregister.com/2019/07/03/nature_study_earthquakes/
    - https://www.technologyreview.com/2015/06/04/72951/why-and-how-baidu-cheated-an-artificial-intelligence-test/
    
How do we avoid this? 

The most important thing is when splitting the data, we lock away the test set and keep it separate from the training data.

Forget it exists temporarily - kinda like forgetting where you put your passport until you need to travel. 

The workflow we generally follow is:

- **Splitting**: Before doing anything, split the data `X` and `y` into `X_train`, `X_test`, `y_train`, `y_test` or `train_df` and `test_df` using `train_test_split`.  
- **Select the best model using cross-validation**: Use `cross_validate` with `return_train_score = True` so that we can get access to training scores in each fold. (If we want to plot train vs validation error plots, for instance.) 
- **Scoring on test data**: Finally, score on the test data with the chosen hyperparameters to examine the generalization performance.

## Let's Practice

Overfitting or Underfitting
1. If our train accuracy is much higher than our test accuracy.
2. If our train accuracy and our test accuracy are both low and relatively similar in value.
3. If our model is using a Decision Tree Classifier for a classification problem with no limit on `max_depth`.


True or False 
1. In supervised learning, the training score is always higher than the validation score.
2. The fundamental tradeoff of ML states that as training score goes up, validation score goes down.
3. More "complicated" models are more likely to overfit than "simple" ones.
5. If our training score is extremely high, that means we're overfitting.

**Coding practice**

Below is some starter code that creates your feature table and target column from the data from the `bball.csv` dataset (in the data folder).

bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Define X and y
X = bball_df.loc[:, ['height', 'weight', 'salary']]
y = bball_df['position']

1. Split the dataset into 4 objects: `X_train`, `X_test`, `y_train`, `y_test`. Make the test set 0.2 (or the train set 0.8) and make sure to use `random_state=7`.
2. Build a decision tree model with `max_depth=5`. 
3. Cross-validate using cross_validate() on the objects X_train and y_train specifying the model and making sure to use 10 fold cross-validation and `return_train_score=True`.
4. Convert the scores into a dataframe and save it in an object named scores_df.
5. Calculate the mean value of each column and save this in an object named mean_scores.
6. Is your model overfitting or underfitting? 



## What We've Learned Today<a id="9"></a>

- The concept of generalization.
- How to split a dataset into train and test sets using `train_test_split` function.
- The difference between train, validation, test, and "deployment" data.
- The difference between training error, validation error, and test error.
- Cross-validation and use `cross_val_score()` and `cross_validate()` to calculate cross-validation error.
- Overfitting, underfitting, and the fundamental tradeoff.
- Golden rule and identify the scenarios when it's violated.