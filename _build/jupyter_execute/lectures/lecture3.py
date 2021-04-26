# Lecture 3 - Baseline, k-Nearest Neighbours

*Hayley Boyce, Monday, April 26th, 2021*

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Lecture-Learning-Objectives" data-toc-modified-id="Lecture-Learning-Objectives-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Lecture Learning Objectives</a></span></li><li><span><a href="#Five-Minute-Recap/-Lightning-Questions" data-toc-modified-id="Five-Minute-Recap/-Lightning-Questions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Five Minute Recap/ Lightning Questions</a></span><ul class="toc-item"><li><span><a href="#Some-lingering-questions" data-toc-modified-id="Some-lingering-questions-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Some lingering questions</a></span></li></ul></li><li><span><a href="#Baseline-Models" data-toc-modified-id="Baseline-Models-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Baseline Models</a></span><ul class="toc-item"><li><span><a href="#Dummy-Classifier" data-toc-modified-id="Dummy-Classifier-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Dummy Classifier</a></span></li><li><span><a href="#Dummy-Regressor" data-toc-modified-id="Dummy-Regressor-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Dummy Regressor</a></span></li></ul></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#Analogy-based-models" data-toc-modified-id="Analogy-based-models-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Analogy-based models</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Example:" data-toc-modified-id="Example:-5.0.1"><span class="toc-item-num">5.0.1&nbsp;&nbsp;</span>Example:</a></span></li></ul></li><li><span><a href="#Analogy-based-models-in-real-life" data-toc-modified-id="Analogy-based-models-in-real-life-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Analogy-based models in real life</a></span></li></ul></li><li><span><a href="#Terminology" data-toc-modified-id="Terminology-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Terminology</a></span><ul class="toc-item"><li><span><a href="#Feature-Vectors" data-toc-modified-id="Feature-Vectors-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Feature Vectors</a></span></li></ul></li><li><span><a href="#Distance" data-toc-modified-id="Distance-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Distance</a></span><ul class="toc-item"><li><span><a href="#Euclidean-distance" data-toc-modified-id="Euclidean-distance-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Euclidean distance</a></span><ul class="toc-item"><li><span><a href="#Calculating-Euclidean-distance-&quot;by-hand&quot;" data-toc-modified-id="Calculating-Euclidean-distance-&quot;by-hand&quot;-7.1.1"><span class="toc-item-num">7.1.1&nbsp;&nbsp;</span>Calculating Euclidean distance "by hand"</a></span></li><li><span><a href="#Calculating-Euclidean-distance--with-sklearn" data-toc-modified-id="Calculating-Euclidean-distance--with-sklearn-7.1.2"><span class="toc-item-num">7.1.2&nbsp;&nbsp;</span>Calculating Euclidean distance  with <code>sklearn</code></a></span></li></ul></li></ul></li><li><span><a href="#Finding-the-Nearest-Neighbour" data-toc-modified-id="Finding-the-Nearest-Neighbour-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Finding the Nearest Neighbour</a></span><ul class="toc-item"><li><span><a href="#Nearest-city-to-a-query-point" data-toc-modified-id="Nearest-city-to-a-query-point-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Nearest city to a query point</a></span></li></ul></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#$k$--Nearest-Neighbours-($k$-NNs)-Classifier" data-toc-modified-id="$k$--Nearest-Neighbours-($k$-NNs)-Classifier-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>$k$ -Nearest Neighbours ($k$-NNs) Classifier</a></span></li><li><span><a href="#Choosing-K" data-toc-modified-id="Choosing-K-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Choosing K</a></span><ul class="toc-item"><li><span><a href="#How-to-choose-$K$-(n_neighbors)?" data-toc-modified-id="How-to-choose-$K$-(n_neighbors)?-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>How to choose $K$ (<code>n_neighbors</code>)?</a></span></li></ul></li><li><span><a href="#Curse-of-Dimensionality" data-toc-modified-id="Curse-of-Dimensionality-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Curse of Dimensionality</a></span></li><li><span><a href="#Let's-practice" data-toc-modified-id="Let's-practice-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Let's practice</a></span></li><li><span><a href="#Regression-with-$k$-NN" data-toc-modified-id="Regression-with-$k$-NN-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Regression with $k$-NN</a></span></li><li><span><a href="#Pros-and-Cons-of-𝑘--Nearest-Neighbours" data-toc-modified-id="Pros-and-Cons-of-𝑘--Nearest-Neighbours-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>Pros and Cons of 𝑘 -Nearest Neighbours</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Pros:" data-toc-modified-id="Pros:-15.0.1"><span class="toc-item-num">15.0.1&nbsp;&nbsp;</span>Pros:</a></span></li><li><span><a href="#Cons:" data-toc-modified-id="Cons:-15.0.2"><span class="toc-item-num">15.0.2&nbsp;&nbsp;</span>Cons:</a></span></li></ul></li></ul></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-16"><span class="toc-item-num">16&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#Support-Vector-Machines-(SVMs)-with-RBF-Kernel" data-toc-modified-id="Support-Vector-Machines-(SVMs)-with-RBF-Kernel-17"><span class="toc-item-num">17&nbsp;&nbsp;</span>Support Vector Machines (SVMs) with RBF Kernel</a></span><ul class="toc-item"><li><span><a href="#Hyperparameters-of-SVM" data-toc-modified-id="Hyperparameters-of-SVM-17.1"><span class="toc-item-num">17.1&nbsp;&nbsp;</span>Hyperparameters of SVM</a></span><ul class="toc-item"><li><span><a href="#gamma-and-the-fundamental-trade-off" data-toc-modified-id="gamma-and-the-fundamental-trade-off-17.1.1"><span class="toc-item-num">17.1.1&nbsp;&nbsp;</span><code>gamma</code> and the fundamental trade-off</a></span></li><li><span><a href="#C-and-the-fundamental-trade-off" data-toc-modified-id="C-and-the-fundamental-trade-off-17.1.2"><span class="toc-item-num">17.1.2&nbsp;&nbsp;</span><code>C</code> and the fundamental trade-off</a></span></li></ul></li></ul></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-18"><span class="toc-item-num">18&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#What-We've-Learned-Today" data-toc-modified-id="What-We've-Learned-Today-19"><span class="toc-item-num">19&nbsp;&nbsp;</span>What We've Learned Today<a id="9"></a></a></span></li></ul></div>

#import sys
#!{sys.executable} -m pip install numpy pandas sklearn graphviz

# Importing our libraries

import pandas as pd
import altair as alt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_validate, train_test_split

import sys
sys.path.append('code/')
from display_tree import display_tree
from plot_classifier import plot_classifier
import matplotlib.pyplot as plt

## Lecture Learning Objectives 

- Use `DummyClassifier` and `DummyRegressor` as baselines for machine learning problems.
- Explain the notion of similarity-based algorithms .
- Broadly describe how KNNs use distances.
- Discuss the effect of using a small/large value of the hyperparameter $K$ when using the KNN algorithm 
- Explain the general idea of SVMs with RBF kernel.
- Describe the problem of the curse of dimensionality.
- Broadly describe the relation of `gamma` and `C` hyperparameters and the fundamental tradeoff.

## House Keeping 
- Assignment due Wednesday April 28th
- 1.7 on Assignment
- rounding for Spotify
- Next Assignment release (either Thursday or Monday depending on how far we get today)
- Technical issues, [Online resource](https://bait509-ubc.github.io/BAIT509/intro.html), reaching out! 

## Five Minute Recap/ Lightning Questions 

- What are the 4 types of data/splits that we discussed last class?
- What is the "Golden Rule of Machine Learning"?
- What do we use to split our data?
- If we have 6-fold cross-validation, how many times is `.fit()` being called? 
- What is overfitting? 

### Some lingering questions

- Are decision trees the most basic model?
- What other models can we build?

## Baseline Models

We saw in the last 2 lectures how to build decision tree models which are based on rules (if-else statements), but how can we be sure that these models are doing a good job besides just accuracy? 

Back in high school in chemistry or biology, we've all likely seen and heard of the "control group" where we have an experimental group, does not receive any experimental treatment.  This control group increases the reliability of the results, often through a comparison between control measurements and the other measurements. 


Our baseline model is something like a control group in the sense that it provides a way to sanity-check your machine learning model. We make baseline models not to use for prediction purposes, but as a reference point when we are building other more sophisticated models.

So what is a baseline model then? 

- Baseline: A simple machine learning algorithm based on simple rules of thumb. For example, 
    - most frequent baseline: always predicts the most frequent label in the training set. 


### Dummy Classifier

We are going to build a most frequent baseline model which always predicts the most frequently labelled in the training set.

from sklearn.dummy import DummyClassifier

voting_df = pd.read_csv('data/cities_USA.csv', index_col=0)
voting_df.head()

# feature table
X = voting_df.drop(columns='vote') 

# the target variable
y = voting_df[['vote']] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

We build our model, in the same way as we built a decision tree model but this time using `DummyClassifier`. 

Since we are using a "most frequent" baseline model, we specify the argument `strategy` as `"most_frequent"`

Other options include: [“stratified”, “prior”, “uniform”, “constant”](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) but you just need to know `most_frequent`. 

dummy_clf = DummyClassifier(strategy="most_frequent")

In the last lecture, we stated that it's at this point that we would usually perform cross-validation. 

scores = cross_validate(dummy_clf, X_train, y_train, cv=10, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df

scores_df.mean()

With Dummy Classifiers, we won't need to because we are not hyperparameter tuning. We are using this just to get a base training score. 

dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_train, y_train)

If we see what our model predicts on the feature table for our training split `X_train`, our model will predict the most occurring class from our training data. 

y_train.value_counts()

dummy_clf.predict(X_train)

We can also now take the test score. 

dummy_clf.score(X_test, y_test)

Here is a good example of when we occasionally have test scores better than the training scores.

In this case, it's higher because our test split has a higher proportion of observations that are of class blue and so more of them will be predicted correctly.

Now if we do a decision tree, we can say that this algorithm is doing better than a model build on this simple "most frequently" occurring model. 

dt_clf = DecisionTreeClassifier()

scores = cross_validate(dt_clf, X_train, y_train, cv=10, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df

scores_df.mean()

dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)

This makes us trust our model a little more. 

### Dummy Regressor

For a Dummy regressor, the same principles can be applied but by using different strategies.

[“mean”, “median”, “quantile”, “constant”](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html?highlight=dummyregressor)

The one we are going to become most familiar with is:

**Average (mean) target value:** always predicts the mean of the training set.

house_df = pd.read_csv("data/kc_house_data.csv")
house_df = house_df.drop(columns=["id", "date"])
house_df.head()

Let get our `X` and `y` objects and split our data. 

X = house_df.drop(columns=["price"])
y = house_df["price"]

We still need to make sure we split our data with baseline models.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

we need to import `DummyRegressor` and construct our model. 

We specify `strategy="mean"` however this is the default value so technically we don't need to specify this. 

We train our model and again, it's not needed to cross-validate for this type of algorithm. 

from sklearn.dummy import DummyRegressor

dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train,y_train)

If we predict on our training data, we see it's making the same prediction for each observation. 

dummy_reg.predict(X_train)

if we compare the mean value of the target, we see that our model is simply predicting the average of the training data which is exactly what we expect. 

y_train.mean()

How well does it do? 

dummy_reg.score(X_train, y_train)

We get an $R^2$ value of 0.0. 

When a model has an  $R^2$=0 that means that the model is doing no better than a model that using the mean which is exactly the case here. 

Looking at the test score we see that our model get's a negative value. 

dummy_reg.score(X_test, y_test)

## Let's Practice 

1. 

Below we have the output of `y_train.value_counts()`

```
Position
Forward     13
Defense      7
Goalie       2
dtype: int64
```

In this scenario, what would a `DummyClassifier(strategy='most_frequent')` model predict on the following observation: 


```
   No.  Age  Height  Weight  Experience     Salary
1   83   34     191     210          11  3200000.0
```

2. 

When using a regression model, which of the following is not a possible return value from .score(X,y) ?
    a) 0.0
    b) 1.0
    c) -0.1
    d) 1.5
    
    
3. 

Below are the values for `y` that were used to train  `DummyRegressor(strategy='mean')`:
```
Grade
0     75
1     80
2     90
3     95
4     85
dtype: int64
```

What value will the model predict for every example?


## Analogy-based models

- Suppose you are given the following training examples with corresponding labels and are asked to label a given test example.

An intuitive way to classify the test example is by finding the most "similar" example(s) from the training set and using that label for the test example.  


<img src='imgs/knn-motivation.png' width="100%">
    

In ML, we are given `X` and `y` next, we learn a mapping function from this training data then, given a new unseen example, we predict the target of this new example using our learn-mapping function. 

In the case of decision trees, we did this by asking a series of questions on some features and some thresholds on future values. 

But, another intuitive way to do this is by using the notion of analogy. 
    

#### Example: 

Suppose we are given many images and their labels.

`X` = set of pictures 

`y` = names associated with those pictures. 

Then we are given a new unseen test example, a picture in this particular case.


<img src='imgs/test_pic.png' width="5%">


We want to find out the label for this new test picture. 

Naturally, we would try and find the most similar picture in our training set and using the label of the most similar picture as the label of this new test example. 

That's the basic idea behind analogy-based algorithms.

### Analogy-based models in real life


- <a href="https://www.hertasecurity.com/en" target="_blank">Herta's High-tech Facial Recognition</a>

<img src="imgs/face_rec.png"  width = "20%" alt="404 image" />

- Recommendation systems 

<img src="imgs/book_rec.png"  width = "90%" alt="404 image" />

## Terminology 

In analogy-based algorithms, our goal is to come up with a way to find similarities between examples.
"similarity" is a bit ambiguous so we need some terminology.


- data: think of observations (rows) as points in a high dimensional space. 
- Each feature: Additional dimension. 





<img src="imgs/3d-table.png"  width = "60%" alt="404 image" />

Above we have: 
- Three features; speed attack and defense. 
- 7 points in this three-dimensional space.

Let's go back to our Canada/USA cities dataset. 

cities_df = pd.read_csv("data/canada_usa_cities.csv")
cities_train_df, cities_test_df = train_test_split(cities_df, test_size=0.2, random_state=123)
cities_train_df.head()

cities_train_df.shape

We have 2 features, so 2 dimensions (`longitude` and `latitude`)  and 167 points.
Visualizing this in 2 dimensions gives us the following: 

cities_viz = alt.Chart(cities_train_df).mark_circle(size=20, opacity=0.6).encode(
    alt.X('longitude:Q', scale=alt.Scale(domain=[-140, -40])),
    alt.Y('latitude:Q', scale=alt.Scale(domain=[20, 60])),
    alt.Color('country:N', scale=alt.Scale(domain=['Canada', 'USA'],
                                           range=['red', 'blue']))
)
cities_viz

What about the housing training dataset we saw? 

house_df = pd.read_csv("data/kc_house_data.csv")
house_df = house_df.drop(columns=["id", "date"])

X = house_df.drop(columns=["price"])
y = house_df["price"]


house_X_train, house_X_test, house_y_train, house_y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

house_X_train

house_X_train.shape

Notice a problem?!

We can only visualize data when the dimensions <= 3. 

BUT, in ML, we usually deal with high-dimensional problems where examples are hard to visualize.

- Dimensions≈20: Low dimensional 
- Dimensions≈1000: Medium dimensional
- Dimensions≈100,000: High dimensional

### Feature Vectors

**Feature vector**: a vector composed of feature values associated with an example.


An example feature vector from the cities dataset:

cities_train_df.drop(columns=["country"]).iloc[0].round(2).to_numpy()

An example feature vector from the housing dataset:

house_X_train.iloc[0].round(2).to_numpy()

## Distance

We have our feature vectors, one for each observation, but how we calculate the similarity between these feature vectors? 

One way to calculate the similarity between two points in high-dimensional space is by calculating the distance between them. 

So, if the distance is higher, that means that the points are less similar and when the distance is smaller, that means that the points are more similar. 

### Euclidean distance

There are different ways to calculate distance but we are going to focus on Euclidean distance. 

**Euclidean distance:** Euclidean distance is a measure of the true straight line distance between two points in Euclidean space. ([source](https://hlab.stanford.edu/brian/euclidean_distance_in.html))


The Euclidean distance between vectors 

$u = <u_1, u_2, \dots, u_n>$ and 

$v = <v_1, v_2, \dots, v_n>$ is defined as: 

<br>

$distance(u, v) = \sqrt{\sum_{i =1}^{n} (u_i - v_i)^2}$


Because that equation can look a bit intimidating, let's use it in an example, particularly our Canadian/US cities data.

#### Calculating Euclidean distance "by hand"

cities_train_df.head()

And here is our 2-dimensional space with the observations as points. 

cities_viz

Let’s take 2 points (two feature vectors) from the cities dataset.

two_cities = cities_df.sample(2, random_state=42).drop(columns=["country"])
two_cities

The two sampled points are shown as black circles.

Our goal is to find how similar these two points are.

cities_viz + alt.Chart(two_cities).mark_circle(size=130, color='black').encode(alt.X('longitude'), alt.Y('latitude'))

First, we subtract these two cities. We are subtracting the city at index 0 from the city at index 1.

two_cities.iloc[1] - two_cities.iloc[0]

Next, we square the differences.

(two_cities.iloc[1] - two_cities.iloc[0])**2

Then we sum up the squared differences.

((two_cities.iloc[1] - two_cities.iloc[0])**2).sum()

And then take the square root of the value.

np.sqrt(np.sum((two_cities.iloc[1] - two_cities.iloc[0])**2))

We end with a value of 13.3898 which is the distance between the two cities.

#### Calculating Euclidean distance  with `sklearn`

That's more work than we really have time for and since `sklearn` knows we are very busy people, they have a function that does this for us. 

# Euclidean distance using sklearn
from sklearn.metrics.pairwise import euclidean_distances
euclidean_distances(two_cities)

When we call this function on our two cities data, it outputs this matrix with four values.

- Our first value is the distance between city 0 and itself. 
- Our second value is the distance between city 0 and city1. 
- Our third value is the distance between city 1and city 0.
- Our fourth value is the distance between city 1 and itself.

As we can see, the distances are symmetric. If we calculate the distance between city 0 and city1, it’s going to have the same value as if we calculated the distance between city 1 and city 0.

This isn’t always the case if we use a different metric to calculate distances. 

## Finding the Nearest Neighbour 

Now that we know how to calculate the distance between two points, we are ready to find the most similar examples.

Let's find the closest cities to City 0 from our `cities_train_df` dataframe. 

Using `euclidean_distances` on the entire dataset will calculate the distances from all the cities to all other cities in our dataframe.

dists = euclidean_distances(cities_train_df[["latitude", "longitude"]])
dists

This is going to be of shape 167 by 167 as this was the number of examples in our training portion.

Each row here gives us the distance of that particular city to all other cities in the training data.

dists.shape

pd.DataFrame(dists)

The distance of each city to itself is going to be zero.

If we don’t replace 0 with infinity, each city’s most similar city is going to be itself which is not useful.

np.fill_diagonal(dists, np.inf)
pd.DataFrame(dists)

Now let's look at the distance between city 0 and some other cities. 

We can look at city 0 with its respective `longitude` and `latitude` values. 

cities_train_df.iloc[[0]]

And the distances from city 0 to the other cities in the training dataset.

dists[0]

Remember that our goal is to find the closest example to city 0. 

We can find the closest city to city 0 by finding the city with the minimum distance. 

np.argmin(dists[0])

The closest city in the training dataset is the city with index 157.

This corresponds to index 96 from the original dataset before shuffling.

cities_train_df.iloc[[157]]

If we look at the `longitude` and `latitude` values for the city at index 157 (labelled 96), they look pretty close to those of city 0. 

cities_train_df.iloc[[0]]

dists[0][157]

So, in this case, the closest city to city 0 is city 157 and the Euclidean distance between the two cities is 0.184. 

### Nearest city to a query point

We can also find the distances to a new "test" or "query" city:

query_point = [[-80, 25]]


dists = euclidean_distances(cities_train_df[["longitude", "latitude"]], query_point)
dists[0:10]

We can find the city closest to the query point (-80, 25) using:

np.argmin(dists)

dists[np.argmin(dists)]

So the city at index 147 is the closest point to (-80, 25) with the Euclidean distance between the two equal to 3.838

We can also use Sklearn's `NearestNeighbors` function to get the closest example and the distance between the query point and the closest example.

from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=1)
nn.fit(cities_train_df[['longitude', 'latitude']]);
nn.kneighbors([[-80, 25]])

We can also use `kneighbors` to find the 5 nearest cities in the training split to one of the cities in the test split. 

cities_test_X = cities_test_df[['longitude', 'latitude']]

nn = NearestNeighbors(n_neighbors=5)
nn.fit(cities_train_df[['longitude', 'latitude']]);
nn.kneighbors(cities_test_X.iloc[1])

We need to be careful here though because we need to make sure we pass in a 2D NumPy array as an input.
 This can be fixed using 2 sets of square brackets with our city. 

nn.kneighbors(cities_test_X.iloc[[1]])

This now shows us the 5 distances to the 5 closest cities and their index. 

## Let's Practice


```
               
       seeds   shape  sweetness   water-content      weight    fruit_veg
0      1        0        35          84               100        fruit
1      0        0        23          75               120        fruit
2      1        1        15          90              1360         veg
3      1        1         7          96               600         veg
4      0        0        37          80                 5        fruit
5      0        0        45          78                40        fruit  
6      1        0        27          83               450         veg
7      1        1        18          73                 5         veg
8      1        1        32          80                76         veg
9      0        0        40          83                65        fruit
```

1. Giving the table above and that we are trying to predict if each example is either a fruit or a vegetable, what would be the dimension of feature vectors in this problem?


2. Which of the following would be the feature vector for example 0. 

    a) `array([1,  0, 1, 1, 0, 0, 1, 1, 1, 0])`

    b) `array([fruit,  fruit, veg, veg, fruit, fruit, veg, veg, veg, fruit])`

    c) `array([1, 0, 35, 84, 100])`

    d) `array([1, 0, 35, 84, 100,  fruit])`


3. Given the following 2 feature vectors, what is the Euclidean distance between the following two feature vectors?

    ```
    u = np.array([5, 0, 22, -11])
    v = np.array([-1, 0, 19, -9])
    ```



**True or False**     

4. Analogy-based models find examples from the test set that are most similar to the test example we are predicting.
5. Feature vectors can only be of length 3 since we cannot visualize past that.
6. A dataset with 10 dimensions is considered low dimensional.
7. Euclidean distance will always have a positive value.
8. When finding the nearest neighbour in a dataset using `kneighbors()` from the `sklearn` library, we must `fit`  the data first.
9. Calculating the distances between an example and a query point takes twice as long as calculating the distances between two examples.


## $k$ -Nearest Neighbours ($k$-NNs) Classifier

Now that we have learned how to find similar examples, can we use this idea in a predictive model?

- Yes! The k Nearest Neighbors (kNN) algorithm
- This is a fairly simple algorithm that is best understood by example


<img src="imgs/scatter.png"  width = "30%" alt="404 image" />


We have two features in our toy example; feature 1 and feature 2.

We have two targets; 0 represented with  <font color="blue">blue</font> points and 1 represented with  <font color="orange">orange</font> points.

We want to predict the point in gray.

Based on what we have been doing so far, we can find the closest example ($k$=1) to this gray point and use its class as the class for our grey point. 

In this particular case, we will predict orange as the class for our query point. 

<img src="imgs/scatter_k1.png"  width = "30%" alt="404 image" />

What if we consider more than one nearest example and let them vote on the target of the query example. 

Let's consider the nearest 3 neighbours and let them vote. 

<img src="imgs/scatter_k3.png"  width = "30%" alt="404 image" />

Let's try this with a smaller set of our data and `sklearn`. 

small_train_df = cities_train_df.sample(30, random_state=1223)
small_X_train = small_train_df.drop(columns=["country"])
small_y_train = small_train_df["country"]

one_city = cities_test_df.sample(1, random_state=33)
one_city

chart_knn = alt.Chart(small_train_df).mark_circle().encode(
    alt.X('longitude', scale=alt.Scale(domain=[-140, -40])),
    alt.Y('latitude', scale=alt.Scale(domain=[20, 60])),
    alt.Color('country', scale=alt.Scale(domain=['Canada', 'USA'], range=['red', 'blue'])))

one_city_point = alt.Chart(one_city).mark_point(
    shape='triangle-up', size=400, fill='darkgreen', opacity=1).encode(
    alt.X('longitude'),
    alt.Y('latitude')
)

chart_knn +  one_city_point

We want to find the class for this green triangle city.  




from sklearn.neighbors import KNeighborsClassifier

neigh_clf = KNeighborsClassifier(n_neighbors=1)
neigh_clf.fit(small_X_train, small_y_train)
neigh_clf.predict(one_city.drop(columns=["country"]))

We can set `n_neighbors` equal to 1 to classify this triangle based on one neighbouring point. 

Our prediction here is Canada since the closest point to the green triangle is a city with the class “Canada”.

Now, what if we consider the nearest 3 neighbours?

neigh_clf = KNeighborsClassifier(n_neighbors=3)
neigh_clf.fit(small_X_train, small_y_train)
neigh_clf.predict(one_city.drop(columns=["country"]))

When we change our model to consider the nearest 3 neighbours, our prediction changes!

It now predicts "USA" since the majority of the 3 nearest points are "USA" cities. 

Let's use our entire training dataset and calculate our training and validation scores 

cities_X_train = cities_train_df.drop(columns=['country'])
cities_y_train = cities_train_df['country']
cities_X_test = cities_test_df.drop(columns=['country'])
cities_y_test = cities_test_df['country']


kn1_model = KNeighborsClassifier(n_neighbors=1)
scores = cross_validate(kn1_model, cities_X_train, cities_y_train, cv=10, return_train_score=True)

scores_df = pd.DataFrame(scores)
scores_df

scores_df.mean()

## Choosing K

Ok, so we saw our validation and training scores for `n_neighbors` =1. What happens when we change that? 

kn90_model = KNeighborsClassifier(n_neighbors=90)

scores_df = pd.DataFrame(cross_validate(kn90_model, cities_X_train, cities_y_train, cv=10, return_train_score=True))
scores_df

scores_df.mean()

Comparing this with the results of `n_neighbors=1` we see that we went from overfitting to underfitting.

Let's look at the decision boundaries now. 

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
kn1_model.fit(cities_X_train, cities_y_train);
plt.title("n_neighbors = 1")
plt.ylabel("latitude")
plt.xlabel("longitude")
plot_classifier(cities_X_train, cities_y_train, kn1_model, ax=plt.gca(), ticks=True)


plt.subplot(1, 2, 2)
plt.title("n_neighbors = 90")
kn90_model.fit(cities_X_train, cities_y_train);
plt.ylabel("latitude")
plt.xlabel("longitude")
plot_classifier(cities_X_train, cities_y_train, kn90_model, ax=plt.gca(), ticks=True)

If we plot these two models with $k=1$ on the left and $k=90$ on the right. 

The left plot shows a much more complex model where it is much more specific and attempts to get every example correct. 

The plot on right is plotting a simpler model and we can see more training examples are being predicted incorrectly. 

### How to choose $K$ (`n_neighbors`)?

So we saw the model was overfitting with $k$=1 and  when $k$=100, the model was underfitting.

So, the question is how do we pick $k$?

- Since $k$ is a hyperparameter (`n_neighbors` in `sklearn`), we can use hyperparameter optimization to choose $k$.

Here we are looping over different values of $k$ and performing cross-validation on each one.

results_dict = {"n_neighbors": list(), "mean_train_score": list(), "mean_cv_score": list()}

for k in range(1,50,5):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_validate(knn, cities_X_train, cities_y_train, cv=10, return_train_score = True)
    results_dict["n_neighbors"].append(k)
    results_dict["mean_cv_score"].append(np.mean(scores["test_score"]))
    results_dict["mean_train_score"].append(np.mean(scores["train_score"]))

results_df = pd.DataFrame(results_dict)
results_df

plotting_source = results_df.melt(id_vars='n_neighbors', 
                                  value_vars=['mean_train_score', 'mean_cv_score'], 
                                  var_name='score_type' ,
                                  value_name= 'accuracy' )
                                  
                                  
K_plot = alt.Chart(plotting_source, width=500, height=300).mark_line().encode(
    alt.X('n_neighbors:Q'),
    alt.Y('accuracy:Q', scale=alt.Scale(domain=[.67, 1.00])),
    alt.Color('score_type:N')
).properties(title="Accuracies of n_neighbors for KNeighborsClassifier")

K_plot

Looking at this graph with k on the x-axis and accuracy on the y-axis, we can see there is a sweet spot where the gap between the validation and training scores is the lowest and cross-validation score is the highest. Here it’s when `n_neighbors` is 11.

How do I know it's 11? 
Here's how! 

results_df.sort_values("mean_cv_score", ascending = False)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(cities_X_train, cities_y_train);
print("Test accuracy:", round(knn.score(cities_X_test, cities_y_test), 3))

This testing accuracy surprisingly higher than the validation mean accuracy we had earlier. 

This could be due to having a small dataset. 

## Curse of Dimensionality 

> $k$ -NN usually works well when the number of dimensions is small.

In the previous module, we discussed one of the most important problems in machine learning which was overfitting the second most important problem in machine learning is **the curse of dimensionality**.

If there are many irrelevant features, $k$-NN is hopelessly confused because all of them contribute to finding similarities between examples.

With enough irrelevant features, the accidental similarity between features wips out any meaningful similarity and $k$-NN becomes is no better than random guessing.

## Let's practice 

Consider this toy dataset:

$$ X = \begin{bmatrix}5 & 2\\4 & 3\\  2 & 2\\ 10 & 10\\ 9 & -1\\ 9& 9\end{bmatrix}, \quad y = \begin{bmatrix}0\\0\\1\\1\\1\\2\end{bmatrix}.$$

1. If $k=1$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$?
2. If $k=3$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$?


**True or False**    

3. The classification of the closest neighbour to the test example always contributes the most to the prediction
4. The `n_neighbors` hyperparameter must be less than the number of examples in the training set.
5. Similar to decision trees, $k$-NNs find a small set of good features.
6. With  $k$ -NN, setting the hyperparameter  $k$  to larger values typically increases training score.
7. $k$-NN may perform poorly in high-dimensional space (say, d > 100)

Consider this graph:

<img src="imgs/Q18a.png"  width = "50%" alt="404 image" />

   
8. What value of `n_neighbors` would you choose to train your model on? 

## Regression with $k$-NN 

In $k$-nearest neighbour regression, we take the average of $k$-nearest neighbours instead of the majority vote.

Let's look at an example. 

Here we are creating some synthetic data with fifty examples and only one feature. 

We only have one feature of `length` and our goal is to predict `weight`. 

Regression plots more naturally in 1D, classification in 2D, but of course we can do either for any $d$

Right now, do not worry about the code and only focus on data and our model. 

np.random.seed(0)
n = 50
X_1 = np.linspace(0,2,n)+np.random.randn(n)*0.01
X = pd.DataFrame(X_1[:,None], columns=['length'])
X.head()

y = abs(np.random.randn(n,1))*2 + X_1[:,None]*5
y = pd.DataFrame(y, columns=['weight'])
y.head()

snake_X_train, snake_X_test, snake_y_train, snake_y_test = train_test_split(X, y, test_size=0.2, random_state=123)

Now let's visualize our training data. 

source = pd.concat([snake_X_train, snake_y_train], axis=1)

scatter = alt.Chart(source, width=500, height=300).mark_point(filled=True, color='green').encode(
    alt.X('length:Q'),
    alt.Y('weight:Q'))

scatter

Now let's try the $k$-nearest neighbours regressor on this data. 

Then we create our `KNeighborsRegressor` object with `n_neighbors=1` so we are only considering 1 neighbour and with `uniform` weights. 

from sklearn.neighbors import KNeighborsRegressor

knnr_1 = KNeighborsRegressor(n_neighbors=1, weights="uniform")
knnr_1.fit(snake_X_train,snake_y_train);

predicted = knnr_1.predict(snake_X_train)
predicted

If we scored over regressors we get this perfect score of one since we have `n_neighbors=1` we are likely to overfit.

knnr_1.score(snake_X_train, snake_y_train)  

Plotting this we can see our model is trying to get every example correct since n_neighbors=1. (the mean of 1 point is just going to be the point value)

plt.figure(figsize=(8, 5))
grid = np.linspace(np.min(snake_X_train), np.max(snake_X_train), 1000)
plt.plot(grid, knnr_1.predict(grid), color='orange', linewidth=1)
plt.plot(snake_X_train, snake_y_train, ".r", markersize=10, color='green')
plt.xticks(fontsize= 14);
plt.yticks(fontsize= 14);
plt.xlabel("length",fontsize= 14)
plt.ylabel("weight",fontsize= 14);

What happens when we use `n_neighbors=10`?

knnr_10 = KNeighborsRegressor(n_neighbors=10, weights="uniform")
knnr_10.fit(snake_X_train, snake_y_train)
knnr_10.score(snake_X_train, snake_y_train)

 Now we can see we are getting a lower score over the training set. Our score decreased from 1.0 when to had `n_neighbors=1` to now having a score of 0.925.  

When we plot our model, we can see that it no longer is trying to get every example correct. 

plt.figure(figsize=(8, 5))
plt.plot(grid, knnr_10.predict(grid), color='orange', linewidth=1)
plt.plot(snake_X_train, snake_y_train, ".r", markersize=10, color='green')
plt.xticks(fontsize= 16);
plt.yticks(fontsize= 16);
plt.xlabel("length",fontsize= 16)
plt.ylabel("weight",fontsize= 16);

## Pros and Cons of 𝑘 -Nearest Neighbours


#### Pros:

- Easy to understand, interpret.
- Simply hyperparameter $k$ (`n_neighbors`) controlling the fundamental tradeoff.
- Can learn very complex functions given enough data.
- Lazy learning: Takes no time to `fit`

<br>

#### Cons:

- Can potentially be VERY slow during prediction time. 
- Often not that great test accuracy compared to the modern approaches.
- Need to scale your features. We'll be looking into this in an upcoming lecture (lecture 4 I think?). 


## Let's Practice

$$ X = \begin{bmatrix}5 & 2\\4 & 3\\  2 & 2\\ 10 & 10\\ 9 & -1\\ 9& 9\end{bmatrix}, \quad y = \begin{bmatrix}0\\0\\1\\1\\1\\2\end{bmatrix}.$$

If $k=3$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$ if we were doing regression rather than classification?


## Support Vector Machines (SVMs) with RBF Kernel

Another popular similarity-based algorithm is Support Vector Machines (SVM).

SVMs use a different similarity metric which is called a “kernel” in "SVM land".

We are going to concentrate on the specific kernel called Radial Basis Functions (RBFs).

Back to the good ol' Canadian and USA cities data.

cities_train_df
cities_test_df

cities_X_train.head()

cities_y_train.head()

Unlike with $k$-nn, we are  not going into detail about how support vector machine classifiers or regressor works but more so on how to use it with `sklearn`.

We can use our training feature table ($X$) and target ($y$) values by using this new SVM model with (RBF) but with the old set up with `.fit()` and `.score()` that we have seen time and time again. 

We import the `SVC` tool from the `sklearn.svm` library (The "C" in SVC represents  *Classifier*. To import the regressor we import `SVR` - R for *Regressor*)

from sklearn.svm import SVC

We can cross-validate and score exactly how we saw before. 

(For now, ignore `gamma=0.01` we are addressing it coming up)

svm = SVC(gamma=0.01)
scores = cross_validate(svm, cities_X_train, cities_y_train, return_train_score=True)
pd.DataFrame(scores)

svm_cv_score = scores['test_score'].mean()
svm_cv_score

The biggest thing to know about support vector machines is that superficially, support vector machines are very similar to 𝑘-Nearest Neighbours.

You can think of SVM with RBF kernel as a "smoothed" version of the $k$-Nearest Neighbours.

svm.fit(cities_X_train, cities_y_train);

kn5_model = KNeighborsClassifier(n_neighbors=5)
kn5_model.fit(cities_X_train, cities_y_train);

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.title("SVC")
plot_classifier(cities_X_train, cities_y_train, svm, ax=plt.gca())
plt.subplot(1, 2, 2)
plt.title("KNN with k = 5")
plot_classifier(cities_X_train, cities_y_train, kn5_model, ax=plt.gca());

An observation is classified as a positive class if on average it looks more like positive examples. An observation is classified as a negative class if on average it looks more like negative examples.

The primary difference between 𝑘-NNs and SVMs is that:

- Unlike $k$-NNs, SVMs only remember the key examples (Those examples are called **support vectors**). 
- When it comes to predicting a query point, we only consider the key examples from the data and only calculate the distance to these key examples. This makes it more efficient than 𝑘-NN. 

### Hyperparameters of SVM

There are  2 main hyperparameters for support vector machines with an RBF kernel;

- `gamma` 
- `C`
    
(told you we were coming back to it!) 

We are not equipped to understand the meaning of these parameters at this point but you are expected to describe their relationship to the fundamental tradeoff. 

(In short, `C` is the penalty the model accepts for wrongly classified examples, and `gamma` is the curvature (see [here](https://towardsdatascience.com/hyperparameter-tuning-for-support-vector-machines-c-and-gamma-parameters-6a5097416167) for more) 

See [`scikit-learn`'s explanation of RBF SVM parameters](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)

#### `gamma` and the fundamental trade-off

`gamma` controls the complexity of a model, just like other hyperparameters we've seen.

- higher gamma, higher the complexity.
- lower gamma, lower the complexity.

plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    gamma = 10.0 ** (i - 3)
    rbf_svm = SVC(gamma=gamma)
    rbf_svm.fit(cities_X_train, cities_y_train)
    plt.title("gamma = %s" % gamma);
    plot_classifier(cities_X_train, cities_y_train, rbf_svm, ax=plt.gca(), show_data=False)

#### `C` and the fundamental trade-off

`C` also controls the complexity of a model and in turn the fundamental tradeoff.

- higher `C` values, higher the complexity.
- lower `C` values, lower the complexity.

plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    C = 10.0 ** (i - 1)
    rbf_svm = SVC(C=C, gamma=0.01)
    rbf_svm.fit(cities_X_train, cities_y_train)
    plt.title("C = %s" % C);
    plot_classifier(cities_X_train, cities_y_train, rbf_svm, ax=plt.gca(), show_data=False)

Obtaining optimal validation scores requires a hyperparameter search between both `gamma` and `C` to balance the fundamental trade-off.
We will learn how to search over multiple hyperparameters at a time in lecture 5. 

## Let's Practice

**True or False** 

1.In Scikit Learn’s SVC classifier, large values of gamma tend to result in higher training scores but probably lower validation score.   
2.If we increase both `gamma` and `C`, we can't be certain if the model becomes more complex or less complex.


**Coding practice**

Below is some starter code that creates your feature table and target column from the data from the `bball.csv` dataset (in the data folder).

bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Define X and y
X = bball_df.loc[:, ['height', 'weight', 'salary']]
y = bball_df['position']

1. Split the dataset into 4 objects: `X_train`, `X_test`, `y_train`, `y_test`. Make the test set 0.2 (or the train set 0.8) and make sure to use `random_state=7`.
2. Create an `SVM` model with `gamma` equal to 0.1 and `C` equal to 10 then name the model `model`.
3. Cross-validate using cross_validate() on the objects X_train and y_train specifying the model and making sure to use 5 fold cross-validation and `return_train_score=True`.
4. Calculate the mean training and cross-validation scores.

## What We've Learned Today<a id="9"></a>

- The concept of baseline models.
- How to initiate a Dummy Classifier and Regressor.
- How to measure Euclidean distance.
- How the $k$NN algorithm works for classification.
- How changing $k$ (`n_neighbors`) affects a model.
- What the curse of dimensionality is.


