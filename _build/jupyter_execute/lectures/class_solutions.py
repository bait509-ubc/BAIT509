# Exercise Solutions

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer

# Lecture 1

## Let's Practice! 

Are the following supervised or unsupervised problems?

1. Finding groups of similar properties in a real estate data set.

> Unsupervised

2. Predicting real estate prices based on house features like number of rooms, learning from past sales as examples.

> Supervised

3. Identifying groups of animals given features such as "number of legs", "wings/no wings", "fur/no fur", etc.

> Unsupervised

4. Detecting heart disease in patients based on different test results and history.

> Supervised

5. Grouping articles on different topics from different news sources (something like Google News app).

> Unsupervised

Are the following classification or regression problems?

1. Predicting the price of a house based on features such as number of rooms and the year built.

> Regression 

2. Predicting if a house will sell or not based on features like the price of the house, number of rooms, etc.

> Classification 

3. Predicting your grade in BAIT 509 based on past grades.

> Regression 

4. Predicting whether you should bicycle tomorrow or not based on the weather forecast.

> Classification 

5. Predicting a cereal’s manufacturer given the nutritional information.

> Classification 

## Let's Practice! 

Using the data `candybars.csv` from the datafolder to aswer the following questions:

1. How many features are there? 

> 8 

2. How many observations are there? 

> 25 

3. What would be a suitable target with this data?

> Probably `availability` but we could use the other features as well. 

***Answer as either `fit`  or `predict`***
1. Is called first (before the other one).

> `fit`

2. Only takes X as an argument.

> `predict`

3. In scikit-learn, we can ignore its output.In scikit-learn, we can ignore its output.

> `fit`

***Quick Questions***
1. What is the top node in a decision tree called? 

> The root

2. What Python structure/syntax are the nodes in a decision tree similar to? 

> If-else statements

candy_df = pd.read_csv('data/candybars.csv',index_col=0)

# Define X and y
X = candy_df.drop(columns=['availability'])
y = candy_df['availability']

# Creating a model
for min_samples_split in [2, 5, 10]:
    hyper_tree = DecisionTreeClassifier(random_state=1, min_samples_split=min_samples_split)
    hyper_tree.fit(X,y)
    print("For min_sample_split =",min_samples_split, "accuracy=", hyper_tree.score(X, y).round(2))

4. a) Which `min_samples_split` value would you choose to predict this data? <br>

> It has the best accuracy with the lowest value of `min_sample_split`.
   
4. b) Would you choose the same `min_samples_split` value to predict new data?

>  No and we will explain this next lecture. 

5. Do you think most of the computational effort for a decision tree takes place in the `.fit()` stage or `.predict()` stage?

>  Most of the computational effort takes places in the .fit() stage, when we create the model.

# Lecture 2

## Let's Practice 

1. When is the most optimal time to split our data?

> Before we visualize/explore it.

2. Why do we split our data?

> To help us assess how well our model generalizes.

3. Fill in the table below:

| datasets   | `.fit()` | `.score()` | `.predict()` |
|------------|:--------:|:----------:|:------------:|
| Train      |    ✔️     |   ✔️        |   ✔️          |
| Validation |          |   ✔️        |     ✔️        |
| Test       |          |    Once    |   Once       |
| Deployment |          |            |      ✔️       |

## Let's Practice 

1. We carry out cross-validation to avoid reusing the same validation set again and again. Let’s say you do 10-fold cross-validation on 1000 examples. For each fold, how many examples do you train on?

> 900

2. With 10-fold cross-validation, you split 1000 examples into 10-folds. For each fold, when you are done, you add up the accuracies from each fold and divide by what?

> 10

True/False:
- 𝑘-fold cross-validation calls fit 𝑘 times and predict 𝑘 times.

> True

## Let's Practice

Overfitting or Underfitting:
1. If our train accuracy is much higher than our test accuracy.

> Overfitting

2. If our train accuracy and our test accuracy are both low and relatively similar in value.

> Underfitting

3. If our model is using a Decision Tree Classifier for a classification problem with no limit on `max_depth`.

> Likely overfitting


True or False 
1. In supervised learning, the training score is always higher than the validation score.

> False

2. The fundamental tradeoff of ML states that as training score goes up, validation score goes down.

> False

3. More "complicated" models are more likely to overfit than "simple" ones.

> True

5. If our training score is extremely high, that means we're overfitting.

> False

bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Define X and y
X = bball_df.loc[:, ['height', 'weight', 'salary']]
y = bball_df['position']

# 1. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# 2. Create a model
model = DecisionTreeClassifier(max_depth=5)

# 3. Cross validate
scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)

# 4. Covert scores into a dataframe
scores_df = pd.DataFrame(scores)

# 5. Calculate the mean value of each column
mean_scores = scores_df.mean()
mean_scores

6. Is your model overfitting or underfitting? 

> The training score is a little higher than the validation score and thus the model is overfitting. 

# Lecture 3

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


>  `Forward`


2. 

When using a regression model, which of the following is not a possible return value from .score(X,y) ?
    a) 0.0
    b) 1.0
    c) -0.1
    d) 1.5
    
> 1.5    
    
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

> 85


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

> 5 dimensions!

2. Which of the following would be the feature vector for example 0. 

    a) `array([1,  0, 1, 1, 0, 0, 1, 1, 1, 0])`

    b) `array([fruit,  fruit, veg, veg, fruit, fruit, veg, veg, veg, fruit])`

    c) `array([1, 0, 35, 84, 100])`

    d) `array([1, 0, 35, 84, 100,  fruit])`

> c) `array([1, 0, 35, 84, 100])`

3. Given the following 2 feature vectors, what is the Euclidean distance between the following two feature vectors?

    ```
    u = np.array([5, 0, 22, -11])
    v = np.array([-1, 0, 19, -9])
    ```

> 7

**True or False**     

4. Analogy-based models find examples from the test set that are most similar to the test example we are predicting.

> False

5. Feature vectors can only be of length 3 since we cannot visualize past that.

> False

6. A dataset with 10 dimensions is considered low dimensional.

> True

7. Euclidean distance will always have a positive value.

> True (0 and positive) 
8. When finding the nearest neighbour in a dataset using `kneighbors()` from the `sklearn` library, we must `fit`  the data first.

> True

9. Calculating the distances between an example and a query point takes twice as long as calculating the distances between two examples.

> False


## Let's practice 

Consider this toy dataset:

$$ X = \begin{bmatrix}5 & 2\\4 & 3\\  2 & 2\\ 10 & 10\\ 9 & -1\\ 9& 9\end{bmatrix}, \quad y = \begin{bmatrix}0\\0\\1\\1\\1\\2\end{bmatrix}.$$

1. If $k=1$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$?

> 1

2. If $k=3$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$?

> 0

**True or False**    

3. The classification of the closest neighbour to the test example always contributes the most to the prediction

> False

4. The `n_neighbors` hyperparameter must be less than the number of examples in the training set.

> True

5. Similar to decision trees, $k$-NNs find a small set of good features.

> False

6. With  $k$ -NN, setting the hyperparameter  $k$  to larger values typically increases training score.

> False

7. $k$-NN may perform poorly in high-dimensional space (say, d > 100)

> True

Consider this graph:

<img src="imgs/Q18a.png"  width = "50%" alt="404 image" />

   
8. What value of `n_neighbors` would you choose to train your model on? 

> 12

# Lecture 4 - SVM

## Let's Practice

$$ X = \begin{bmatrix}5 & 2\\4 & 3\\  2 & 2\\ 10 & 10\\ 9 & -1\\ 9& 9\end{bmatrix}, \quad y = \begin{bmatrix}0\\0\\1\\1\\1\\2\end{bmatrix}.$$

If $k=3$, what would you predict for $x=\begin{bmatrix} 0\\0\end{bmatrix}$ if we were doing regression rather than classification?

> 1/3

## Let's practice 


**True or False** 

1.In Scikit Learn’s SVC classifier, large values of gamma tend to result in higher training scores but probably lower validation scores.   

> True 

2.If we increase both `gamma` and `C`, we can't be certain if the model becomes more complex or less complex.

> False

**Coding practice**

Below is some starter code that creates your feature table and target column from the data from the `bball.csv` dataset (in the data folder).

bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Define X and y
X = bball_df.loc[:, ['height', 'weight', 'salary']]
y = bball_df['position']



1. Split the dataset into 4 objects: `X_train`, `X_test`, `y_train`, `y_test`. Make the test set 0.2 (or the train set 0.8) and make sure to use `random_state=7`.
2. Create an `SVM` model with `gamma` equal to 0.1 and `C` equal to 10.
3. Cross-validate using cross_validate() on the objects X_train and y_train specifying the model and making sure to use 5 fold cross-validation and `return_train_score=True`.
4. Calculate the mean training and cross-validation scores.

# 1. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# 2. Create a model
model = SVC(gamma=0.1, C=10)

# 3. Cross-validate
scores_df = pd.DataFrame(cross_validate(model,X_train,y_train, cv=5, return_train_score=True))

# 4.  Calculate the mean training and cross-validation scores.
print('The mean training score is', scores_df.mean()['train_score'].round(3), 'and the mean validation score is', scores_df.mean()['test_score'].round(3))


Yikes! I wonder how this can be improved?! More on this in the next class :) 

# Lecture 4 - Pipeline

## Let's practice 

1. Name a model that will still produce meaningful predictions with different scaled column values.

> Decision Tree Classifier

2. Complete the following statement: Preprocessing is done ____.  


- To the model but before training
- To the data before training the model
- To the model after training
- To the data after training the model

>  To the data before training the model

3. `StandardScaler` is a type of what?

> Transformer

4. What data splits does `StandardScaler` alter (Training, Testing, Validation, None, All)?

> All

**True or False**   

5. Columns with lower magnitudes compared to columns with higher magnitudes are less important when making predictions.  

> False

6. A model less sensitive to the scale of the data makes it more robust.

> True

## Let's Practice

1. When/Why do we need to impute our data?

> When we have missing data.

2. If we have `NaN` values in our data, can we simply drop the column missing the data?

> Yes, if the majority of the values are missing from the column


3. Which scaling method will never produce negative values?

> Normalization (`MinMaxScaler`)


4. Which scaling method will never produce values greater than 1?

> Normalization (`MinMaxScaler`)


5. Which scaling method will produce values where the range depends on the values in the data?

> Standardization (StandardScaler)



**True or False**     
6. `SimpleImputer` is a type of transformer.  

> True

7. Scaling is a form of transformation.   

> True


8. We can use `SimpleImputer` to impute values that are missing from numerical and categorical columns.    
> True



1. Which of the following steps cannot be used in a pipeline?
    - Scaling
    - Model building 
    - Imputation
    - Data Splitting

> Data Splitting


2. Why can't we fit and transform the training and test data together?

> It's violating the golden rule.


**True or False**     
3. We have to be careful of the order we put each transformation and model in a pipeline.   

> True


4. Pipelines will fit and transform on both the training and validation folds during cross-validation.

> False



Let's bring in the basketball dataset again.


# Loading in the data
bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Define X and y
X = bball_df.loc[:, ['height', 'weight', 'salary']]
y = bball_df['position']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

Build a pipeline named `bb_pipe` that: 
1. Imputes using "median" as a strategy, 
2. scale using `StandardScaler` 
3. builds a `KNeighborsClassifier`.


Next, do 5 fold cross-validation on the pipeline using `X_train` and `y_train` and save the results in an dataframe.
Take the mean of each column and assess your model.

# Build a pipeline
bb_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")),
                   ("scaler", StandardScaler()),
                   ("knn", KNeighborsClassifier())])

# Do 5 fold cross-validation on the pipeline using `X_train` and `y_train` and save the results in an dataframe.
cross_scores = pd.DataFrame(cross_validate(bb_pipe, X_train, y_train, return_train_score=True))

# Transform cross_scores to a dataframe and take the mean of each column
# Save the result in an object named mean_scores
mean_scores = cross_scores.mean()
mean_scores

# Lecture 5

## Let's Practice


```
           name    colour    location    seed   shape  sweetness   water-content  weight  popularity
0         apple       red     canada    True   round     True          84         100      popular
1        banana    yellow     mexico   False    long     True          75         120      popular
2    cantaloupe    orange      spain    True   round     True          90        1360      neutral
3  dragon-fruit   magenta      china    True   round    False          96         600      not popular
4    elderberry    purple    austria   False   round     True          80           5      not popular
5           fig    purple     turkey   False    oval    False          78          40      neutral
6         guava     green     mexico    True    oval     True          83         450      neutral
7   huckleberry      blue     canada    True   round     True          73           5      not popular
8          kiwi     brown      china    True   round     True          80          76      popular
9         lemon    yellow     mexico   False    oval    False          83          65      popular

```

1.  What would be the unique values given to the categories in the `popularity` column, if we transformed it with ordinal encoding?

- `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- `[0, 1, 2]` 
- `[1, 2, 3]`
- `[0, 1, 2, 3]`

> `[0, 1, 2]` 

2. Does it make sense to be doing ordinal transformations on the `colour` column?

> No

3. If we one hot encoded the `shape` column, what datatype would be the output after using `transform`?

> NumPy array

4. Which of the following outputs is the result of one-hot encoding the `shape` column?    

a)    

``` 
array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
       [1, 0, 1, 1, 1, 0, 0, 1, 1, 0]])
```

b)    

```
array([[0, 0, 1],
       [1, 0, 0],
       [0, 0, 1],
       [0, 0, 1],
       [0, 0, 1],
       [0, 1, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 0, 1],
       [0, 1, 0]])
```

c)

```
array([[0, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0]])
```

d) 

```
array([[0],
       [5],
       [0],
       [3],
       [0],
       [0],
       [3],
       [0],
       [5],
       [3],
       [1],
       [4],
       [3],
       [2]])

```

> B

5. On which column(s) would you use `OneHotEncoder(sparse=False, dtype=int, drop="if_binary")`?

> `seed`, `sweetness`


**True or False?**    
    
6. Whenever we have categorical values, we should use ordinal encoding. 

> False

7. If we include categorical values in our feature table, `sklearn` will throw an error.

> True

8. One-hot encoding a column with 5 unique categories will produce 5 new transformed columns.

> True


9. The values in the new transformed columns after one-hot encoding, are all possible integer or float values.

> False

10. It’s important to be mindful of the consequences of including certain features in your predictive model.

> True


## Let's Practice 


Refer to the dataframe to answer the following question.
```
       colour   location    shape   water_content  weight
0       red      canada      NaN         84          100
1     yellow     mexico     long         75          120
2     orange     spain       NaN         90          NaN
3    magenta     china      round        NaN         600
4     purple    austria      NaN         80          115
5     purple    turkey      oval         78          340
6     green     mexico      oval         83          NaN
7      blue     canada      round        73          535
8     brown     china        NaN         NaN        1743  
9     yellow    mexico      oval         83          265
```

<br>
 
1. How many categorical columns are there and how many numeric?

> 3 categoric columns and 2 numeric columns

2. What transformations are being done to both numeric and categorical columns?

> Imputation


Use the diagram below to answer the following questions.

```
Pipeline(
    steps=[('columntransformer',
               ColumnTransformer(
                  transformers=[('pipeline-1',
                                  Pipeline(
                                    steps=[('simpleimputer',
                                             SimpleImputer(strategy='median')),
                                           ('standardscaler',
                                             StandardScaler())]),
                      ['water_content', 'weight', 'carbs']),
                                ('pipeline-2',
                                  Pipeline(
                                    steps=[('simpleimputer',
                                             SimpleImputer(fill_value='missing',
                                                                strategy='constant')),
                                           ('onehotencoder',
                                             OneHotEncoder(handle_unknown='ignore'))]),
                      ['colour', 'location', 'seed', 'shape', 'sweetness',
                                                   'tropical'])])),
         ('decisiontreeclassifier', DecisionTreeClassifier())])
```

3. How many columns are being transformed in `pipeline-1`?

> 3

4. Which pipeline is transforming the categorical columns?

> pipeline-2

5. What model is the pipeline fitting on?

> DecisionTreeClassifier

**True or False**     
6. If there are missing values in both numeric and categorical columns, we can specify this in a single step in the main pipeline.   

> False

7. If we do not specify `remainder="passthrough"` as an argument in `ColumnTransformer`, the columns not being transformed will be dropped.

> True

8. `Pipeline()` is the same as `make_pipeline()` but  `make_pipeline()` requires you to name the steps.

> False

## Let's Practice

1. 
What is the size of the vocabulary for the examples below?

```
X = [ "Take me to the river",
    "Drop me in the water",
    "Push me in the river",
    "dip me in the water"]
```

> 10


2. 

Which of the following is not a hyperparameter of `CountVectorizer()`?   
- `binary`
- `max_features` 
- `vocab`
- `ngram_range`

>  `vocab`

3. What kind of representation do we use for our vocabulary? 

> Bag of Words 

**True or False**     

4. As you increase the value for the `max_features` hyperparameter of `CountVectorizer`, the training score is likely to go up.

> True 

5. If we encounter a word in the validation or the test split that's not available in the training data, we'll get an error.

> False 


### Coding Practice 

We are going to bring in a new dataset for you to practice on. (Make sure you've downloaded it from the `data` folder from the `lectures` section in Canvas). 

This dataset contains a text column containing tweets associated with disaster keywords and a target column denoting whether a tweet is about a real disaster (1) or not (0). [[Source](Source)]

# Loading in the data
tweets_df = pd.read_csv('data/balanced_tweets.csv').dropna(subset=['target'])

# Split the dataset into the feature table `X` and the target value `y`
X = tweets_df['text']
y = tweets_df['target']

# Split the dataset into X_train, X_test, y_train, y_test 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)

X_train.head()

1. Make a pipeline with `CountVectorizer` as the first step and `SVC()` as the second.
2. Perform 5 fold cross-validation using your pipeline and return the training score. 
3. Convert the results into a dataframe. 
4. What are the mean training and validation scores? 
5. Train your pipeline on the training set.
6. Score the pipeline  on the test set.

# 1. Make a pipeline with `CountVectorizer` as the first step and `SVC()` as the second
pipe = make_pipeline(CountVectorizer(), SVC())

# 2. Perform 5 fold cross-validation using your pipeline and return the training score
cv_scores = cross_validate(pipe, X_train, y_train, return_train_score=True)

# 3. Convert the results into a dataframe
cv_scores_df = pd.DataFrame(cv_scores)

# 4. What are the mean training and validation scores?
print('Mean training score:', cv_scores_df.mean()['train_score'])
print('Mean Validation score:', cv_scores_df.mean()['test_score'])

# 5. Train your pipeline on the training set
pipe.fit(X_train, y_train)

# 6. Score the pipeline  on the test set
tweet_test_score = pipe.score(X_test, y_test)
print('Test score:', tweet_test_score)

# Lecture 6

## Let's Practice

Using Naive bayes by hand, what class would naive Bayes predict for the second example "I like Sauder". 

df = pd.DataFrame({'X': [
                        "URGENT!! As a valued network customer you have been selected to receive a £900 prize reward!",
                        "Lol you are always so convincing.",
                        "Sauder has interesting courses.",
                        "URGENT! You have won a 1 week FREE membership in our £100000 prize Jackpot!",
                        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free!",
                        "Sauder has been interesting so far." ],
                   'y': ["spam", "non spam", "non spam", "spam", "spam", "non spam"]})
df

count_vect = CountVectorizer(max_features = 4, stop_words='english')
data = count_vect.fit_transform(df['X'])
train_bow_df = pd.DataFrame(data.toarray(), columns=sorted(count_vect.vocabulary_), index=df['X'])

train_bow_df['target'] = df['y'].tolist()
train_bow_df

test_bow_df.iloc[[1]]

Let's do some of the steps here: 

#### spam

1. Prior probability: 
    $P(\text{spam}) = 3/6$ 

2. Conditional probabilities: 
    1. $P(\text{free} = 0 \mid \text{spam}) = 1/3$
    2. $P(\text{prize} = 0 \mid \text{spam}) = 1/3$
    3. $P(\text{sauder} = 1 \mid \text{spam}) = 0/3$
    4. $P(\text{urgent} = 0 \mid \text{spam}) = 1/3$
    
3. $P(\textrm{spam}|\text{free} = 0, \text{prize} = 0, \text{sauder} = 1,  \text{urgent} = 0) = P(\text{free} = 0|\textrm{spam})*P(\text{prize} = 0|\textrm{spam})*P(\textrm{sauder = 1}|\textrm{spam})*P(\text{urgent} = 0|\textrm{spam})*P(\textrm{spam})$

$=  \frac{1}{3} * \frac{1}{3}* \frac{0}{3} * \frac{1}{3} *\frac{3}{6} $

$ = 0$


#### non spam
4. Prior probability: 
    $P(\text{non spam}) = 3/6$ 

5. Conditional probabilities: 
    1. $P(\text{free} = 0 \mid \text{non spam}) = 3/3$
    2. $P(\text{prize} = 0 \mid \text{non spam}) = 3/3$
    3. $P(\text{sauder} = 1 \mid \text{non spam}) = 2/3$
    4. $P(\text{urgent} = 0 \mid \text{non spam}) = 3/3$
    
6.  $P(\textrm{non spam}|\text{free} = 0, \text{prize} = 0, \text{sauder} = 1,  \text{urgent} = 0) = P(\text{free} = 0|\textrm{non spam})*P(\text{prize} = 0|\textrm{non spam})*P(\textrm{sauder = 1}|\textrm{non spam})*P(\text{urgent} = 0|\textrm{non spam})*P(\textrm{non spam})$

$=  \frac{3}{3} * \frac{3}{3}* \frac{2}{3} * \frac{3}{3} *\frac{3}{6} $

$ = 1/3$


#### Final Class
7. CLASS AS: 

## Let's Practice

1. Which method will attempt to find the optimal hyperparameter for the data by searching every combination possible of hyperparameter values given?

> Exhaustive Grid Search (`GridSearchCV`)

2. Which method gives you fine-grained control over the amount of time spent searching?

> Randomized Grid Search (`RandomizedSearchCV`)


3. If I want to search for the most optimal hyperparameter values among 3 different hyperparameters each with 3 different values how many trials of cross-validation would be needed?

> $3 * 3 * 3 = 27$


**True or False** 

4. A Larger `n_iter` will take longer but will search over more hyperparameter values.

> True

5. Automated hyperparameter optimization can only be used for multiple hyperparameters.

> False

### Coding Practice 

We are going to practice grid search using our basketball dataset that we have seen before. 

# Loading in the data
bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Define X and y
X = bball_df.loc[:, ['height', 'weight', 'salary']]
y = bball_df['position']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

bb_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")),
                   ("scaler", StandardScaler()),
                   ("knn", KNeighborsClassifier())])

1. Using the pipeline `bb_pipe` provided, create a grid of parameters to search over `param_grid`. Search over the values 1, 5, 10, 20, 30, 40, and 50 for the hyperparameter `n_neighbors` and 'uniform' and 'distance' for the hyperparameter `weights` (make sure to call them appropriately). 
2. Use `GridSearchCV` to hyperparameter tune using cross-validate equal to 10 folds. Make sure to specify the arguments `verbose=2` and `n_jobs=-1`.
3. Train your  pipeline with grid search.
4. Find the best hyperparameter values. Make sure to print these results.
5. Lastly, score your model on the test set.

# 1. Build a grid of the parameters you wish to search. 
param_grid = {
    "knn__n_neighbors" : [1, 5, 10, 20, 30, 40, 50],
    "knn__weights" : ['uniform', 'distance']
}

# 2. Conduct grid search with 10 fold cross-validation
grid_search = GridSearchCV(bb_pipe, param_grid, cv=10, verbose=2, n_jobs=-1)

# 3. Train your  pipeline with grid search. 
grid_search.fit(X_train, y_train)

# 4. Find the best hyperparameter values.
best_hyperparams = grid_search.best_params_
print(best_hyperparams)

# 5. Lastly, score your model on the test set.
bb_test_score = grid_search.score(X_test, y_test)
bb_test_score

# Lecture 7

## Let's Practice

1. What is the name of a well-known `Ridge` hyperparameter?

> `alpha`

2. What value of this hyperparameter makes it equivalent to using `LinearRegression`?

> 0 

Use the following equation to answer the questions below: 

$$ \text{predicted(backpack_weight)} =  3.02 * \text{#laptops} + 0.3 * \text{#pencils} + 0.5 $$

3. What is our intercept value?

> 0.5

4. If I had 2 laptops 3 pencils in my backpack, what weight would my model predict for my backpack?

> 7.44

**True or False:**  
5. Ridge is a regression modelling approach.

> True

6. Increasing the hyperparameter from Question 1 increases model complexity.  

> False

7. `Ridge` can be used with datasets that have multiple features.  

> True

8. With `Ridge`, we learn one coefficient per training example.  

> False

9. Coefficients can help us interpret our model even if unscaled.  

> True

## Let's Practice 

We have the following text, which we wish to classify as either a positive or negative movie review.     
Using the words below (which are features in our model) with associated coefficients, answer the next 2 questions.     
The input for the feature value is the number of times the word appears in the review. 


|   Word            | Coefficient | 
|--------------------|-------------|
|excellent           | 2.2         | 
|disappointment      | -2.4        |
|flawless            | 1.4         |
|boring              | -1.3        |
|unwatchable         | -1.7        |

Intercept = 1.3


1. What value do you calculate after using the weights in the model above for the above review? 

***I thought it was going to be excellent but instead, it was unwatchable and boring.***

The input feature value would be the number of times the word appears in the review (like `CountVectorizer`).

> 0.5

2. Would the model classify this review as a positive or negative review (classes are specified alphabetically) ?

- Positive review

We are trying to predict if a job applicant would be hired based on some features contained in their resume. 



Below we have the output of `.predict_proba()` where column 0 shows the probability the model would predict "hired" and column 1 shows the probability the model would predict "not hired".


```out
array([[0.04971843, 0.95028157],
       [0.94173513, 0.05826487],
       [0.74133975, 0.25866025],
       [0.13024982, 0.86975018],
       [0.17126403, 0.82873597]])
```

Use this output to answer the following questions.

3. If we had used `.predict()` for these examples instead of `.predict_proba()`, how many of the examples would the model have predicted "hired"?

> 2


4. If the true class labels are below, how many examples would the model have correctly predicted with `predict()`? 

```out
['hired', 'hired', 'hired', 'not hired', 'not hired']
```

> 4

**True or False?**     
5. Increasing logistic regression's `C` hyperparameter increases the model's complexity.   

> True 

6. Unlike with `Ridge` regression, coefficients are not interpretable with logistic regression.    

>. False
7.  `predict` returns the positive class if the predicted probability of the positive class is greater than 0.5.     

> True 

8. In logistic regression, a function is applied to convert the raw model output into probabilities.    

> True 


**Coding Problem**

Let’s import the Pokémon dataset from our `data` folder. We want to see how well our model does with logistic regression. Let’s try building a simple model with default parameters.

pk_df = pd.read_csv('data/pokemon.csv')

train_df, test_df = train_test_split(pk_df, test_size=0.2, random_state=1)

X_train = train_df.drop(columns=['legendary'])
y_train = train_df['legendary']
X_test = test_df.drop(columns=['legendary'])
y_test = test_df['legendary']


numeric_features = ["attack",
                    "defense" ,
                    "sp_attack",
                    "sp_defense",
                    "speed",
                    "capture_rt"]

drop_features = ["type", "deck_no", "gen", "name", "total_bs"]

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

preprocessor = make_column_transformer(
    ("drop", drop_features),
    (numeric_transformer, numeric_features))

1. Build and fit a pipeline containing the column transformer and a logistic regression model using the parameter class_weight="balanced" (you will learn about this in lecture 9!).
2. Score your model on the test set.
3. Find the model’s feature coefficients and answer the below questions 
    a. Which feature contributes the most in predicting if an example is legendary or not.
    b.As the capture rate value increases, will the model more likely predict a legendary or not legendary Pokémon?

# 1. Build a pipeline containing the column transformer and a Logistic Regression model
# use the parameter class_weight="balanced"
pkm_pipe = make_pipeline(preprocessor, LogisticRegression(class_weight="balanced"))

# 1. Fit your pipeline on the training data
pkm_pipe.fit(X_train, y_train);

# 2. Score your model on the test set 
lr_scores = pkm_pipe.score(X_test, y_test)
print("logistic Regression Test Score:", lr_scores)

# Find the model’s feature coefficients
pkm_coefs = pd.DataFrame({'features':numeric_features, 'coefficients':pkm_pipe['logisticregression'].coef_[0]})
pkm_coefs

a. Which feature contributes the most in predicting if an example is legendary or not.
> `defense`, this feature has the highest magnitude!
    
b.As the capture rate value increases, will the model more likely predict a legendary or not legendary Pokémon?
> Not Legendary, since the sign of the coefficient is negative. 

## Lecture 8 

## Let's Practice 

1. What question is usually more complex?

> Business question/objective

2. What model needs to be made for all problems?

> Baseline - Dummy 

3. In supervised learning, once we have our business objective, part of our statistical question is identifying what?

> Our target variable 

**True or False:**

4. When writing your reports, it's important to consider who is reading it.  

> True

5. Sometimes you may need to dig a little to figure out exactly what the client wants.

> True

6. In supervised learning, we should take into consideration the uncertainty of our models.

> True

## Let's Practice 

1. As we increase features, which score will always increase? 

> Training score

2. Between `RFE` and `RFECV` which one finds the optimal number of features for us?

> `RFECV`

3. Which method starts with all our features and iteratively removes them from our model?

> Recursive Feature Elimination

4. Which method starts with no features and iteratively adds features?

> Forward Selection

5. Which method does not take into consideration `feature_importance` when adding/removing features? 

> Forward Selection