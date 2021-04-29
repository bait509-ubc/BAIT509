# Lecture 5 - Preprocessing Categorical Features and Column Transformer

*Hayley Boyce, Monday, May 3rd, 2021*

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Lecture-5---Preprocessing-Categorical-Features-and-Column-Transformer" data-toc-modified-id="Lecture-5---Preprocessing-Categorical-Features-and-Column-Transformer-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Lecture 5 - Preprocessing Categorical Features and Column Transformer</a></span><ul class="toc-item"><li><span><a href="#House-Keeping" data-toc-modified-id="House-Keeping-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>House Keeping</a></span></li><li><span><a href="#Lecture-Learning-Objectives" data-toc-modified-id="Lecture-Learning-Objectives-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Lecture Learning Objectives</a></span></li><li><span><a href="#Five-Minute-Recap/-Lightning-Questions" data-toc-modified-id="Five-Minute-Recap/-Lightning-Questions-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Five Minute Recap/ Lightning Questions</a></span><ul class="toc-item"><li><span><a href="#Some-lingering-questions" data-toc-modified-id="Some-lingering-questions-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Some lingering questions</a></span></li></ul></li><li><span><a href="#Introducing-Categorical-Feature-Preprocessing" data-toc-modified-id="Introducing-Categorical-Feature-Preprocessing-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Introducing Categorical Feature Preprocessing</a></span></li><li><span><a href="#Ordinal-encoding" data-toc-modified-id="Ordinal-encoding-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Ordinal encoding</a></span></li><li><span><a href="#One-Hot-Encoding" data-toc-modified-id="One-Hot-Encoding-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>One-Hot Encoding</a></span><ul class="toc-item"><li><span><a href="#How-to-one-hot-encode" data-toc-modified-id="How-to-one-hot-encode-1.6.1"><span class="toc-item-num">1.6.1&nbsp;&nbsp;</span>How to one-hot encode</a></span></li><li><span><a href="#What-happens-if-there-are-categories-in-the-test-data,-that-are-not-in-the-training-data?" data-toc-modified-id="What-happens-if-there-are-categories-in-the-test-data,-that-are-not-in-the-training-data?-1.6.2"><span class="toc-item-num">1.6.2&nbsp;&nbsp;</span>What happens if there are categories in the test data, that are not in the training data?</a></span></li><li><span><a href="#Cases-where-it's-OK-to-break-the-golden-rule" data-toc-modified-id="Cases-where-it's-OK-to-break-the-golden-rule-1.6.3"><span class="toc-item-num">1.6.3&nbsp;&nbsp;</span>Cases where it's OK to break the golden rule</a></span></li></ul></li><li><span><a href="#Binary-features" data-toc-modified-id="Binary-features-1.7"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Binary features</a></span></li><li><span><a href="#Do-we-actually-want-to-use-certain-features-for-prediction?" data-toc-modified-id="Do-we-actually-want-to-use-certain-features-for-prediction?-1.8"><span class="toc-item-num">1.8&nbsp;&nbsp;</span>Do we actually want to use certain features for prediction?</a></span></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-1.9"><span class="toc-item-num">1.9&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#ColumnTransformer" data-toc-modified-id="ColumnTransformer-1.10"><span class="toc-item-num">1.10&nbsp;&nbsp;</span>ColumnTransformer</a></span><ul class="toc-item"><li><span><a href="#Applying-ColumnTransformer" data-toc-modified-id="Applying-ColumnTransformer-1.10.1"><span class="toc-item-num">1.10.1&nbsp;&nbsp;</span>Applying ColumnTransformer</a></span><ul class="toc-item"><li><span><a href="#Do-we-need-to-preprocess-categorical-values-in-the-target-column?" data-toc-modified-id="Do-we-need-to-preprocess-categorical-values-in-the-target-column?-1.10.1.1"><span class="toc-item-num">1.10.1.1&nbsp;&nbsp;</span>Do we need to preprocess categorical values in the target column?</a></span></li></ul></li></ul></li><li><span><a href="#Make-Syntax" data-toc-modified-id="Make-Syntax-1.11"><span class="toc-item-num">1.11&nbsp;&nbsp;</span>Make Syntax</a></span><ul class="toc-item"><li><span><a href="#make_pipeline" data-toc-modified-id="make_pipeline-1.11.1"><span class="toc-item-num">1.11.1&nbsp;&nbsp;</span><code>make_pipeline</code></a></span></li></ul></li><li><span><a href="#make_column_transformer-syntax" data-toc-modified-id="make_column_transformer-syntax-1.12"><span class="toc-item-num">1.12&nbsp;&nbsp;</span><em>make_column_transformer</em> syntax</a></span></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-1.13"><span class="toc-item-num">1.13&nbsp;&nbsp;</span>Let's Practice</a></span></li><li><span><a href="#Text-Data" data-toc-modified-id="Text-Data-1.14"><span class="toc-item-num">1.14&nbsp;&nbsp;</span>Text Data</a></span></li><li><span><a href="#Bag-of-words-(BOW)-representation" data-toc-modified-id="Bag-of-words-(BOW)-representation-1.15"><span class="toc-item-num">1.15&nbsp;&nbsp;</span>Bag of words (BOW) representation</a></span><ul class="toc-item"><li><span><a href="#Extracting-BOW-features-using-scikit-learn" data-toc-modified-id="Extracting-BOW-features-using-scikit-learn-1.15.1"><span class="toc-item-num">1.15.1&nbsp;&nbsp;</span>Extracting BOW features using <em>scikit-learn</em></a></span></li><li><span><a href="#Important-hyperparameters-of-CountVectorizer" data-toc-modified-id="Important-hyperparameters-of-CountVectorizer-1.15.2"><span class="toc-item-num">1.15.2&nbsp;&nbsp;</span>Important hyperparameters of <code>CountVectorizer</code></a></span></li><li><span><a href="#Is-this-a-realistic-representation-of-text-data?" data-toc-modified-id="Is-this-a-realistic-representation-of-text-data?-1.15.3"><span class="toc-item-num">1.15.3&nbsp;&nbsp;</span>Is this a realistic representation of text data?</a></span></li></ul></li><li><span><a href="#Let's-Practice" data-toc-modified-id="Let's-Practice-1.16"><span class="toc-item-num">1.16&nbsp;&nbsp;</span>Let's Practice</a></span><ul class="toc-item"><li><span><a href="#Coding-Practice" data-toc-modified-id="Coding-Practice-1.16.1"><span class="toc-item-num">1.16.1&nbsp;&nbsp;</span>Coding Practice</a></span></li></ul></li></ul></li></ul></div>

# Importing our libraries
import pandas as pd
import altair as alt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVR, SVC

import sys
sys.path.append('code/')
from display_tree import display_tree
from plot_classifier import plot_classifier
import matplotlib.pyplot as plt

# Preprocessing and pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

## House Keeping 
- Quiz on Wednesday 
- Grading issues contact these TA:
    - Question 1 and 2: Andy
    - Question 3: Daniel
    - Question 4: Ali
- Polls! 

## Lecture Learning Objectives 

- Identify when it's appropriate to apply ordinal encoding vs one-hot encoding.
- Explain strategies to deal with categorical variables with too many categories.
- Explain `handle_unknown="ignore"` hyperparameter of `scikit-learn`'s `OneHotEncoder`.
- Use the scikit-learn `ColumnTransformer` function to implement preprocessing functions such as `MinMaxScaler` and `OneHotEncoder` to numeric and categorical features simultaneously.
- Use `ColumnTransformer` to build all our transformations together into one object and use it with `scikit-learn` pipelines.
- Explain why text data needs a different treatment than categorical variables.
- Use `scikit-learn`'s `CountVectorizer` to encode text data.
- Explain different hyperparameters of `CountVectorizer`.

## Five Minute Recap/ Lightning Questions 

- Where does most of the work happen in $k$-nn - `fit` or `predict`?
- What are the 2 hyperparameters we looked at with Support Vector Machines with RBF kernel? 
- What is the range of values after Normalization? 
- Imputation will help data with missing values by removing which of the following the column or the row?
- Pipelines help us not violate what?

### Some lingering questions

- What about categorical features??!  How do we use them in our model!?
- How do we combine everything?!
- What about data with text? 

## Introducing Categorical Feature Preprocessing

Let's bring back our California housing dataset that we explored last class. 
Remember we engineered some of the features in the data.

housing_df = pd.read_csv("data/housing.csv")
train_df, test_df = train_test_split(housing_df, test_size=0.1, random_state=123)

train_df = train_df.assign(rooms_per_household = train_df["total_rooms"]/train_df["households"],
                           bedrooms_per_household = train_df["total_bedrooms"]/train_df["households"],
                           population_per_household = train_df["population"]/train_df["households"])

test_df = test_df.assign(rooms_per_household = test_df["total_rooms"]/test_df["households"],
                         bedrooms_per_household = test_df["total_bedrooms"]/test_df["households"],
                         population_per_household = test_df["population"]/test_df["households"])

train_df = train_df.drop(columns=['total_rooms', 'total_bedrooms', 'population'])  
test_df = test_df.drop(columns=['total_rooms', 'total_bedrooms', 'population']) 

train_df.head()


Last class, we dropped the categorical feature `ocean_proximity` feature.

But it may help with our prediction! We've talked about how dropping columns is not always the best idea especially since we could be dropping potentially useful features.

Let's create our `X_train` and `X_test` again but this time keeping the `ocean_proximity` feature in the data.

X_train = train_df.drop(columns=["median_house_value"])
y_train = train_df["median_house_value"]

X_test = test_df.drop(columns=["median_house_value"])
y_test = test_df["median_house_value"]

Can we make a pipeline and fit it with our `X_train` that has this column now? 

pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", KNeighborsRegressor()),
    ]
)

pipe.fit(X_train, y_train)

Well, that was rude. 

<img src='imgs/denied.png' width="40%">


It does not like the categorical column. 

`scikit-learn` only accepts numeric data as an input and it's not sure how to handle the `ocean_proximity` feature. 


**What now?**

We can:
- Drop the column (not recommended)
- We can transform categorical features into numeric ones so that we can use them in the model. 
- There are two transformations we can do this with:
    - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html" target="_blank">Ordinal encoding</a>
    - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html" target="_blank">One-hot encoding</a> (recommended in most cases)

## Ordinal encoding

Ordinal encoding gives an ordinal numeric value to each unique value in the column. 

Let's take a look at a dummy dataframe to explain how to use ordinal encoding. 

Here we have a categorical column specifying different movie ratings.

X_toy = pd.DataFrame({'rating':['Good', 'Bad', 'Good', 'Good', 
                                  'Bad', 'Neutral', 'Good', 'Good', 
                                  'Neutral', 'Neutral', 'Neutral','Good', 
                                  'Bad', 'Good']})
X_toy

pd.DataFrame(X_toy['rating'].value_counts()).rename(columns={'rating': 'frequency'}).T

Here we can simply assign an integer to each of our unique categorical labels.

We can use sklearn's [`OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) transformer. 

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(dtype=int)
oe.fit(X_toy)
X_toy_ord = oe.transform(X_toy)

X_toy_ord

Since `sklearn`'s transformed output is an array, we can add it next to our original column to see what happened. 

encoding_view = X_toy.assign(rating_enc=X_toy_ord)
encoding_view

we can see that each rating has been designated an integer value. 

For example, `Neutral` is represented by an encoded value of 2 and `Good` a value of 1. Shouldn't `Good` have a higher value? 

We can change that by setting the parameter `categories` within `OrdinalEncoder`

ratings_order = ['Bad', 'Neutral', 'Good']

oe = OrdinalEncoder(categories = [ratings_order], dtype=int)
oe.fit(X_toy)
X_toy_ord = oe.transform(X_toy)

encoding_view = X_toy.assign(rating_enc=X_toy_ord)
encoding_view

Now our `Good` rating is given an ordinal value of 2 and the `Bad` rating is encoded as 0. 

But let's see what happens if we look at a different column now, for example, a categorical column specifying different languages. 

X_toy = pd.DataFrame({'language':['English', 'Vietnamese', 'English', 'Mandarin', 
                                  'English', 'English', 'Mandarin', 'English', 
                                  'Vietnamese', 'Mandarin', 'French','Spanish', 
                                  'Mandarin', 'Hindi']})
X_toy

pd.DataFrame(X_toy['language'].value_counts()).rename(columns={'language': 'frequency'}).T

oe = OrdinalEncoder(dtype=int)
oe.fit(X_toy);
X_toy_ord = oe.transform(X_toy)

encoding_view = X_toy.assign(language_enc=X_toy_ord)
encoding_view

What's the problem here though? 
- We have imposed ordinality on the feature that is no ordinal in value.
- For example, imagine when you are calculating distances. Is it fair to say that French and Hindi are closer than French and Spanish? 
- In general, label encoding is useful if there is ordinality in your data and capturing it is important for your problem, e.g., `[cold, warm, hot]`. 

So what do we do when our values are not truly ordinal categories?

We can do something called ...

## One-Hot Encoding

One-hot encoding (OHE) creates a new binary column for each category in a categorical column.
- If we have $c$ categories in our column.
    - We create $c$ new binary columns to represent those categories.
    


### How to one-hot encode

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, dtype='int')
ohe.fit(X_toy);
X_toy_ohe = ohe.transform(X_toy)

X_toy_ohe

We can convert it to a Pandas dataframe and see that instead of 1 column, we have 6! (We don't need to do this step we are just showing you how it works)

pd.DataFrame(
    data=X_toy_ohe,
    columns=ohe.get_feature_names(['language']),
    index=X_toy.index,
)

Let's try this on our California housing dataset now. 

Although `ocean_proximity` seems like an ordinal feature, let's look at the possible categories.



X_train['ocean_proximity'].unique()

How would you order these? 

Should `NEAR OCEAN` be higher in value than `NEAR BAY`? 

In unsure times, maybe one-hot encoding is the better option. 

ohe = OneHotEncoder(sparse=False, dtype="int")
ohe.fit(X_train[["ocean_proximity"]])
X_imp_ohe_train = ohe.transform(X_train[["ocean_proximity"]])

X_imp_ohe_train

Ok great we've transformed our data, however, Just like before, the transformer outputs a NumPy array. 

transformed_ohe = pd.DataFrame(
    data=X_imp_ohe_train,
    columns=ohe.get_feature_names(['ocean_proximity']),
    index=X_train.index,
)

transformed_ohe.head()

### What happens if there are categories in the test data, that are not in the training data?

Usually, if this is the case, an error will occur. (Not buying it? Try one hot-encoding in the assignment without this argument!)

In the `OneHotEncoder` we can specify `handle_unknown="ignore"` which will then create a row with all zeros. 

That means that all categories that are not recognized to the transformer will appear the same for this feature. 

You'll get to use this in your assignment. 

So our transformer above would then look like this: 

ohe = OneHotEncoder(sparse=False, dtype="int", handle_unknown="ignore")
ohe.fit(X_train[["ocean_proximity"]])
X_imp_ohe_train = ohe.transform(X_train[["ocean_proximity"]])

X_imp_ohe_train

### Cases where it's OK to break the golden rule 

- If it's some fixed number of categories.

For example, if the categories are provinces/territories of Canada, we know the possible values and we can just specify them.

If we know the categories, this might be a reasonable time to “violate the Golden Rule” (look at the test set) and just hard-code all the categories.

This syntax allows you to pre-define the categories.

provs_ters = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']

ohe_cat_example = OneHotEncoder(categories=provs_ters)

## Binary features

Let's say we have the following toy feature, that contains information on if a beverage has caffeine in it or not. 

X_toy = pd.DataFrame({'Caffeine':['No', 'Yes', 'Yes', 'No', 
                                  'Yes', 'No', 'No', 'No', 
                                  'Yes', 'No', 'Yes','Yes', 
                                  'No', 'Yes']})
X_toy

When we do one-hot encoding on this feature, we get 2 separate columns.

ohe = OneHotEncoder(sparse=False, dtype='int')
ohe.fit(X_toy);
X_toy_ohe = ohe.transform(X_toy)

X_toy_ohe

Do we really need 2 columns for this though? 

Either something contains caffeine, or it does not. So we only really need 1 column for this. 

pd.DataFrame(
    data=X_toy_ohe,
    columns=ohe.get_feature_names(['Caffeine']),
    index=X_toy.index,
)

So, for this feature with binary values, we can use an argument called `drop` within `OneHotEncoder` and set it to `"if_binary"`.


ohe = OneHotEncoder(sparse=False, dtype='int', drop="if_binary")
ohe.fit(X_toy);
X_toy_ohe = ohe.transform(X_toy)

X_toy_ohe

pd.DataFrame(
    data=X_toy_ohe,
    columns=ohe.get_feature_names(['Caffeine']),
    index=X_toy.index,
)

Now we see that after one-hot encoding we only get a single column where the encoder has arbitrarily chosen one of the two categories based on the sorting.

In this case, alphabetically it was [`'No'`, `'Yes'`] and it dropped the first category; `No`. 

## Do we actually want to use certain features for prediction?

Sometimes we may have column features like `race` or `sex` that may not be a good idea to include in your model. 

The systems you build are going to be used in some applications. 

It's extremely important to be mindful of the consequences of including certain features in your predictive model. 

Dropping the features like this to avoid racial and gender biases would be a strong suggestion. 

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

2. Does it make sense to be doing ordinal transformations on the `colour` column?
3. If we one hot encoded the `shape` column, what datatype would be the output after using `transform`?
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

5. On which column(s) would you use `OneHotEncoder(sparse=False, dtype=int, drop="if_binary")`?


**True or False?**    
    
6. Whenever we have categorical values, we should use ordinal encoding.  
7. If we include categorical values in our feature table, `sklearn` will throw an error.
8. One-hot encoding a column with 5 unique categories will produce 5 new transformed columns.
9. The values in the new transformed columns after one-hot encoding, are all possible integer or float values.
10. It’s important to be mindful of the consequences of including certain features in your predictive model.


 **But ....now what?**

How do we put this together with other columns in the data before fitting a regressor? 

We want to apply different transformations to different columns.  

Enter... `ColumnTransformer`.

## ColumnTransformer

We left off wondering where to go after we transform our categorical features. 

Problem: Different transformations on different columns.

Right now before we can even fit our regressor we have to apply different transformations on different columns:

- Numeric columns
    - imputation 
    - scaling         
- Categorical columns 
    - imputation 
    - one-hot encoding 
    
What if we have features that are binary, features that are ordinal and features that need just standard one-hot encoding? 
    
    
We can’t use a pipeline since not all the transformations are occurring on every feature.

We could do so without but then we would be violating the Golden Rule of Machine learning when we did cross-validation.

So we need a new tool and it’s called ColumnTransformer!
    

Sklearn's [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html?highlight=columntransformer#sklearn.compose.ColumnTransformer) makes this more manageable.

A big advantage here is that we build all our transformations together into one object, and that way we're sure we do the same operations to all splits of the data.

Otherwise, we might, for example, do the OHE on both train and test but forget to scale the test data.


<img src='imgs/column-transformer.png' width="100%">



### Applying ColumnTransformer

Let's use this new tool on our California housing dataset. 

Just like any new tool we use, we need to import it. 

from sklearn.compose import ColumnTransformer

We must first identify the different feature types perhaps categorical and numeric columns in our feature table. 

If we had binary values or ordinal features, we would split those up too. 

X_train.head()

X_train.info()

numeric_features = [ "longitude",
                     "latitude",
                     "housing_median_age",
                     "households",
                     "median_income",
                     "rooms_per_household",
                     "bedrooms_per_household",
                     "population_per_household"]
                     
categorical_features = ["ocean_proximity"]

Next, we build a pipeline for our dataset.

This means we need to make at least 2 preprocessing pipelines; one for the categorical and one for the numeric features! 

(If we needed to use the ordinal encoder for binary data or ordinal features then we would need a third or fourth.)

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
           ("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

Next, we can actually make our ColumnTransformer.

col_transformer = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ], 
    remainder='passthrough'    
)

We call the numeric and categorical features with their respective pipelines (transformers)  in `ColumnTransformer()`.

The `ColumnTransformer` syntax is somewhat similar to that of `Pipeline` in that you pass in a list of tuples.

But, this time, each tuple has 3 values instead of 2: (name of the step, transformer object, list of columns)

A big advantage here is that we build all our transformations together into one object, and that way we're sure we do the same operations to all splits of the data.

**What does `remainder="passthrough"` do?**

The `ColumnTransformer` will automatically remove columns that are not being transformed.
- AKA: the default value for `remainder` is `'drop'`.
    
We can instead set  `remainder="passthrough"` to keep the columns in our feature table which do not need any preprocessing.     

We don't have any columns that are being removed for this dataset, but this is important to know if we are only interested in a few features. 

Now, you'll start to foreshadow that just like we've seen with most syntax in `sklearn` we need to `fit` our ColumnTransformer.

col_transformer.fit(X_train)

When we `fit` with the `col_transformer`, it calls `fit` on ***all*** the transformers.

And when we transform with the preprocessor, it calls `transform` on ***all*** the transformers.

How do we access information from this now?
Let's say I wanted to see the newly created columns from One-hot-encoding? How do I get those? 

onehot_cols = col_transformer.named_transformers_["categorical"].named_steps["onehot"].get_feature_names(categorical_features)
onehot_cols

Combining this with the numeric feature names gives us all the column names.

columns = numeric_features + list(onehot_cols)
columns

Or we can look at what our X_train looks like after transformation. 

*(Ignore this code, you'll not have to use it in the future, this is just for learning)*

x = list(X_train.columns.values)
del x[5]
X_train_pp = col_transformer.transform(X_train)
pd.DataFrame(X_train_pp, columns= (x  + list(col_transformer.named_transformers_["categorical"].named_steps["onehot"].get_feature_names(categorical_features)))).head()

Now let's make a pipeline with our column transformer and a $k$-nn regressor.

The first step in this pipeline is our `ColumnTransformer` and the second is our  $k$-nn regressor.

main_pipe = Pipeline(
    steps=[
        ("preprocessor", col_transformer), # <-- this is the ColumnTransformer!
        ("reg", KNeighborsRegressor())])

We can then use `cross_validate()` and find our mean training and validation scores!

with_categorical_scores = cross_validate(main_pipe, X_train, y_train, return_train_score=True)
categorical_score = pd.DataFrame(with_categorical_scores)
categorical_score

categorical_score.mean()

In lecture 4, when we did not include this column, we obtain training and test scores of test_score  0.692972 and   0.797033 respectively so we can see a small increase. 

if we had more columns, we could improve our scores in a much more substantial way instead of throwing the information away which is what we have been doing!

There are a lot of steps happening in `ColumnTransformer`, we can use `set_config` from sklearn and it will display a diagram of what is going on in our main pipeline. 

from sklearn import set_config
set_config(display='diagram')
main_pipe

print(main_pipe)

We can also look at this image which shows the more generic version of what happens in `ColumnTransformer` and where it stands in our main pipeline.

<center><img src="imgs/pipeline.png"  width = "90%" alt="404 image" /></center>

#### Do we need to preprocess categorical values in the target column?

- Generally, there is no need for this when doing classification. 
- `sklearn` is fine with categorical labels (y-values) for classification problems. 

## Make Syntax

When we looked at our California housing dataset we had the following pipelines, ColumnTransformer and main pipeline with our model. 

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
           ("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

col_transformer = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ], 
    remainder='passthrough'    
)

main_pipe = Pipeline(
    steps=[
        ("preprocessor", col_transformer), 
        ("reg", KNeighborsRegressor())])

This seems great but it seems quite a lot. 

Well, luckily there is another method and tool that helps make our life easier.

It's call `make_pipeline` and `make_column_transformer`.

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer

### `make_pipeline`

We can shorten our code when we use `Pipeline` from this: 

model_pipeline = Pipeline(
    steps=[
        ("scaling", StandardScaler()),
        ("reg", SVR())])

model_pipeline

print(model_pipeline)

to this: 

model_pipeline = make_pipeline(
            StandardScaler(), SVR())

model_pipeline

print(model_pipeline)

`make_pipeline()` is a shorthand for the `Pipeline()` constructor and does not permit, naming the steps.

Instead, their names will be set to the lowercase of their types automatically.

Now let's adjust our code for our numeric and categoric pipelines for this data using `make_pipeline` instead of `Pipeline()`. 

numeric_transformer = make_pipeline(SimpleImputer(strategy="median"),
                                    StandardScaler())

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder()
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

pipe = make_pipeline(preprocessor, SVR())

Look how much less effort our pipeline took!

Our `ColumnTransformer` may still have the same syntax but guess what?! We have a solution for that too! 

## *make_column_transformer* syntax

Just like `make_pipeline()`,  we can make our column transformer with `make_column_transformer()`.

This eliminates the need to designate names for the numeric and categorical transformations. 

Our code goes from this: 

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features) ]
)

preprocessor

print(preprocessor)

to this:

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features))

preprocessor

print(preprocessor)

This eliminates the need to designate names for the numeric and categorical transformations. 

So our whole thing becomes:

numeric_transformer = make_pipeline(SimpleImputer(strategy="median"),
                                    StandardScaler())

categorical_transformer = make_pipeline(
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder())
                
preprocessor = make_column_transformer(
               (numeric_transformer, numeric_features), 
               (categorical_transformer, categorical_features))
               
pipe = make_pipeline(preprocessor, SVR())


scores = cross_validate(pipe, X_train, y_train, cv=5, return_train_score=True)

pd.DataFrame(scores)

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
2. What transformations are being done to both numeric and categorical columns?


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
4. Which pipeline is transforming the categorical columns?
5. What model is the pipeline fitting on?

**True or False**     
6. If there are missing values in both numeric and categorical columns, we can specify this in a single step in the main pipeline.   
7. If we do not specify `remainder="passthrough"` as an argument in `ColumnTransformer`, the columns not being transformed will be dropped.
8. `Pipeline()` is the same as `make_pipeline()` but  `make_pipeline()` requires you to name the steps.

## Text Data 

Machine Learning algorithms that we have seen so far prefer numeric and fixed-length input that looks like this. 


$$X = \begin{bmatrix}1.0 & 4.0 & \ldots & & 3.0\\ 0.0 & 2.0 & \ldots & & 6.0\\ 1.0 & 0.0 & \ldots & & 0.0\\ \end{bmatrix}$$ 

and 
$$y = \begin{bmatrix}spam \\ non spam \\ spam \end{bmatrix}$$




But what if we are only given data in the form of raw text and associated labels?

How can we represent such data into a fixed number of features? 

Spam/non-spam toy example

Would you be able to apply the algorithms we have seen so far on the data that looks like this?

$$X = \begin{bmatrix}\text{"URGENT!! As a valued network customer you have been selected to receive a £900 prize reward!",}\\ \text{"Lol your always so convincing."}\\ \text{"Congrats! 1 year special cinema pass for 2 is yours. call 09061209465 now!"}\\ \end{bmatrix}$$

and 

$$y = \begin{bmatrix}spam \\ non spam \\ spam \end{bmatrix}$$

<br>

- In categorical features or ordinal features, we have a fixed number of categories.
- In text features such as above, each feature value (i.e., each text message) is going to be different. 
- How do we encode these features? 

## Bag of words (BOW) representation

<center><img src="imgs/bag-of-words.png"  width = "75%" alt="404 image" /></center>

<a href="https://web.stanford.edu/~jurafsky/slp3/4.pdf" target="_blank">Attribution: Daniel Jurafsky & James H. Martin</a> 


One way is to use a simple bag of words (BOW) representation which involves two components. 
- The vocabulary (all unique words in all documents) 
- A value indicating either the presence or absence or the count of each word in the document. 

### Extracting BOW features using *scikit-learn*

Let's say we have 1 feature in our `X` dataframe consisting of the following text messages. 

In our target column, we have the classification of each message as either `spam` or `non spam`. 

X = [
    "URGENT!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "Lol you are always so convincing.",
    "Nah I don't think he goes to usf, he lives around here though",
    "URGENT! You have won a 1 week FREE membership in our £100000 prize Jackpot!",
    "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
    "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"]

y = ["spam", "non spam", "non spam", "spam", "spam", "non spam"]

We import a tool called `CountVectorizer`. 

from sklearn.feature_extraction.text import CountVectorizer

`CountVectorizer` converts a collection of text documents to a matrix of word counts.     

- Each row represents a "document" (e.g., a text message in our example). 
- Each column represents a word in the vocabulary in the training data. 
- Each cell represents how often the word occurs in the document. 
    
    
In the NLP community, a text data set is referred to as a **corpus** (plural: corpora).  

The features should be a 1 dimension array as an input. 

vec = CountVectorizer()
X_counts = vec.fit_transform(X);
bow_df = pd.DataFrame(X_counts.toarray(), columns=sorted(vec.vocabulary_), index=X)
bow_df

### Important hyperparameters of `CountVectorizer` 

There are many useful and important hyperparameters of `CountVectorizer`. 

- `binary`:   
    - Whether to use absence/presence feature values or counts.
- `max_features`:
    - Only considers top `max_features` ordered by frequency in the corpus.
- `max_df`:
    - When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold.
- `min_df`:
    - When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
- `ngram_range`:
    - Consider word sequences in the given range.

X

`CountVectorizer` is carrying out some preprocessing because of the default argument values.   

- Converting words to lowercase (`lowercase=True`). Take a look at the word "urgent" In both cases. 
- getting rid of punctuation and special characters (`token_pattern ='(?u)\\b\\w\\w+\\b'`)


vec.get_feature_names()

bow_df

We can use `CountVectorizer()` in a pipeline just like any other transformer. 

pipe = make_pipeline(CountVectorizer(), SVC())

pipe.fit(X, y);

pipe.predict(X)

pipe.score(X,y)

Here we get a perfect score on our toy dataset data that it's seen already.

How well does it do on unseen data?

X_new = [
    "Congratulations! You have been awarded $1000!",
    "Mom, can you pick me up from soccer practice?",
    "I'm trying to bake a cake and I forgot to put sugar in it smh. ",
    "URGENT: please pick up your car at 2pm from servicing",
    "Call 234950323 for a FREE consultation. It's your lucky day!" ]
    
y_new = ["spam", "non spam", "non spam", "non spam", "spam"]

pipe.score(X_new,y_new)

It's not perfect but it seems to do well on this data too. 

### Is this a realistic representation of text data? 

Of course, this is not a great representation of language.

- We are throwing out everything we know about language and losing a lot of information. 
- It assumes that there is no syntax and compositional meaning in language.  


 ...But it works surprisingly well for many tasks. 

## Let's Practice

1. 
What is the size of the vocabulary for the examples below?

```
X = [ "Take me to the river",
    "Drop me in the water",
    "Push me in the river",
    "dip me in the water"]
```
2. 

Which of the following is not a hyperparameter of `CountVectorizer()`?   
- `binary`
- `max_features` 
- `vocab`
- `ngram_range`

3. What kind of representation do we use for our vocabulary? 

**True or False**     

3. As you increase the value for the `max_features` hyperparameter of `CountVectorizer`, the training score is likely to go up.

4. If we encounter a word in the validation or the test split that's not available in the training data, we'll get an error.


### Coding Practice 

We are going to bring in a new dataset for you to practice on. (Make sure you've downloaded it from the `data` folder from the `lectures` section in Canvas). 

This dataset contains a text column containing tweets associated with disaster keywords and a target column denoting whether a tweet is about a real disaster (1) or not (0). [[Source](https://www.kaggle.com/vstepanenko/disaster-tweets)]

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

## What We've Learned Today<a id="9"></a>

- How to process categorical features.
- How to apply ordinal encoding vs one-hot encoding.
- How to use `ColumnTransformer`, `make_pipeline` `make_column_transformer`.
- How to work with text data.
- How to use `CountVectorizer` to encode text data.
- What the different hyperparameters of `CountVectorizer` are.



