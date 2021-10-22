#!/usr/bin/env python
# coding: utf-8

# # BAIT 509 Assignment 2: Preprocessing, Pipelines and Hyperparameter Tuning
# 
# __Evaluates__: Lectures 4 - 6. 
# 
# __Rubrics__: Your solutions will be assessed primarily on the accuracy of your coding, as well as the clarity and correctness of your written responses. The MDS rubrics provide a good guide as to what is expected of you in your responses to the assignment questions and how the TAs will grade your answers. See the following links for more details:
# 
# - [mechanics_rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_mech.md): submit an assignment correctly.
# - [accuracy rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_accuracy.md): evaluating your code.
# - [autograde rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_autograde.md): evaluating questions that are either right or wrong.
# - [reasoning rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_reasoning.md): evaluating your written responses.
# 
# ## Tidy Submission 
# rubric={mechanics:2}
# 
# - Complete this assignment by filling out this jupyter notebook.
# - You must use proper English, spelling, and grammar.
# - You will submit two things to Canvas:
#     1. This jupyter notebook file containing your responses ( an `.ipynb` file); and,
#     2. An `.html` file of your completed notebook (use `jupyter nbconvert --to html_embed assignment.ipynb` in the terminal to generate the html file or under `File` -> `Export Notebook As` -> `HTML`).
#     
#  <br>  
# 
#  Submit your assignment through [UBC Canvas](https://canvas.ubc.ca/courses/58082) by **11:59 pm Monday, May 10th**.

# ## Answering Questions
# 
# - Places that you see `raise NotImplementedError # No Answer - remove if you provide an answer`. Substitute the `None` above it and replace the `raise NotImplementedError # No Answer - remove if you provide an answer` with your completed code and answers, then proceed to run the cell!
# 
# - Any place you see `____`, you must fill in the function, variable, or data to complete the code.
# 

# In[ ]:


# Import libraries
from hashlib import sha1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()

pd.set_option("display.max_colwidth", 200)


from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

import test_assignment2 as t


# ## Introduction and learning goals <a name="in"></a>
# <hr>
# 
# Welcome to the assignment! After working on this assignment, you should be able to:
# 
# - Identify when to implement feature transformations such as imputation and scaling.
# - Apply `sklearn.pipeline.Pipeline` to build a machine learning pipeline.
# - Use `sklearn` for applying numerical feature transformations on the data.
# - Identify when it's appropriate to apply ordinal encoding vs one-hot encoding.
# - Explain strategies to deal with categorical variables with too many categories.
# - Use `ColumnTransformer` to build all our transformations together into one object and use it with `scikit-learn` pipelines.
# - Carry out hyperparameter optimization using `sklearn`'s `GridSearchCV` and `RandomizedSearchCV`.

# ## Introduction <a name="in"></a>
# <hr>
# 
# A crucial step when using machine learning algorithms on real-world datasets is preprocessing. This assignment will give you some practice to build a preliminary supervised machine learning pipeline on a real-world dataset. 

# ## Exercise 1: Introducing and Exploring the dataset <a name="1"></a>
# <hr>
# 
# In this assignment, you will be working on a sample of [the adult census dataset](https://www.kaggle.com/uciml/adult-census-income#). We have made some modifications to this data so that it's easier to work with. 
# 
# This is a classification dataset and the classification task is to predict whether income exceeds 50K per year or not based on the census data. You can find more information on the dataset and features [here](http://archive.ics.uci.edu/ml/datasets/Adult).
# 
# 
# *Note that many popular datasets have sex as a feature where the possible values are male and female. This representation reflects how the data were collected and is not meant to imply that, for example, gender is binary.*

# In[ ]:


### BEGIN STARTER CODE
census_df = pd.read_csv("adult.csv")
census_df.head()
### END STARTER CODE


# ### 1.1 Data splitting 
# rubric={accuracy:2}
# 
# To avoid violation of the golden rule, the first step before we do anything is splitting the data. 
# 
# Split the data into `train_df` (80%) and `test_df` (20%). Keep the target column (`income`) in the splits so that we can use it in EDA. 
# 
# To pass the test, please use `random_state=42`
# 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test_1_1(train_df,test_df)


# Let's examine our `train_df`. 

# In[ ]:


### BEGIN STARTER CODE

train_df.sort_index().head()

### END STARTER CODE


# We see some missing values represented with a "?". Probably these were the questions not answered by some people during the census.  Usually `.describe()` or `.info()` methods would give you information on missing values. But here, they won't pick "?" as missing values as they are encoded as strings instead of an actual NaN in Python. So let's replace them with `np.NaN` before we carry out EDA. If you do not do it, you'll encounter an error later on when you try to pass this data to a classifier. 

# In[ ]:


### BEGIN STARTER CODE

train_df_nan = train_df.replace("?", np.NaN)
test_df_nan = test_df.replace("?", np.NaN)
train_df_nan.head()
### END STARTER CODE


# ### 1.2 Numeric vs. categorical features
# rubric={reasoning:2}
# 
# Identify numeric and categorical features and create lists for each of them.
# 
# We've started this by adding a column label for each feature type. 
# 
# *Save the column names as string elements in each of the corresponding lists below*

# In[ ]:


numeric_features = ["age", ]
categorical_features = ["workclass", ]


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test_1_2_1(numeric_features)


# In[ ]:


t.test_1_2_2(categorical_features)


# ### 1.3 Describing your data
# rubric={accuracy:2}
# 
# 
# Use `.describe()` to show summary statistics of each feature in the `train_df_nan` dataframe. It's important to call the argument `include="all"` to get statistics of ***all*** the columns. 
# 
# *Save this in an object named `train_stats`.* 

# In[ ]:


train_stats = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test_1_3(train_stats)


# ### 1.4 Visualizing features
# rubric={viz:1}
# 
# We have provided you with some code that will visualize the distributions of the **numeric features** in our data, plotted as histograms. 
# 
# Fill in the code where you see `___` below so that it produces the graphs.  
# 
# *This may take 10 seconds to run.*

# In[ ]:


# def plot_histogram(df,feature):
#     """
#     plots a histogram of the distribution of features

#     Parameters
#     ----------
#     feature: str
#         the feature name
#     Returns
#     -------
#     altair.vegalite.v3.api.Chart
#         an Altair histogram 
#     """


#    ## Creates a visualization named histogram that plots the feature on the x-axis
#    ##  and the frequency/count on the y-axis and colouring by the income column
#     histogram = alt.Chart(df).mark_bar(
#         opacity=0.7).encode(
#         alt.X(str(feature) + str(':O'), bin=alt.Bin(maxbins=50)),
#         alt.Y('count()', stack=None),
#         alt.Color('income:N')).properties(
#         title= ('Feature:' + feature))
#     return histogram


# ## This is where we call our function on our training feature table and create a plot 
# ## for each numeric feature in it. 

# figure_dict = dict()
# for feature in ___:
#     train_df_nan = train_df_nan.sort_values('income')
#     figure_dict.update({feature:plot_histogram(___,feature)})
# figure_panel = alt.vconcat(*figure_dict.values())
# figure_panel


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### 1.5 Explaining the Visualizations
# rubric={reasoning:2}
# 
# Based on the above visualizations, which features seem relevant for the given prediction task?

# YOUR ANSWER HERE

# ### 1.6 Separating feature vectors and targets  
# rubric={accuracy:2}
# 
# Create `X_train`, `y_train`, `X_test`, `y_test` from `train_df_nan` and `test_df_nan`. 

# In[ ]:


X_train = None
y_train = None
X_test = None
y_test = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test_1_6(X_train,X_test,y_train,y_test)


# ### 1.7 Training?
# rubric={reasoning:2}
# 
# 
# If you train [`sklearn`'s `SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) model on `X_train` and `y_train` at this point, would it work? Why or why not?

# YOUR ANSWER HERE

# ## Exercise 2: Preprocessing <a name="3"></a>
# <hr>

# In this exercise, you'll be wrangling the dataset so that it's suitable to be used with `scikit-learn` classifiers. 

# ### 2.1 Identifying transformations that need to be applied
# rubric={reasoning:7}
# 
# Identify the columns on which transformations need to be applied and tell us what transformation you would apply in what order by filling in the table below. Example transformations are shown for the feature `age` in the table.  
# 
# Note that for this problem, no ordinal encoding will be executed on this dataset. 
# 
# Are there any columns that you think should be dropped from the features? If so, explain your answer.
# 

# | Feature | Transformation |
# | --- | ----------- |
# | age | imputation, scaling |
# | workclass |  |
# | fnlwgt |  |
# | education |  |
# | education_num |  |
# | marital_status |  |
# | occupation |  |
# | relationship |  |
# | race |  |
# | sex |  |
# | capital_gain |  |
# | capital_loss |  |
# | hours_per_week |  |
# | native_country |  |

# YOUR ANSWER HERE

# ### 2.2 Numeric feature pipeline
# rubric={accuracy:2}
# 
# Let's start making our pipelines. Use `make_pipeline()` or `Pipeline()` to make a pipeline for the numeric features called `numeric_transformer`. 
# 
# Use `SimpleImputation()` with `strategy='median'`. For the second step make sure to use standardization with `StandardScaler()`.

# In[ ]:


numeric_transformer = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test_2_2(numeric_transformer)


# ### 2.3 Categorical feature pipeline
# rubric={accuracy:2}
# 
# Next, make a pipeline for the categorical features called `categorical_transformer`. 
# 
# Use `SimpleImputation()` with `strategy='most_frequent'`. 
# 
# Make sure to use the necessary one-hot encoding transformer with `dtype=int` and `handle_unknown="ignore"`.

# In[ ]:


categorical_transformer = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test_2_3(categorical_transformer)


# ### 2.4 ColumnTransformer
# rubric={accuracy:2}
# 
# Below we have defined a column transformer using [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html?highlight=columntransformer#sklearn.compose.ColumnTransformer) called `preprocessor` for the numerical, categorical, binary and dropped features.
# 
# Fill in the blank (`___`) so that the code executes and a ColumnTransformer transforms the appropriate feature with their transformations.
# 

# In[ ]:


# preprocessor = ColumnTransformer(
#     ___=[("numeric", ___, numeric_features),
#          ("categorical", ___, ___)           
#                  ], 
#     remainder='drop'    
# )

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test_2_4(preprocessor)


# ## Exercise 3: Building a Model <a name="4"></a>
# <hr>
# 
# Now that we have preprocessed features, we are ready to build models. 

# ### 3.1 Dummy Classifier
# rubric={accuracy:3}
# 
# It's important to build a dummy classifier to compare our model to. Make a `DummyClassifier` and make sure to train it and then score it on the training and test sets. 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### 3.2 Main pipeline
# rubric={accuracy:2}
# 
# Define a main pipeline that transforms all the different features and uses an `SVC` model with default hyperparameters. 
# 
# If you are using `Pipeline` instead of `make_pipeline`, name each of your steps `columntransformer` and `svc` respectively. 

# In[ ]:


main_pipe = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### 3.3 Hyperparameter tuning
# 
# rubric={accuracy:3}
# 
# Now that we have our pipelines and a model, let's tune the hyperparameters `gamma` and `C`. 
# 
# Sweep over the hyperparameters in `param_grid` using `RandomizedSearchCV` with a  `cv=5`, `n_iter=10` and setting `return_train_score=True` naming the object `random_search`. It also may be a good idea to set `random_state` here. Setting `verbose=3` will also give you information as the search is occurring. 
# 
# You will need to fit your `RandomizedSearchCV` object.
# 
# This step is quite demanding computationally so be prepared for this to take 2 or 3 minutes and your fan may start to run! 
# 

# In[ ]:


param_grid = {
    "svc__gamma": [0.1, 1.0, 10, 100],
    "svc__C": [0.1, 1.0, 10, 100]
}

random_search = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# Let's take a look at the scores below: 

# In[ ]:


### BEGIN STARTER CODE
pd.DataFrame(random_search.cv_results_)[["params",
                                         "mean_test_score", 
                                         "mean_train_score",
                                         "rank_test_score"]]
### END STARTER CODE


# ### 3.4 Choosing your hyperparameters
# 
# rubric={accuracy:2, reasoning:1}
# 
# What values for `gamma` and `C` would you choose for your final model and why? 
# 
# (You can answer this by either looking at the table above or using the methods `.best_params_`  and `.best_score_`)
# 

# In[ ]:


best_gamma = None 
best_c = None
best_score = None 

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# YOUR ANSWER HERE

# # 4. Evaluating on the test set <a name="5"></a>
# <hr>
# 
# Now that we have a best-performing model, it's time to assess our model on the test set. 

# ### 4.1 Scoring your final model
# rubric={accuracy:2}
# 
# What is the training and test score of the best scoring model? 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### 4.2 Assessing your model
# rubric={reasoning:2}
# 
# Finalize this report by comparing these scores with the validation scores from the randomized grid search in question 3.3. Compare your final model accuracy with your baseline model from question 3.1. How confident are you in your model? What were some of the downsides of using `SVC`? 
# 

# YOUR ANSWER HERE

# ### Submission to Canvas
# 
# **PLEASE READ: When you are ready to submit your assignment do the following:**
# 
# - Read through your solutions
# - **Restart your kernel and clear output and rerun your cells from top to bottom** 
# - Makes sure that none of your code is broken 
# - Verify that the tests from the questions you answered have obtained the output "Success"
# - Convert your notebook to .html format by going to File -> Export Notebook As... -> Export Notebook to HTML
# - Upload your `.ipynb` file and the `.html` file to Canvas under Assignment1. 
# - **DO NOT** upload any `.csv` files. 

# ### Congratulations on finishing Assignment 2! Now you are ready to build a simple ML pipeline on real-world datasets!
