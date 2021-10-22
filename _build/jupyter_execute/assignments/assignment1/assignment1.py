#!/usr/bin/env python
# coding: utf-8

# # BAIT 509 Assignment 1: An introduction to Decision Trees, $k$-NN, Cross-validation and ML Fundamentals
# 
# __Evaluates__: Lectures 1 - 3. 
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
#  Submit your assignment through [UBC Canvas](https://canvas.ubc.ca/courses/58082) by **11:59 pm Wednesday, April 28th**.

# ## Answering Questions
# 
# - Places that you see `raise NotImplementedError # No Answer - remove if you provide an answer`. Substitute the `None` above it and replace the `raise NotImplementedError # No Answer - remove if you provide an answer` with your completed code and answers, then proceed to run the cell!
# 
# - Any place you see `____`, you must fill in the function, variable, or data to complete the code.
# 

# In[ ]:


# Import libraries
import re
import sys
from hashlib import sha1

import numpy as np
import pandas as pd

# Visualization
import altair as alt
import matplotlib.pyplot as plt

# Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier

# Data splitting and model selection
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split

# Autograding
import tests_assignment1 as t


# ## Introduction and learning goals <a name="in"></a>
# <hr>
# 
# Welcome to the assignment! After working on this assignment, you should be able to:
# 
# - create `X` (feature vectors) and `y` (targets) from a given dataset.  
# - use the `fit` and `predict` paradigms in `sklearn`.
# - use the `score` method in `sklearn` to calculate classification accuracy. 
# - use `train_test_split` for data splitting and explain the importance of shuffling during data splitting. 
# - train a decision tree using `sklearn`.
# - build a decision tree classifier on a real-world dataset.
# - build a $k$-nn classifier and explore different hyperparameters.
# - discuss the relationship between train accuracy and test accuracy and overfitting.
# - Choose an appropriate hyperparameter value for your model.

# ### Exercise 1: Decision trees with a toy dataset <a name="1"></a>
# <hr>
# 
# Suppose you have three different job offers with comparable salaries and job descriptions. You want to decide which one to accept, and you want to make this decision based on which job is likely to make you happy. Being a very systematic person, you come up with three features associated with the offers, which are important for your happiness: whether the colleagues are supportive, work-hour flexibility, and whether the company is a start-up or not (the columns `SupportiveRUS`, `Flexiblez` and `SadStartup` respectively). 

# In[ ]:


offer_data = {
    # Features
    "SupportiveRUS": [1, 0, 0],
    "Flexiblez": [0, 0, 1],
    "SadStartup": [0, 1, 1],
    # Target
    "target": ["?", "?", "?"],
}

offer_df = pd.DataFrame(offer_data)
offer_df


# Next, you ask the following questions to some of your friends (who you think have similar notions of happiness) regarding their jobs:
# 
# 1. Do you have supportive colleagues? (1 for 'yes' and 0 for 'no')
# 2. Do you have flexible work hours? (1 for 'yes' and 0 for 'no')
# 3. Do you work for a start-up? (1 for 'start up' and 0 for 'non start up')
# 4. Are you happy with your job? (happy or unhappy)
# 
# You get the following data from this survey. You want to train a machine learning model using this data and then use this model to predict which job is likely to make you happy. 

# In[ ]:


happiness_data = {
    # Features
    "SupportiveRUS": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    "Flexiblez": [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    "SadStartup": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    # Target
    "target": [
        "happy",
        "happy",
        "happy",
        "unhappy",
        "unhappy",
        "happy",
        "happy",
        "unhappy",
        "unhappy",
        "unhappy",
    ],
}

train_df = pd.DataFrame(happiness_data)
train_df


# ### 1.1 Decision stump by hand 
# rubric={autograde:2}
# 
# **Your tasks:**
# 
# If you built a decision stump (decision tree with only 1 split) by splitting on the condition `SupportiveRUS == 1` by hand, how would you predict each of the employees? 
# 
# Save your prediction for each employee as a string element in a list named `predict_employees`. 
# example:
# ```
# predict_employees = ['happy', 'unhappy', 'unhappy',  'unhappy', 'unhappy', 'happy', 'happy', 'happy',  'unhappy',  'unhappy'] 
# ```
# 
# (Note: you do not need to use a model here. By looking at the target column and the feature `SupportiveRUS` what rows would you predict with which labels?) 

# In[ ]:


predict_employees = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test1_1(predict_employees)


# ### 1.2 Decision stump accuracy
# 
# rubric={autograde:2}
# 
# What training accuracy would you get with this decision stump above?
# 
# Save the accuracy as a decimal in an object named `supportive_colleagues_acc`. 

# In[ ]:


supportive_colleagues_acc = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test1_2(supportive_colleagues_acc)


# ### 1.3 Create `X`, `y`
# rubric={mechanics:2}
# 
# Recall that in `scikit-learn` before building a classifier we need to create `X` (features) and `y` (target). 
# 
# **Your tasks:**
# 
# From `train_df`, create `X` and `y`; save them in objects named `X` and `y`, respectively. 

# In[ ]:


X = None
y = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test1_3(X, y)


# ### 1.4 `fit` a decision tree classifier 
# rubric={accuracy:2}
# 
# The idea of a machine learning algorithm is to *fit* the best model on the given training data, `X` (features) and `y` (their corresponding targets) and then using this model to *predict* targets for new examples. 
# 
# **Your tasks:**
# 
# `fit` `sklearn`'s decision tree model on this toy dataset. 
# Build a decision tree named `toy_tree` and fit it on the toy data using `sklearn`'s [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

# In[ ]:


toy_tree = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test1_4(toy_tree)


# ### 1.5 `score` 
# rubric={accuracy:2}
# 
# Score the decision tree on the training data (`X` and `y`).
# Save the results in an object named `toy_score`. 
# 

# In[ ]:


toy_score = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test1_5(toy_score)


# ### 1.6 Explain training score
# rubric={reasoning:2}
# 
# Do you get perfect training accuracy? Why or why not? 

# YOUR ANSWER HERE

# ### 1.7 Getting features
# 
# rubric={accuracy:2}
# 
# The first `offer_df` dataframe has no target values and we want to use the model we just made to make predictions. 
# Drop the column `target` from the object and rename this dataframe `test_X`. 

# In[ ]:


test_X = None 

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test1_7(test_X)


# ### 1.8 `predict`
# rubric={accuracy:2}
# 
# Now make predictions on the jobs offered in `test_X`. Save the predictions in an object named `predicted`. 
# 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test1_8(predicted)


# ### 1.9 Happy job
# rubric={reasoning:2}
# 
# Looking at the predictions, in which job you are likely to be happy? (answer in 1-2 sentences)

# YOUR ANSWER HERE

# ## Exercise 2: Decision trees on a real dataset <a name="2"></a>
# <hr>

#  ### Introducing the Spotify Song Attributes dataset
#  
#  
# For the rest of the assignment, you'll be using Kaggle's [Spotify Song Attributes](https://www.kaggle.com/geomack/spotifyclassification/home) dataset.
# The dataset contains a number of features of songs from 2017 and a binary target variable representing whether the user liked the song or not. See the documentation of all the features [here](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/). The supervised machine learning task for this dataset is predicting  whether the user likes a song or not given a number of song features.
# 
# This dataset is publicly available on Kaggle, but not licensed to be freely distributed. So we do not provide this dataset, and you will have to download it yourself and add it to the same folder as this assignment. Follow the steps below to get the data CSV. 
# 
# - If you do not have an account with Kaggle, you will first need to create one. (It's free.) 
# - Login to your account and [download the data](https://www.kaggle.com/geomack/spotifyclassification/downloads/spotifyclassification.zip/1) the dataset;  
# - (You should always) Read the [terms and conditions](https://www.kaggle.com/terms) before using the data.
# - Save the CSV in the same folder as you saved this file and the `test_assignment1.py` file. (You DO NOT submit this `.csv` file)   

# The starter code below reads the data CSV file into the notebook. make sure you named the csv file `spotify.csv`

# In[ ]:


### BEGIN STARTER CODE

spotify_df = pd.read_csv("spotify.csv", index_col=0)
spotify_df.head()
### END STARTER CODE


# ### 2.1 Split your data
# rubric={accuracy:2}
# 
# Split your `spotify_df` into your train and test splits.  Name the training data `train_df` and the testing data `test_df` using an 80/20 train to test split. Set your `random_state` to 77 to pass the test. 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# In[ ]:


t.test2_1(train_df, test_df)


# The starter code below produces a histogram for the values in the feature `danceability`.
# 
# *Note: I am using Altair here, If you wish to learn more please see the resources I posted [here](https://bait509-ubc.github.io/BAIT509/things_to_know/how.html#visualizations). You do not need to fully understand the code here.*

# In[ ]:


histogram = alt.Chart(train_df).mark_bar(opacity=0.7).encode(
    alt.X('danceability', bin=alt.Bin(maxbins=50)),
    alt.Y('count()', stack=None)).properties(
    title= 'Distribution for all danceability values')
histogram


# Instead, let's make a separate plot a look at the `danceability` distribution for songs that are disliked vs liked (a `target` value of `1` vs `0`). 

# In[ ]:


histogram_facet = alt.Chart(train_df).mark_bar(opacity=0.7).encode(
    alt.X('danceability', bin=alt.Bin(maxbins=50)),
    alt.Y('count()', stack=None)).facet('target').properties(
    title= 'Distribution for separate target values values')
histogram_facet


# It might be easier to compare the 2 distributions if they were layered on top of each other so that we can see whether danceability tends to be any different for disliked vs. liked songs.

# In[ ]:


histogram_overlayed = alt.Chart(train_df.sort_values(by='target')).mark_bar(opacity=0.6).encode(
    alt.X('danceability', bin=alt.Bin(maxbins=50)),
    alt.Y('count()', stack=None),
    alt.Color('target:N')).properties(
    title= 'Distribution for separate target values values')
histogram_overlayed


# Here we can see in the data there are more songs with danceability values between 0.70-0.86 with `target=1` and more songs with 0.38-0.62 with a `target=0`. We can somewhat see how well this feature could help a potential model make predictions. For example, a decision boundary could be: if danceability > 0.7 then predict like (`1`), else predict dislike (`0`)”

# ### 2.2 Plotting histograms 
# 
# rubric={accuracy:3}
# 
# Take the code below that we started for you and fill in the blank areas (`____`) so that the code produces histograms for the following features (in order) that show the distribution of the feature values, separated for 0 and 1 target values. 
# 
# - `acousticness`
# - `tempo`
# - `instrumentalness`
# - `energy`
# - `valence`
# 

# In[ ]:


# def plot_histogram(df,feature):
#     """
#     plots a histogram of a decision trees feature
     
#     Parameters
#     ----------
#     feature: str
#         the feature name
#     Returns
#     -------
#     altair.vegalite.v3.api.Chart
#         an Altair histogram 
#     """
#     histogram = alt.Chart(df.sort_values(by='target')).mark_bar(
#         opacity=0.7).encode(
#         alt.X(feature, bin=alt.Bin(maxbins=50)),
#         alt.Y('count()', stack=None),
#         alt.Color('target:N')).properties(
#         title= str.title(feature))
#     return histogram

# feature_list = ____
# figure_dict = dict()
# for feature in ____ :
#     figure_dict.update({feature:plot_histogram(____,feature)})
# figure_panel = alt.vconcat(*figure_dict.values())
# figure_panel


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### 2.3 Explaining histograms 
# 
# rubric={reasoning:3}
# 
# Answering in 1-2 sentences, which features and split values might be useful in differentiating the target classes?

# YOUR ANSWER HERE

# ## Exercise 3: Cross-validation and model building <a name="3"></a>
# <hr>
# Recall that in machine learning what we care about is generalization; we want to build models that generalize well on unseen examples. One way to ensure this is by splitting the data into training data and test data, building and tuning the model only using the training data, and then doing the final assessment on the test data. 

# We've provided you with some starter code that separates `train_df` and `test_df` into their respective features and target objects. We removed the columns `song_title` and `artist` from the feature objects since they need additional processing to use them in our model. 

# In[ ]:


### BEGIN STARTER CODE

X_train = train_df.drop(columns = ['song_title', 'artist','target'])
y_train = train_df['target']
X_test = test_df.drop(columns = ['song_title', 'artist','target'])
y_test = test_df['target']

### END STARTER CODE


# ### 3.1 Building a Dummy Classifier
# rubric={accuracy:3}
# 
# Build a `DummyClassifier` using the strategy `most_frequent`.
# 
# Train it on `X_train` and `y_train`. Score it on the train **and** test sets.

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### 3.2 Building a Decision Tree Classifier
# rubric={accuracy:3}
# 
# Build a Decision Tree classifier without setting any hyperparameters. Cross-validate with the appropriate objects, passing `return_train_score=True` and setting the number of folds to 10. (See the note in lecture 2 for help).
# 
# Display the scores from `.cross_validate()` in a dataframe. 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### Question 3.3 Decision Tree training and validation scores
# rubric={accuracy:1, reasoning:1}
# 
# What are the mean validation and train scores? In 1-2 sentences, explain your results. Is your model overfitting or underfitting? 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# YOUR ANSWER HERE

# ### 3.4 Building a $k$-NN Classifier
# rubric={accuracy:3}
# 
# Build a $k$-NN classifier using the default hyperparameters. Cross-validate with the appropriate objects, passing `return_train_score=True` and setting the number of folds to 10.
# 
# Display the scores from `.cross_validate()` in a dataframe. 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### Question 3.5 $k$-NN training and validation scores 
# rubric={accuracy:1, reasoning:1}
# 
# What are the mean validation and train scores for your $k$-NN classifier? In 1-2 sentences, explain your results.

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# YOUR ANSWER HERE

# ### 3.6 Compare the models
# rubric={reasoning:2}
# 
# In 1-2 sentences, compare the 3 models.

# 
# YOUR ANSWER HERE

# ## Exercise 4: Hyperparameters <a name="5"></a>
# <hr>
# 
# We explored the `max_depth` hyperparameter of the `DecisionTreeClassifier` in lecture 2 but in this assignment, you'll explore another hyperparameter, `min_samples_split`. See the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) for more details on this hyperparameter.

# ## 4.1 `min_samples_splits`
# rubric={accuracy:5}
# 
# Using 10-fold cross-validation and the training set only, find an appropriate value within the range 5 to 105 for the `min_samples_split` hyperparameter for a decision tree classifier.
# 
# For each `min_samples_split` value:
#     - Create a `DecisionTreeClassifier` object with the `min_samples_split` value.
#     - Run 10-fold cross-validation with this `min_samples_split` using `cross_validate` to get the mean train and validation accuracies. Remember to use `return_train_score` argument to get the training score in each fold. 
# 
# In a pandas dataframe, for each `min_samples_split` show the mean train and cross-validation score. 
# 
# *Hint: We did something similar in lecture 2 (under **The "Fundamental Tradeoff" of Supervised Learning**) which you can refer to if you need help.* 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### 4.2 Plotting and interpreting
# rubric={accuracy:3, viz:1}
# 
# Using whatever tool you like for plotting,  make a plot with the `min_samples_split` of the decision tree on the *x*-axis and the accuracy on the train and validation sets on the *y*-axis. 
# 
# (Again we did this in lecture 2 if you need any guidance)

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# ### 4.3 Picking `min_samples_split`
# rubric={accuracy:1, reasoning:2}
# 
# Based on your results from 4.2, what `min_samples_split` value would you pick in your final model? In 1-2 sentences briefly explain why you chose this particular value.

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


# YOUR ANSWER HERE

# ### 4.4 Final model
# rubric={accuracy:2,reasoning:1}
# 
# Train a decision tree classifier with the best `min_samples_split` using `X_train` and `y_train` and now carry out a final assessment and then obtain the test score on the test set. In one sentence comment on your results. 
# 

# In[ ]:


# your code here
raise NotImplementedError # No Answer - remove if you provide an answer


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

# ### Congratulations on finishing Assignment 1! Well done! 
