#!/usr/bin/env python
# coding: utf-8

# # Multi-Class, Pandas Profiling, Finale 
# 
# *Hayley Boyce, May 19th, 2021*

# In[1]:


# Importing our libraries
import pandas as pd
import altair as alt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

import sys
sys.path.append('code/')
from display_tree import display_tree
from plot_classifier import plot_classifier
import matplotlib.pyplot as plt

# Preprocessing and pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import  plot_confusion_matrix, classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

import scipy
from sklearn.model_selection import RandomizedSearchCV


# ## House Keeping 
# 
# - Our last class 😭!
# - Assignment due at 11:59pm tonight!
# - Project time today due on Wednesday May 26th at 11:59pm 
# - Extra office hours on Tuesday(12:00pm)? -> Poll
# - [Teaching evaluations](https://canvas.ubc.ca/courses/30777/external_tools/6073) - I know you all are very busy, but I would be super appreciative if you could fill it out for me. ❤️ 
# <img src="imgs/appreciate.png"  width = "30%" alt="404 image" />
# 

# ## Lecture Learning Objectives 
# 
# - Explain components of a confusion matrix with respect to multi-class classification.
# - Define precision, recall, and f1-score with multi-class classification
# - Carry out multi-class classification using OVR and OVO strategies.

# ## Five Minute Recap/ Lightning Questions 
# 
# - What metrics is calculated using the equation $\frac{TP}{TP + FP}$ ?
# - What function can be used to find the calculated values of precision, recall, and f1? 
# - What function do we use to identify the number of false positives, false negatives and correctly identified positive and negative values? 
# - What argument/parameter is important to use when we make our own scorer where lower values are better? 
# - What regression metric will give funky units? 

# ### Some lingering questions
# 
# - What happens if we have data where there is a lot of one class and very few of another?
# - How do we measure precision and recall and what do our confusion matrices look like now

# ## Multi-class classification
# 
# - Often we will come across problems where there are more than two classes to predict.
# - We call these multi-class problems.

# - Some algorithms can natively support multi-class classification, for example:
#     - Decision Trees
#     - $K$-nn
#     - Naive Bayes
# - Below is an example of a Decision Tree Classifier used to classify 3 labels

# <img src='imgs/multi_class_dt.png' width="60%">

# And here's the graph:
# 
# <img src='imgs/multi_class_dt_graph.png' width="450">

# - Here's an example of KNN:
# 
# <img src='imgs/multi_class_knn.png' width="550">

# Other models, like SVMs and Logistic Regression, don't natively support multi-class classification.
# 
# Instead, there are two common strategies to help us:
# - One-vs-rest
#  - One-vs-one

# ### One-vs-Rest
# (also known as one-vs-all)
# 
# - It's the default for most sklearn algorithms, e.g., LogisticRegression, SVM.
# - Turns $k$-class classification into $k$ binary classification problems.
# - Builds $k$ binary classifiers; for each classifier, the class is fitted against all the other classes.
# - For *k* classes, that means we need *k* models in total, e.g.:
#     - blue vs (red & orange)
#     - red vs (blue & orange)
#     - orange vs (blue & red)
# - We use all models to make a prediction, and then choose the category with the highest prediction/probability/confidence.
# - You can do this yourself for any binary classifier using [`OneVsRestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)

# Here we are importing `OneVsRestClassifier` from `sklearn.multiclass`

# In[2]:


from sklearn.multiclass import OneVsRestClassifier


# We are going to use a wine dataset that has 3 different classes; 0, 1, 2 (maybe red, white and rose?)

# In[3]:


data = datasets.load_wine()
X = pd.DataFrame(data['data'], columns=data["feature_names"])
X = X[['alcohol', 'malic_acid']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)
X_train.head()


# In[4]:


pd.DataFrame(y_train).value_counts()


# In[5]:


ovr = OneVsRestClassifier(LogisticRegression(max_iter=100000))
ovr.fit(X_train, y_train)
ovr.score(X_train, y_train)


# In[7]:


#plot_classifier(X_train, y_train, ovr);


# ### One-vs-One
# 
# - One-vs-One fits a model to all pairs of categories.
# - If there are 3 classes ("blue", "red", "orange"), we fit a model on:
#     - blue vs red
#     - blue vs orange
#     - red vs orange
# - So we have 3 models in this case, or in general $\frac{n*(n-1)}{2}$
# - For 100 classes, we fit 4950 models!
# - All models are used during prediction and the classification with the most “votes” is predicted.
# - Computationally expensive, but can be good for models that scale poorly with data, because each model in OvO only uses a fraction of the dataset.

# In[8]:


from sklearn.multiclass import OneVsOneClassifier


# In[9]:


ovo = OneVsOneClassifier(LogisticRegression(max_iter=100000))
ovo.fit(X_train, y_train)
ovo.score(X_train, y_train)


# In[11]:


#plot_classifier(X_train, y_train, ovo);


# ## Multi-class measurements
# 
# Similar to how we can use different classification metrics for binary classification, we can do so with multi-class too. 
# 
# Let's look at this with a larger version of this wine dataset. 

# In[12]:


data = datasets.load_wine()
X = pd.DataFrame(data['data'], columns=data["feature_names"])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)
X_train.head()


# In[13]:


X_train.info()


# Since our data here isn't missing any values and it's all numeric, we can make a pipeline with just `StandardScaler()` and a model, we are going to use `LogisticRegression`.

# In[14]:


pipe = make_pipeline(
    (StandardScaler()),
    (LogisticRegression())
)
pipe.fit(X_train,y_train);


# In[15]:


predictions = pipe.predict(X_test)
pipe.score(X_test,y_test)


# We can predict on our test set and see that we get an accuracy of 93%. 
# 
# But what does this mean for our metrics?

# ### Multiclass confusion metrics
# 
# We can still create confusion matrices but now they are greater than a 2 X 2 grid. 
# 
# We have 3 classes for this data, so our confusion matrix is 3 X 3. 

# In[16]:


plot_confusion_matrix(pipe, X_test, y_test, cmap='Greys');


# We see that we can still compute a confusion matrix, for problems with more than 2 labels in the target column. 
# 
# The diagonal values are the correctly labelled wines and the rest are the errors.
# 
# Here we can see the model mistakenly predicted:
# 
# - 1 wine of true class 1 as class 0 and,
# - 1 wine of true class 1 as class 2. 
# - 1 of the wines with a class of 2 as class 1. 

# ### Multiclass classification report
# 
# Precision, recall, etc. don't apply directly but like we said before, but depending on which class we specify as our "positive" label and consider the rest to be negative, then we can.

# In[17]:


print(classification_report(y_test, predictions, digits=4))


# If class `0` is our positive class then our precision, recall and f1-scores are 0.95, 1.00, 0.9744 respectively. 
# 
# If class `1` is our positive class then now the precision, recall and f1-scores are 0.9375, 0.8824, 0.9091. 
# 
# And finally, if class `2` is our positive class then the precision, recall and f1-scores are 0.8889, 0.8889, 0.8889. 
# 
# Again the `support` column on the right shows the number of examples of each wine class. 

# ## Multi-class coefficients 
# 
# Let's look at the coefficients with this multi-class problem. (Ignore the `max_iter` for now. You can look into it [here](https://medium.com/analytics-vidhya/a-complete-understanding-of-how-the-logistic-regression-can-perform-classification-a8e951d31c76) if you like)

# In[18]:


pipe.named_steps['logisticregression'].coef_


# In[19]:


pipe.named_steps['logisticregression'].coef_.shape


# What is going on here?
# 
# Well, now we have one coefficient per feature *per class*.
# 
# The interpretation is that these coefficients contribute to the prediction of a certain class. 
# 
# The specific interpretation depends on the way the logistic regression is implementing multi-class (OVO, OVR).

# ## Multi-class and `predict_proba` 
# 
# If we look at the output of `predict_proba` you'll also see that there is a probability for each class and each row adds up to 1 as we would expect (total probability = 1).

# In[20]:


pipe.predict_proba(X_test)[:5]


# ## Let's Practice 
# 
# 1\. Which wrapper is more computationally expensive?    
# 2\. Name a model that can handle multi-class problems without any issues or needing any additional strategies.    
# 3\. If I have 6 classes, how many models will be built if I use the One-vs-Rest strategy?    
# 4\. If I have 6 classes, how many models will be built if I use the One-vs-One strategy?    
# 
# Use the diagram below to answer the next few questions:    
# <img src="imgs/multi-classQ.png"  width = "70%" alt="404 image" />
# 
# 5\. How many examples did the model correctly predict?     
# 6\. How many examples were incorrectly labelled as `G`?    
# 7\. How many `F-C` labels were in the data?     
# 
# **True or False:**      
# 
# 8\. Decision Trees use coefficients for multi-class data.      
# 9\. Using 1 target label as the positive class will make all other target labels negative.   

# ```{admonition} Solutions!
# :class: dropdown
# 
# 1. One-vs-One
# 2. Decision Trees, K-nn
# 3. 6
# 4. $6(5)/2=15$
# 5. 52
# 6. 3
# 7. 6 
# 8. False
# 9. True
# 
# ```

# ## Pandas Profiler 
# 
# - EDA secret! (Careful to only use this on your training split though -> Golden Rule!) 
# - quickly generate summaries of dataframes including dtypes, stats, visuals, etc.
# - [Pandas profiling](https://github.com/pandas-profiling/pandas-profiling) is not part of base Pandas
# - If using conda, install with: `conda install -c conda-forge pandas-profiling`

# In[21]:


import pandas as pd
from pandas_profiling import ProfileReport


# In[22]:


df = pd.read_csv('data/housing.csv')


# In[24]:


profile = ProfileReport(df)
profile


# ## Project time 
# 
# - Off to your groups!
# - I'll be here to answer any questions in the main room.

# ## Final Remarks 
# 
# - Course evaluation. It would help me immensely if you could fill out the course evaluation. 
# 
# - It's been wonderful to teach you all! This was my first synchronous class and I've really enjoyed getting to know you all. Thank you so much! 
# 
# 
# 
# <img src="imgs/grateful.png"  width = "40%" alt="404 image" />
# 
# <img src="imgs/logoff.png"  width = "40%" alt="404 image" />

# ## What We've Learned Today
# 
# - How to carry out multi-class classification.
# - How to utilize pandas profilier for EDA. 
# - How great it was to teach everyone! 
