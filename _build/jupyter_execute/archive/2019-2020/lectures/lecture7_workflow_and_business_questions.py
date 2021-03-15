# BAIT 509: Business Applications of Machine Learning
## Lecture 7 - Machine learning workflow and forming good machine learning questions from business questions
Tomas Beuzen, 27th January 2020

# Lecture outline
- [0. Recap (5 mins)](#0)
- [1. Lecture learning objectives](#1)
- [2. Forming statistical questions to answer business objectives](#2)
- [--- Break --- (10 mins)](#break)
- [3. Machine learning pipelines (advanced material - only if we have time)](#3)
- [4. Final project](#4)

# Announcements

- Final project released! We'll talk about it in today's lecture (due **Friday 14th February, 11:59pm**)
- Assignment 3 will be released tomorrow (due **Friday 7th February, 11:59pm**)
- Assigment 1 grades will be released tomorrow
- A few students have mentioned an issue with their computer running out of memory when using CountVectorizer and fitting an estimator
    - This will usually happen if you try to coerce your vectorized data to an array using `.toarray()` (or similar)
    - By default, `CountVectorizer` returns a sparse matrix and you should leave the data in this form!
    - Sparse matrices are a more memory-efficient way of storing data
    
<img src='./img/sparse.png' width="600"> 

# 0. Recap  (5 mins) <a id=0></a>

- Feature selection: how to select features that are important for your models
- Model selection: how to choose the best model for your problem
- Advanced hyperparameter tuning: how to use `GridSearchCV` to efficiently tune our models

# 1. Lecture learning objectives <a id=1></a>

- Distilling supervised learning questions form business questions/objectives
- Describe the typical ML workflow/pipeline
- Use the sklearn function `pipeline` to create an ML pipeline

# 2. Forming statistical questions to answer business objectives (30 mins) <a id=2></a>

Today, we’re going to take a break from supervised learning methods, and look at the process involved to use machine learning to address a question/problem faced by an organization.

Generally, there are four parts of a machine learning analysis. In order from high to low level:

1. **The business question/objective**
2. **The statistical question/objective**
3. **The data and model**
4. **The data product**

Doing a machine learning analysis is about distilling from the highest level to the lowest level. As such, there are three distillations to keep in mind: 1-2, 2-3, and 3-4:

- **1-2 is about asking the right questions**
- **2-3 is about building a useful model**
- **3-4 is about communicating the results**
    
Note that an analysis isn’t a linear progression through these “steps”; rather, the process is iterative. This is because none of the components are independent. Making progress on any of the three distillations gives you more information as to what the problem is.

We’ll look at each of these distillations in turn.

## 2.1 (1 - 2) Asking useful statistical questions

- Usually, a company is not served up a machine learning problem, complete with data and a description of the response and predictors.
- Instead, they’re faced with some high-level objective/question that we’ll call the **business question/objective**
- This question needs refining to a **statistical question/objective** – one that is directly addressable by machine learning.

### 2.1.1 Business objectives: examples
- This [altexsoft blog post](https://www.altexsoft.com/blog/business/supervised-learning-use-cases-low-hanging-fruit-in-data-science-for-businesses/) is a great introduction to business use cases of data science/ML
- Examples of business objectives (for which machine learning is a relevant approach)
    - Reduce the amount of spam email received
    - Early prediction of product failure
    - Find undervalued mines
    - Make a transit system more efficient
    - Hire efficient staff

### 2.1.2 Refining business objectives to statistical objectives
- Statistical objectives need to be specific
- Remember that supervised learning is about predicting a response $Y$ from predictors $X_1,…,X_p$
- So we need to refine our business objectives to a statistical question(s) we can answer
- This typically involves:
    - Identifying the **response variable** ($Y$) that is most aligned with the business objective.
    - Identifying the **data** (observations + features) that will be used for model development/testing.
    - Note: part of this is the task of feature selection (a topic that we've covered briefly) – but, this is also largely, a human decision based on what we think is more informative, as well as a resource questions (what data is actually available?)

### 2.1.3 Statistical objectives: examples
Statistical objectives corresponding to the above business objective examples might be:

| Business Objective | Statistical Question |
| :--- | :--- |
| Reduce the amount of spam email received | <ul><li>$Y$ = classifying an email as spam/not spam <li> $X$ = words present in name and body of email and other metadata (sender email, time, etc.) <li> Cases of spam will be gathered over time as employees identify emails as spam/not spam. Model improved in future as misclassifications are encountered.</ul>
| Early prediction of product failure | <ul><li>$Y$ = classifying a product as faulty/not faulty <li> $X$ = Relevant features chosen by expert <li> Data obtained from test facility</ul>
| Find undervalued mines | <ul><li>$Y$ = total volume of gold and silver at a site <li> $X$ =  concentrations of other minerals found in drill samples, geographic information, historical data, etc <li> Data obtained from mines where total volumes are already known</ul>
| Make a transit system more efficient | <ul><li>$Y$ = predict the time it takes a bus to travel between set stops <li> $X$ = time of day/week/year, weather, etc. <li> Use data from company server tracking bus movements</ul>
| Hire efficient staff | <ul><li>$Y$ = predict monthly sales <li> $X$ = a personality test, years of work experience, field of experience, etc. <li> Use data based on current employees</ul>

### 2.1.4 Statistical questions are not the full picture!
- Almost always, the business objective is more complex than the statistical objective
- By refining a business objective to a statistical one, we may lose part of the essence of the business objective.
- It’s important to have a sense of the ways in which your statistical objective falls short, and the ways in which it’s on the mark, so that you keep a sense of the big picture.
- For example, predicting whether a new staff hire will be efficient or not is a useful statistical question, but doesn't consider why a company might be attracting certain applicants, how long staff will remain, how staff work together, etc.

### 2.1.5 Statistical objectives unrelated to supervised learning
- We are only focussing on statistical questions related to supervised learning and prediction in this course
- But there are other kinds of questions you can ask too
- Consider the following example

**Business objective**: To gain insight into the productivity of two branches of a company.

Examples of statistical questions:

- **Hypothesis testing**: Is the mean number of sick days per employee different between two branches of a company?
    - Supervised learning doesn’t cover testing for differences.
- **Unsupervised learning**: What is the mean sentiment of the internal discussion channels from both branches?
    - There is no data of feature + response here, as required by supervised learning (by definition).
- **Statistical inference**: Estimate the mean difference in monthly revenue generated by both branches, along with how certain you are with that estimate.
    - Supervised learning typically isn’t concerned about communicating how certain your estimate is.

## 2.2 (2 - 3) Building a useful model
- This is really the main focus of this course
- This involves using ML algorithms (kNN, loess, decision trees, etc) to build a predictive model from data
- One piece of advice here is always start with/include a null model (mean/mode prediction) and a simple model (linear regress/logistic regression)
- Oftentimes a simple model does as well as a more complex approach! Or at the very least can help guide you on what more complex approaches to take next.

## 2.3 (3 - 4) Communicating results
- So you've distilled your business objectives to a statistical question
- You've developed a model to answer the statistical question
- Now your model needs to be delivered and used by others (or your future self)!
- The final delivery is often called "the data product" because it may conist of a variety of things:
    - a report
    - a presentation
    - an app
    - a software package/pipeline
- Sometimes the client requests a specific data product
- But note that their suggestion might not always be the best option. Perhaps they request a report and presentation communicating your findings, when a more appropriate product also includes an interactive app that allows them to explore your findings for themselves.
- Either way, the key here is communication. Two import challenges (relevant to your final project):
    - Using appropriate language: there is a lot of jargon in ML, the key is to talk more about the output and the general idea of your model(s), but not machine learning or statistical jargon.
    - Communication with visual design: this is about choosing what visuals are the most effective for communicating.
- Usually, the first step is to set up a framework of your data product. For a report, this means outlining what you intend to write about, and where.
- Showing it to your client is useful as a sanity check to see that you’re about to produce something that the client currently sees as being potentially useful.

# -------- Break (10 mins) -------- <a id="break"></a>

# 3. Machine learning pipelines (advanced material - only if we have time) (15 mins) <a id=3></a>

- So far we've learned about several key steps in ML development:
    - Feature pre-processing
    - Feature selection
    - Model selection (algorithm choice and hyperparameter tuning)
- We've also learned that it's important to keep our training and testing datasets separate (**the golden rule**)
- But implementing and remembering all these steps in a suitable way can quickly become unwieldy
- [Pipelines](https://scikit-learn.org/stable/modules/compose.html) exists to help with this issue
- The most relevant function for you is sklearn's [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline)
- It will be easiest to see by example
- I am going to use the twitter-airline-sentiment dataset we used previously to demonstrate a ML pipeline
- We will do the following steps:
    - Vectorise the data with `CountVectorizer()`
    - Select the best k features
    - Select between a Logistic Regression model and Naive Bayes model (including tuning hyperparameters)

- First let's load in the packages we need for this demo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline # <--- this is the pipeline function!

- Now let's load in the twitter airline data and split it into train/test portions

df = pd.read_csv('data/twitter-airline-sentiment.csv')
X = df['tweet']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=123)

- Okay, so we are going to set-up our workflow using a pipeline
- Pipelines are simply objects that contain all of our modelling steps in one place
- They usually comprise pre-processing techniques, feature selection techniques, and the last step is always a ML algorithm
- We enter steps into the pipeline as `(name, estimator)` pairs
- The below code sets up a pipeline for our problem, it contains:
    - `('vectorizer', CountVectorizer(stop_words='english')`: transforming our words into numeric vectors
    - `('selector', SelectKBest(chi2, k=5))`: selecting our best features (we've entered k=5 for now, but it's just a placeholder that will vary in the pipeline)
    - `('classifier', LogisticRegression(solver='lbfgs'))`: our ML algorithm (we've entered LogisticRegression for now, but it's just a palceholder that will vary in the pipeline)

pipe = Pipeline([('vectorizer', CountVectorizer(stop_words='english')),
                 ('selector', SelectKBest(chi2, k=5)),
                 ('classifier', LogisticRegression(solver='lbfgs'))])

- The next thing we need to do is define our search space
- This search space contains all the hyperparameters and models that we want to search through
- Let's talk through the following dictionary
- Notably, to change the hyperparameters of these different steaps, we need to use double-underscore notation `__`

search_space = [{'selector__k': [5, 10, 20, 100]},
                {'classifier': [LogisticRegression(solver='lbfgs')],
                 'classifier__C': [0.01, 0.1, 1.0]},
                {'classifier': [MultinomialNB()],
                 'classifier__alpha': [0.01, 0.1, 1, 10]}]

- Now we are ready to use `GridSearchCV` to look through our list of hyperparameters!
- We will first create the object

clf = GridSearchCV(pipe, search_space, cv=5, verbose=0)

- Now we will fit the GridSearchCV with our training data

best_model = clf.fit(X_train, y_train)

- Finally we can access the various results of our model using its attributes
- Let's find what the best model was:

best_model.best_estimator_.get_params()['classifier']

- Now let's find what the best number of features (`k`) was:

best_model.best_params_

- We can also find the best cross-validation score that our 'best combination' achieved

print(f"Best model error = {1 - best_model.best_score_:.2f}")

- Now the wonderful thing here is that everything is contained within our `best_model` object
- We can use it to make predictions on new data and it will apply the exact same steps (preprocessing, feature selection, optimum model) to that data
- Let's give it a try!
- First let's get the error on the test data
- Remember that the test data is still in its raw form of tweets

X_test.head()

- But all the necessary preprocessing techniques are built into our `best_model` object
- So we can use it to directly test our test data

print(f"Best model error = {1 - best_model.score(X_test, y_test):.2f}")

- We can also pass in some arbitrary text data to predict!

best_model.predict(['this is some text data'])

# 4. Final project (40 mins) <a id=4></a>

For today’s practical component, you’ll be working with your final project team to start your final projects for this course. Use this time to:

- Meet with your team mates;
- Think about a project - choose the data and business objective.
- Propose a statistical objective to address this.
- Also, elaborate on the statistical objective. What’s your plan for the analysis?