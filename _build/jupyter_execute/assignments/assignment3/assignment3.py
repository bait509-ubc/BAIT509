# BAIT 509 Assignment 3: Logistic Regression and Evaluation Metrics  

__Evaluates__: Lectures 7 - 9. 

__Rubrics__: Your solutions will be assessed primarily on the accuracy of your coding, as well as the clarity and correctness of your written responses. The MDS rubrics provide a good guide as to what is expected of you in your responses to the assignment questions and how the TAs will grade your answers. See the following links for more details:

- [mechanics_rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_mech.md): submit an assignment correctly.
- [accuracy rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_accuracy.md): evaluating your code.
- [autograde rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_autograde.md): evaluating questions that are either right or wrong.
- [reasoning rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_reasoning.md): evaluating your written responses.

## Tidy Submission 
rubric={mechanics:2}

- Complete this assignment by filling out this jupyter notebook.
- You must use proper English, spelling, and grammar.
- You will submit two things to Canvas:
    1. This jupyter notebook file containing your responses ( an `.ipynb` file); and,
    2. An `.html` file of your completed notebook (use `jupyter nbconvert --to html_embed assignment.ipynb` in the terminal to generate the html file or under `File` -> `Export Notebook As` -> `HTML`).
    
 <br>  

 Submit your assignment through [UBC Canvas](https://canvas.ubc.ca/courses/58082) by **11:59 pm Monday, May 19th**.

## Answering Questions

- Places that you see `raise NotImplementedError # No Answer - remove if you provide an answer`. Substitute the `None` above it and replace the `raise NotImplementedError # No Answer - remove if you provide an answer` with your completed code and answers, then proceed to run the cell!

- Any place you see `____`, you must fill in the function, variable, or data to complete the code.


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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import plot_confusion_matrix, classification_report

from scipy.stats import lognorm, loguniform, randint

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import make_scorer

import test_assignment3 as t

## Introduction and learning goals <a name="in"></a>
<hr>

Welcome to the assignment! After working on this assignment, you should be able to:

- Explain components of a confusion matrix.
- Define precision, recall, and f1-score and use them to evaluate different classifiers.
- Identify whether there is class imbalance and whether you need to deal with it.
- Explain `class_weight` and use it to deal with data imbalance.
- Apply different scoring functions with `cross_validate` and `GridSearchCV` and `RandomizedSearchCV`.
- Explain the general intuition behind linear models.
- Explain the `fit` and `predict` paradigm of linear models.
- Use `scikit-learn`'s `LogisticRegression` classifier.
    - Use `fit`, `predict` and `predict_proba`.   
    - Use `coef_` to interpret the model weights.
- Explain the advantages and limitations of linear classifiers. 


## Exercise 1:  Precision, recall, and f1 score "by hand" (without `sklearn`) <a name="1"></a>
<hr>


Consider the problem of predicting whether a new product will be successful or not and is worth investing in. Below are confusion matrices of two machine learning models: Model A and Model B. 

##### Model A
|    Actual/Predicted         | Predicted successful| Predicted not successful |
| :-------------------------- | ------------------: | -----------------------: |
| **Actually successful**     | 3                   | 5                        |
| **Actually not successful** | 6                   | 96                       |
 

##### Model B
|    Actual/Predicted         | Predicted successful| Predicted not successful |
| :-------------------------- | ------------------: | -----------------------: |
| **Actually successful**     | 6                   |                        14 |
| **Actually not successful** | 0                  |                       90 |  

### 1.1 Positive vs. negative class
rubric={autograde:1, reasoning:1}

Precision, recall, and f1 score depend crucially upon which class is considered "positive", that is the thing you wish to find. In the example above, which class ( `Actually successful` or `Actually not successful`)  is likely to be the "positive" class and why?

Save the label name in a string object named `answer_1_1`.


answer_1_1 = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer
answer_1_1

t.test_1_1(answer_1_1)

YOUR ANSWER HERE

### 1.2 Accuracy
rubric={autograde:2}

Calculate accuracies for Model A and Model B. 

Save the values of each calculations as a fraction in objects name `model_a_acc` and `model_b_acc` respectively.

model_a_acc = None
model_b_acc = None 

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

t.test_1_2_1(model_a_acc)

t.test_1_2_2(model_b_acc)

### 1.3 Which model would you pick? 
rubric={reasoning:1}

Which model would you pick simply based on the accuracy metric? 

YOUR ANSWER HERE

### 1.4 Model A - Precision, recall, f1-score
rubric={accuracy:3}

Calculate precision, recall, f1-score for **Model A** by designating the appropriate fraction to objects named `a_precision`, `a_recall` and `a_f1`. 

You can use the objects `a_precision` and `a_recall` to use in your `a_f1` calculation.

a_precision = None
a_recall = None
a_f1 = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

t.test_1_4_1(a_precision)

t.test_1_4_2(a_recall)

t.test_1_4_3(a_f1)

### 1.5 Model B - Precision, recall, f1-score
rubric={accuracy:3}

Calculate precision, recall, f1-score for **Model B** by designating the appropriate fraction to objects named `b_precision`, `b_recall` and `b_f1`. 

You can use the objects `b_precision` and `b_recall` to use in your `b_f1` calculation.

b_precision = None
b_recall = None
b_f1 = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

t.test_1_5_1(b_precision)

t.test_1_5_2(b_recall)

t.test_1_5_3(b_f1)

### 1.6 Metric choice
rubric={reasoning:2}

Which metric(s) is more informative in this case? Why? 


YOUR ANSWER HERE

### 1.7 Model choice
rubric={reasoning:2}

Which model would you pick based on this information and why? 

YOUR ANSWER HERE

## Exercise 2: Sentiment analysis on the IMDB dataset: model building <a name="3"></a>
<hr>

<img src="https://ia.media-imdb.com/images/M/MV5BMTk3ODA4Mjc0NF5BMl5BcG5nXkFtZTgwNDc1MzQ2OTE@._V1_.png"  width = "40%" alt="404 image" />

In this exercise, you will carry out sentiment analysis on a real corpus, [the IMDB movie review dataset](https://www.kaggle.com/utathya/imdb-review-dataset).
The starter code below loads the data CSV file (assuming that it's in the data directory) as a pandas DataFrame called `imdb_df`.

The supervised learning task is, given the text of a movie review, to predict whether the review sentiment is positive (reviewer liked the movie) or negative (reviewer disliked the movie). We have done a bit of preprocessing on the dataset already where the positive review are labelled `1` and the negative reviews are labelled `0`.

### BEGIN STARTER CODE
imdb_df = pd.read_csv("imdb_speed.csv")
train_df, test_df = train_test_split(imdb_df, test_size=0.2, random_state=77)
train_df.head()
### END STARTER CODE

### 2.1 Feature and target objects 
rubric={accuracy:2}

Separate our feature vectors from the target.

You will need to do this for both `train_df` and `test_df`.

Save the results in objects named `X_train`, `y_train`, `X_test` and `y_test`. 

(Makes sure that all 4 of these objects are of type Pandas Series. We will be using `CountVectorizer` for future questions and this transformation requires an input of Pandas Series)

X_train, y_train = None, None
X_test, y_test = None, None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

t.test_2_1(X_train,y_train,X_test,y_test)

### 2.2 Dummy classifier
rubric={accuracy:3}

Make a baseline model using `DummyClassifier`.

Carry out cross-validation using the `stratified` strategy. Pass the following `scoring` metrics to `cross_validate`. 
- accuracy
- f1
- recall
- precision

(We are using cross-validation here since we can obtain multiple scores at once) 

Make sure you use  `return_train_score=True` and 5-fold cross-validation.

Save your results in a dataframe named `dummy_scores_df`. 

dummy_scores_df = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

dummy_scores_df

t.test_2_2(dummy_scores_df)

### 2.3 Dummy classifier mean
rubric={accuracy:1}

What is the mean of each column in `dummy_scores_df`?

Save your result in an object named `dummy_mean`. 

dummy_mean = None 

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

dummy_mean

### 2.4 Pipeline
rubric={accuracy:2}

Let's make a pipeline now. 

Since we only have 1 column to preprocess, we only need 1 main pipeline for this question. 

Create a pipeline with 2 steps, one for `CountVectorizer` and one with a `LogisticRegression` model. For the LogisticRegression model, it's a good idea to set the argument `max_iter=1000` to avoid any warnings and convergence issues. Also let's balance the classes in our splits by setting the appropriate argument in `LogisticRegression` to "balanced". 

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### 2.5 Hyperparameter optimization
rubric={accuracy:4}

Use `RandomizedSearchCV` to jointly optimize the hyperparameters in the `params_grid` that we have provided for you. Name your `RandomizedSearchCV` object `random_search` as we have some code we will be giving you in exercise 3 that relies on this object name. 

Specify `n_iter=10`, `cv=5`, `n_jobs=-1`, `verbose=2`, `return_train_score=True` and **make sure  to use an f1 scoring metric here**. 

Make sure to fit your model on the training portion of the IMDB dataset. 

This can take quite a while (2-5 minutes for me!) so please be patient.

### BEGIN STARTER CODE
param_grid = {
    "countvectorizer__max_features": randint(10, 10000),
    "logisticregression__C": loguniform(0.01, 100), 
}
### END STARTER CODE

random_search = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### 2.6  Best hyperparameters
rubric={accuracy:2}

What are the best hyperparameter values found by `RandomizedSearchCV` for `C` and `max_features`. 
What was the corresponding validation score? 

optimal_parameters = None
optimal_score = None

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### 2.7 Train and test scores of best scoring model
rubric={accuracy:2}

What is the train and test `f1` score of the best scoring model?

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### 2.8 Hyperparameters and the fundamental tradeoff
rubric={reasoning:3}

From the set of possible models in the search, did your search return a relatively simple CountVectorizer or a relatively complex one? Did it return a relatively simple LogisticRegression or a relatively complex one? Here ‘simple’ and ‘complex’ we mean with respect to the fundamental tradeoff.

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

YOUR ANSWER HERE

### 2.9 Confusion matrix
rubric={accuracy:2}

Plot a confusion matrix on the test set using your random search object as your estimator. You may also want to add `display_labels` so that it's easier to recognized which class is a positive review and which is negative. 

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### 2.10 Classification report
rubric={accuracy:4}

Print a classification report on the `X_test` predictions of your random search object's best model with measurements to 4 decimal places . Use this information to answer the questions below.

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

A) What is the recall if we classify `1` as our "positive" class? 

B) What is the precision weighted average? Save the result to 4 decimal places. 

C) What is the `f1` score using `1` as your positive class? Save the result to 4 decimal places in an object named `answer3_5c`.

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### 2.11 Model Remarks
rubric={reasoning:2}

How well did your model do classifying the movie reviews? Take into consideration the new metrics we have learned. 

YOUR ANSWER HERE

## Exercise 3: Model Interpretation <a name="4"></a>
<hr>

One of the primary advantage of linear models is their ability to interpret models in terms of important features. In this exercise, we'll explore the weights learned by logistic regression classifier.

Below we've create a dataframe that contains the words used in our optimal model along with their coefficients (remember to name your `RandomizedSearchCV` object in question 2.5 `random_search` or the code here will return an error)  

### BEGIN STARTER CODE
best_estimator = random_search.best_estimator_

coef_df = pd.DataFrame({'words': best_estimator[ "countvectorizer"].get_feature_names(),
              'coefficient': best_estimator["logisticregression"].coef_[0]})

coef_df
### END STARTER CODE

### 3.1 Get the most informative positive words
rubric={accuracy:1, reasoning:1}

Using the dataframe `coef_df` above, find the 10 words that are most indicative of a positive review.

Elaborate on the positive words here - Do they make sense with their target value?

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

YOUR ANSWER HERE

### 3.2 Get the most informative negative words
rubric={accuracy:1, reasoning:1}

Using the dataframe `coef_df` above, find the 10 words that are most indicative of a positive review.

Elaborate on the positive words here - Do they make sense with their target value?

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

YOUR ANSWER HERE

### 3.3 Explaining the coefficients?
rubric={reasoning:2}

Do the words associated with positive and negative reviews make sense? Why is it useful to get access to this information?

YOUR ANSWER HERE

### 3.4 Using `predict` vs `predict_proba`
rubric={accuracy:3}

Make a dataframe named `results_df` that contains these 5 columns: 

- `review` - this should contain the reviews from `X_test`.
- `true_label` - This should contain the true `y_test` values. 
- `predicted_y` - The predicted labels generated from `best_model` for the `X_test` reviews using `.predict()`. 
- `neg_label_prob` - The probabilities of class `0` generated from `best_model` for the `X_test` reviews. These can be found at index 0 of the `predict_proba` output (you can get that using `[:,0]`). 
-  `pos_label_prob` - The probabilities of class `1` generated from `best_model` for the `X_test` reviews. These can be found at index 0 of the `predict_proba` output (you can get that using `[:,1]`). 

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

t.test_3_4(results_df)

### 3.5 Looking into the probability scores with positive reviews 
rubric={accuracy:2}

Find the top 5 movie reviews in `results_df` with the highest predicted probability of being positive (i.e., where the model is most confident that the review is positive).

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

Feel free to explore these reviews and see how positive they read!

### 3.6 Looking into the probability scores with negative reviews 
rubric={accuracy:2}

Find the top 5 movie reviews in `results_df` with the highest predicted probability of being negative (i.e., where the model is most confident that the review is negative).

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### 3.7 Looking at uncertain reviews (optional)
rubric={0}

This is an optional question!

(You'll get 0 marks for this one but you may have fun doing it?!) 

Find the 5 movie reviews in the test set with the most divided probability of being negative or positive (i.e., where the model is least confident in either review sentiment).

Can you see why the model is confused?

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### 3.8 Looking at wrongly predicted reviews. (optional)
rubric={0}

Here is another optional question!

Examine a review from the test set where our `best_model` is making mistakes, i.e., where the true labels do not match the predicted labels. 

Is is clear why this model predicted this review incorrectly?

# your code here
raise NotImplementedError # No Answer - remove if you provide an answer

### Submission to Canvas

**PLEASE READ: When you are ready to submit your assignment do the following:**

- Read through your solutions
- **Restart your kernel and clear output and rerun your cells from top to bottom** 
- Makes sure that none of your code is broken 
- Verify that the tests from the questions you answered have obtained the output "Success"
- Convert your notebook to .html format by going to File -> Export Notebook As... -> Export Notebook to HTML
- Upload your `.ipynb` file and the `.html` file to Canvas under Assignment1. 
- **DO NOT** upload any `.csv` files. 

### Congratulations on finishing Assignment 3!