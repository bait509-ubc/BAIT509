# BAIT 509 Assignment 2

__Evaluates__: Lectures 1 - 6. 

__Rubrics__: Your solutions will be assessed primarily on the accuracy of your coding, as well as the clarity and correctness of your written responses. The MDS rubrics provide a good guide as to what is expected of you in your responses to the assignment questions. In particular, here are the most relevant ones:

- [accuracy rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_accuracy.md), for evaluating your code.
- [reasoning rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_reasoning.md), for evaluating your written responses.

__Attribution__: This assignment was created by Tomas Beuzen and Vincenzo Coia.

## Tidy Submission (5%)

- Complete this assignment by filling out this jupyter notebook.
- You must use proper English, spelling, and grammar.
- You will submit two things to Canvas:
    1. This jupyter notebook file containing your responses; and,
    2. A html file of your completed notebook (use `jupyter nbconvert --to html_embed assignment.ipynb` in the terminal to generate the html file).
- Submit your assignment through [UBC Canvas](https://canvas.ubc.ca/courses/35074) by **11:59pm Monday 27th January**.

## Exercise 1: Revision of concepts (worth a total of 20%)
The following questions relate to material covered in lectures 1-4. Respond to the questions without using code. Provide clear and concise (1-3 sentence) answers to any written questions.

### 1.1 (5%)

Briefly explain what are the train, validation and test sets.

#### <font color="green">Solution 1.1</font>

- Training: used to fit/learn the model.
- Validation: used to optimize the model (e.g., by tuning hyperparameters).
- Testing: used as a final test of the model performance (not to be used at all to develop/select the model).

### 1.2 (5%)

Consider the hypothetical situation where we have two-predictors ($X_1$ and $X_2$) that we have split into 5 groups (A, B, C, D, E). Which of the following partitions of the predictor space into the 5 groups correspond to a decision tree model -- Figure 1 or Figure 2? What is the first split in the tree?

<img src='fig1.png' width="600">

#### <font color="green">Solution 1.2</font>

Figure 1 corresponds to a decision tree.

The first split in the tree is at $X_1 = 0.5$.

### 1.3 (5%)

The following figure is a scatterplot of 10 observations labelled either **x** or **o**, along with one unlabelled observation indicated by a **?**. In this question you are to answer the following using k-Nearest Neighbours:

1. Classify ? with k=1.
2. Classify ? with k=3.
3. Classify ? with k=10.

<img src='fig2.png' width="400"> 

#### <font color="green">Solution 1.3</font>

1. O
2. X
3. X

### 1.4 (5%)

Briefly describe what is wrong with the following code (assume all packages required to runt he code have already been loaded):

```
# Load the data from dataset.csv
df = pd.read_csv('dataset.csv', index_col=0)
X = df.drop(columns=['response'])
y = df[['response']]

# Scale the data using StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
                                                    
# Create a KNN model
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
```

#### <font color="green">Solution 1.4</font>

The problem with this code is that the data is scaled before splitting into training and testing portions. This means that the (future) test data is being used to fit the StandardScaler which violates the golden rule. In other words, we are using the test data to help us scale the data which is bad practice.

## Exercise 2: Logistic regression (worth a total of 35%)

In this exercise you will work with a logistic regression model. Recall that logistic regression, despite the name, is used for classification tasks. Typically it is used to model the relationship between a binary target variable and one or more numeric or categorical features.

In this exercise we will be focusing on predicting the presence or absence of heart disease in a patient based on a set of 13 different biophysical measures. The classification of heart disease in patients is obviously of great importance for cardiovascular disease diagnosis and prevention. Machine learning offers novel and potentially effective methods of forming predictive models from heart disease data, and this particular dataset was an early example of how data and machine learning can be leveraged to create incredibly effective predictive models. You can download and read more about the original dataset on the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). A slightly modified version of this dataset has been made available to you for this assignment called `heart_disease.csv`. You will see that it contains 303 observations (patients) and 14 columns (the 13 features and 1 target variable).

### 2.1 (10%)

Your first task is to wrangle this dataset into a format suitable for use with the scikit-learn library. This includes:

1. Loading the dataset;
2. Feature preprocessing (one-hot encoding and scaling); and,
3. Splitting data into training and testing sets.

This dataset has both numeric and categorical features which makes the feature preprocessing step a little harder. There is an sklearn function called `ColumnTransformer` that helps you to apply numeric feature preprocessing (e.g., `StandardScaler` and categorical feature preprocessing (e.g., `OneHotEncoder`) simultaneously. To help you understand the data wrangling process, the code required to perform the pre-processing tasks above is provided. The code has been arranged into five blocks performing the tasks above but these blocks are in the wrong order. **Rearrange the code below to correctly wrangle the data and add a short, one-line comment to each block to describe what the code is doing.**

# Imports, you don't need to rearrange this cell!
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Rearrange this cell to get it to run and properly wrangle the heart disease data

# YOUR COMMENT HERE
X_train = pd.DataFrame(preprocessor.fit_transform(X_train),
                       index=X_train.index,
                       columns=(numeric_features +
                                list(preprocessor.named_transformers_['ohe']
                                     .get_feature_names(categorical_features))))
X_test = pd.DataFrame(preprocessor.transform(X_test),
                      index=X_test.index,
                      columns=X_train.columns)

# YOUR COMMENT HERE
numeric_features = ['age', 'resting_blood_pressure', 'cholesterol',
                    'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']
categorical_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
                        'exercise_induced_angina', 'st_slope', 'thalassemia']

# YOUR COMMENT HERE
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), numeric_features),
        ('ohe', OneHotEncoder(drop="first"), categorical_features)])

# YOUR COMMENT HERE
heart_df = pd.read_csv('heart_disease.csv', index_col=0)

# YOUR COMMENT HERE
X_train, X_test, y_train, y_test = train_test_split(heart_df.drop(columns='target'),
                                                    heart_df[['target']],
                                                    test_size=0.3,
                                                    random_state=123)

#### <font color="green">Solution 2.1</font>

# 1. Load the data
heart_df = pd.read_csv('heart_disease.csv', index_col=0)

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(heart_df.drop(columns='target'),
                                                    heart_df[['target']],
                                                    test_size=0.3,
                                                    random_state=123)


# 2. Identify the categorical and numeric columns
numeric_features = ['age', 'resting_blood_pressure', 'cholesterol',
                    'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']
categorical_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
                        'exercise_induced_angina', 'st_slope', 'thalassemia']

# 3. Set-up data transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), numeric_features),
        ('ohe', OneHotEncoder(drop="first"), categorical_features)])


# 5. Apply data transformations and convert to dataframe
X_train = pd.DataFrame(preprocessor.fit_transform(X_train),
                       index=X_train.index,
                       columns=(numeric_features +
                                list(preprocessor.named_transformers_['ohe']
                                     .get_feature_names(categorical_features))))
X_test = pd.DataFrame(preprocessor.transform(X_test),
                      index=X_test.index,
                      columns=X_train.columns)

### 2.2 (5%)

Train a logistic regression model on the training data (using default hyperparameter settings) and calculate the model's error on the training data and testing data.

#### <font color="green">Solution 2.2</font>

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Error on training data = {1 - model.score(X_train, y_train):.2f}")
print(f"Error on test data = {1 - model.score(X_test, y_test):.2f}")

### 2.3 (5%)

Based on your results from Q2.2 would you say your logisitic regression model is overfit? Why/why not?

#### <font color="green">Solution 2.3</font>
The model does not appear to be overfit as the training and testing scores are similar.

### 2.4 (10%)

Recall that a logistic regression model outputs a probability between 0 and 1, where, by default, probabilities less than 0.5 are assigned to class 0 and probabilites greater than 0.5 are assigned to class 1. The predictions of the logistic regression model can be calculated using the `predict_proba()` method and the particular classes the predictions refer to can be obtained from the `classes_` attribute.

1. What is the predicted probability that the first observation in the test set (patient_id 11) has heart disease (target = 1)?
2. What is the predicted probability that the first observation in the test set (patient_id 11) does not have heart disease (target = 0)?
3. What patient ID has the highest predicted probability of heart disease (give the actual patient_id number, not the index location).

#### <font color="green">Solution 2.4</font>

# probability of heart disease for the first observation in the test set (patient_id=11)
pd.DataFrame(data = model.predict_proba(X_test)[0],
             index = ['No heart disease', 'Heart disease'],
             columns = ['Probability'])

# There are a few ways we can find the patient with the highest probability of heart disease
# Below I first find the location of the maximum probability using np.argmax
# I then use that location to index the relevant dataframe row using .iloc
# It's patient 124
patient_id = np.argmax(model.predict_proba(X_test)[:, 1])
X_test.iloc[[patient_id]]

### 2.5 (5%)

Recall that we can investigate the coefficient values of our logistic regression model to help understand the importance of the different features. Information of the coefficients is exposed by the `coef_` attribute of your model. Which feature appears to have the most influence on the prediction of heart disease?

#### <font color="green">Solution 2.5</font>

# We are looking for the coefficient with the largest absolute value
# We see that the feature chest_pain_type_non-anginal pain has the biggest
# influence over the probability of heart disease with a value of 1.55
pd.DataFrame(data = model.coef_.T,
             index = X_train.columns,
             columns = ["coef"]).sort_values(by='coef')

## Exercise 3: Naive Bayes for sentiment analysis (40%) <a name="3"></a>

Naive Bayes is popular in text classification tasks. In this exercise you will use 
the Naive Bayes algorithm to conduct [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis), which is a problem of assigning positive or negative labels to text based on the sentiment (attitude) expressed within it. [Here](https://www.youtube.com/watch?v=uXu2uEubV9Q&list=PLoROMvodv4rOFZnDyrlW3-nI7tMLtmiJZ&index=33) is a short video explaining the task of sentiment analysis. 

We will be using a dataset of movie reviews for this exercise. The dataset is know as the IMDB movie review data set and is available on [kaggle](https://www.kaggle.com/utathya/imdb-review-dataset). It contains 100,000 different movie reviews, labelled as either being positive or negative. The file is quite large (around 129 mb) so please download it directly from kaggle (simply click on the button towards the top of the screen that says `Download (129mb)`), you may need to create a free account to do this. 

### 3.1 (10%)

1. Load the data file you downloaded into a pandas DataFrame called `imdb_df` (hint: you will need to use the arguments `index_col=0` and `encoding = "ISO-8859-1"` in `pd.read_csv` to open this particular csv file correctly);
2. Drop the columns `"type"` and `"file"` from the dataframe;
3. Notice that in the `"labels"` columns there are three possible values: `pos`, `neg`, and `unsup`. Discard rows with the `unsup` label from the dataframe. There are several ways to perform this operation, I like to use the `query()` function, but feel free to use any method you wish.

Your final dataframe should have two columns and 50,000 rows.

#### <font color="green">Solution 3.1</font>

df = pd.read_csv('imdb_master.csv', index_col=0, encoding = "ISO-8859-1")
df = df.drop(columns = ['type', 'file'])
df = df.query('label == "neg" | label == "pos"')

### 3.2 (5%)
Split the data into train (80%) and test (20%) sets. 


#### <font color="green">Solution 3.2</font>

X_train, X_test, y_train, y_test = train_test_split(df['review'], 
                                                    df['label'], 
                                                    test_size = 0.2, 
                                                    random_state = 123)

### 3.3 (5%)

The current data is in the form of movie reviews (text paragraphs) and their targets (`pos` or `neg`). 
We need to encode the movie reviews into feature vectors so that we can train supervised machine learning models with `scikit-learn`. We will use sklearn's [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to transform the text data into a *bag-of-words* representation (i.e., when text is represented as a vector of counts, disregarding word order and grammar). Your tasks are to:

1. Create a `CountVectorizer` object.
2. Call `CountVectorizer`'s `fit_transform` method on the train split of the movie review data to get the features for the train set.
3. Call `CountVectorizer`'s `transform` method on the test split to get the features for the test set.

#### <font color="green">Solution 3.3</font>

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)  
X_test = vectorizer.transform(X_test) 

### 3.4 (5%)

1. Fit a [multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB) model to the training data.
2. Report the train and test errors of the model.

#### <font color="green">Solution 3.4</font>

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
print(f'Training error: {1 - model.score(X_train, y_train):.2f}')
print(f'    Test error: {1- model.score(X_test, y_test):.2f}')

### 3.5 (5%)

When using sklearn's `CountVectorizer` text is represented as a vector of counts. Do you think this is an adequate representation of the information contained with the text? When dealing with text data, what other information might be important for making accurate predictions with a machine learning model?

#### <font color="green">Solution 3.5</font>
`CountVectorizer` does not consider word order, grammar or how frequently a partiuclar word occurs in the whole training set (words that occur many times in a dataset may not be as informative as words that only occur a few times).

### 3.6 (10%)

Below is an additional movie review not included in the original IMDB dataset.

1. Do you think this review is positive or negative overall?
2. Use your Naive Bayes model to predict the sentiment of this review. What is the prediction?
3. Does you model predict the sentiment you expected? If not, why do you think that might be the case?

review = ['''It could have been a great movie. It could have been excellent, 
          and to all the people who have forgotten about the older, 
          greater movies before it, will think that as well. 
          It does have beautiful scenery, some of the best since Lord of the Rings. 
          The acting is well done, and I really liked the son of the leader of the Samurai.
          He was a likeable chap, and I hated to see him die...
          But, other than all that, this movie is nothing more than hidden rip-offs.
          ''']

#### <font color="green">Solution 3.5</font>
1. The review appears negative overall. But there are a few positives in it.

# 2.
prediction = model.predict(vectorizer.transform(review))
print(f"The Naive Bayes model predicts that this review is: {prediction[0]}")

3. The model does not predict the senitment we expect, it predicts positive when the overall review is negative. It seems to fail for more complex examples, where understanding the context and overall text is essential to correctly classify reviews. The last example has many positive words in the beginning but the last sentence negates all positivity in the previous text. We need to incorporate deeper linguistic knowledge to correctly classify such cases.