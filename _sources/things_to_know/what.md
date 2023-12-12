# What: Learning Outcomes 

## Course Objectives
 
 1.	Describe fundamental machine learning concepts such as: supervised and unsupervised learning, regression and classification, overfitting, training/validation/testing error, parameters and hyperparameters, and the golden rule.
2. Broadly explain how common machine learning algorithms work, including: naïve Bayes, k-nearest neighbors, decision trees, support vector machines, and logistic regression.
3. Identify when and why to apply data pre-processing techniques such as scaling and one-hot encoding.
4. Use Python and the scikit-learn package to develop an end-to-end supervised machine learning pipeline.
5. Apply and interpret machine learning methods to carry out supervised learning projects and to answer business objectives.

### Lecture 1

- Explain motivation to study machine learning.
- Differentiate between supervised and unsupervised learning.
- Differentiate between classification and regression problems.
- Explain machine learning terminology such as features, targets, training, and error.
- Explain the `.fit()` and `.predict()` paradigm and use `.score()` method of ML models.
- Broadly describe how decision trees make predictions.
- Use `DecisionTreeClassifier()` and `DecisionTreeRegressor()` to build decision trees using scikit-learn.
- Explain the difference between parameters and hyperparameters.
- Explain how decision boundaries change with `max_depth`.

### Lecture 2

- Explain the concept of generalization.
- Split a dataset into train and test sets using `train_test_split` function.
- Explain the difference between train, validation, test, and "deployment" data.
- Identify the difference between training error, validation error, and test error.
- Explain cross-validation and use `cross_val_score()` and `cross_validate()` to calculate cross-validation error.
- Explain overfitting, underfitting, and the fundamental tradeoff.
- State the golden rule and identify the scenarios when it's violated.

### Lecture 3

- Use `DummyClassifier` and `DummyRegressor` as baselines for machine learning problems.
- Explain the notion of similarity-based algorithms .
- Broadly describe how KNNs use distances.
- Discuss the effect of using a small/large value of the hyperparameter $K$ when using the KNN algorithm 
- Explain the general idea of SVMs with RBF kernel.
- Describe the problem of the curse of dimensionality.
- Broadly describe the relation of `gamma` and `C` hyperparameters and the fundamental tradeoff.


### Lecture 4

- Identify when to implement feature transformations such as imputation and scaling.
- Describe the difference between normalizing and standardizing and be able to use scikit-learn's `MinMaxScaler()` and `StandardScaler()` to pre-process numeric features.
- Apply `sklearn.pipeline.Pipeline` to build a machine learning pipeline.
- Use `sklearn` for applying numerical feature transformations to the data.
- Discuss the golden rule in the context of feature transformations.


### Lecture 5

- Identify when it's appropriate to apply ordinal encoding vs one-hot encoding.
- Explain strategies to deal with categorical variables with too many categories.
- Explain `handle_unknown="ignore"` hyperparameter of `scikit-learn`'s `OneHotEncoder`.
- Use the scikit-learn `ColumnTransformer` function to implement preprocessing functions such as `MinMaxScaler` and `OneHotEncoder` to numeric and categorical features simultaneously.
- Use `ColumnTransformer` to build all our transformations together into one object and use it with `scikit-learn` pipelines.
- Explain why text data needs a different treatment than categorical variables.
- Use `scikit-learn`'s `CountVectorizer` to encode text data.
- Explain different hyperparameters of `CountVectorizer`.

### Lecture 6

- Identify when it's appropriate to apply ordinal encoding vs one-hot encoding.
- Explain strategies to deal with categorical variables with too many categories.
- Explain `handle_unknown="ignore"` hyperparameter of `scikit-learn`'s `OneHotEncoder`.
- Use the scikit-learn `ColumnTransformer` function to implement preprocessing functions such as `MinMaxScaler` and `OneHotEncoder` to numeric and categorical features simultaneously.
- Use `ColumnTransformer` to build all our transformations together into one object and use it with `scikit-learn` pipelines.
- Explain why text data needs a different treatment than categorical variables.
- Use `scikit-learn`'s `CountVectorizer` to encode text data.
- Explain different hyperparameters of `CountVectorizer`.

### Lecture 7

- Explain the general intuition behind linear models.
- Explain the `fit` and `predict` paradigm of linear models.
- Use `scikit-learn`'s `LogisticRegression` classifier.
    - Use `fit`, `predict` and `predict_proba`.   
    - Use `coef_` to interpret the model weights.
- Explain the advantages and limitations of linear classifiers. 
- Apply scikit-learn regression model (e.g., Ridge) to regression problems.
- Relate the Ridge hyperparameter `alpha` to the `LogisticRegression` hyperparameter `C`.
- Compare logistic regression with naive Bayes.

### Lecture 8

- In the context of supervised learning, form statistical questions  from business questions/objectives.
- Understand the different forms your client may expect you to communicate results. 
- Explain the general concept of feature selection.
- Discuss and compare different feature selection methods at a high level.
- Use sklearn's implementation of recursive feature elimination (RFE).
- Implement the forward search algorithm.


### Lecture 9

- Explain why accuracy is not always the best metric in ML.
- Explain components of a confusion matrix.
- Define precision, recall, and f1-score and use them to evaluate different classifiers.
- Identify whether there is class imbalance and whether you need to deal with it.
- Explain `class_weight` and use it to deal with data imbalance.
- Appropriately select a scoring metric given a regression problem.
- Interpret and communicate the meanings of different scoring metrics on regression problems. MSE, RMSE, $R^2$, MAPE.
- Apply different scoring functions with `cross_validate`, `GridSearchCV` and `RandomizedSearchCV`.


### Lecture 10

- Explain ethical considerations in data science, relating to multiple phases of machine learning pipelines.
- Be able to analyze a confusion matrix and think about how different scoring metrics affect diverse stakeholders.
- Explain components of a confusion matrix with respect to multi-class classification.
- Define precision, recall, and f1-score with multi-class classification
- Carry out multi-class classification using OVR and OVO strategies.
