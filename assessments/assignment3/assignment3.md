# BAIT 509 Assignment 3

__Evaluates__: Class meetings 08 and 09.

## Instructions (5%)

- You must use proper spelling and grammar.
- Follow the language specifications for each question.
- Submit your assignment through [UBC Canvas](https://canvas.ubc.ca/) by the deadline. 
- If you submit more than one file for your assignment, be sure to also include a README file to inform the grader of how to navigate your solutions. This can be a plain text file.

## Exercise 1 (15%): Probabilistic Forecasting 

The `diamonds` dataset from the `ggplot2` R package contains information about 53,940 diamonds. We will be using length, width, and depth (`x`, `y`, and `z`m respectively) to predict the quality of the cut (variable `cut`). Cut quality is categorical with five possible levels. 

You own a shop that sells diamond, and you receive word of two new diamonds, with the following dimensions: 

- Diamond 1: `x=4`, `y=4`, and `z=3`.
- Diamond 2: `x=6`, `y=6`, and `z=4`.

You can choose only one diamond to include in your store, but only have this information. You want the diamond with the highest cut quality.

Answer the following questions.

1. Produce a probabilistic forecast of the cut quality for both diamonds, using a moving window (loess-like) approach with a window width of 0.5. It's sufficient to produce a bar plot showing the probabilities of each class, for each of the two diamonds. 
2. What cut quality would be predicted by a local classifier for both diamonds? Does this help you make a decision?
3. Looking at the probabilistic forecasts, make a case for one diamond over the other by weighing the pros and cons of each.

Hint: I don't know of any software that automates this procedure, so you'll probably have to do this manually. Refer to lecture 8's worksheet for sample code. The `filter` function in the `dplyr` package in R is useful for subsetting a data frame.

## Exercise 2: Quantile Regression

__Language hints__: You might find the `rq` function in the `quantreg` package useful in R, or even the `geom_quantile` function in the `ggplot2` package for plotting. If you choose to use python, [here](http://www.statsmodels.org/dev/examples/notebooks/generated/quantile_regression.html) is a demo that you might find useful.

The [`auto_data.csv`](https://raw.githubusercontent.com/vincenzocoia/BAIT509/master/assessments/assignment3/data/auto_data.csv) file contains automotive data for 392 cars. The `mtcars` dataset, that "comes with" R, contains automotive data for 32 cars.

### 2(a) (15%)

Using the `mtcars` dataset only, use linear quantile regression to estimate the 0.25-, 0.5-, and 0.75-quantiles of fuel efficiency (`mpg`) from the weight (`wt`) of a vehicle. 

1. Plot the data with the three lines/curves overtop. This gives us a "continuous version" of a boxplot (at least, the "box part" of the boxplot).
2. For a car that weighs 3500 lbs (i.e., `wt=3.5`), what is an estimate of the 0.75-quantile of `mpg`? Interpret this quantity.
3. What problems do we run into when estimating these quantiles for a car that weighs 1500 lbs (i.e., `wt=1.5`)? Hint: check out your three quantile estimates for this car.

### 2(b) (15%)

Let's now use the `auto_data.csv` data as our training data, and the `mtcars` data as a validation set. So, you can ignore your results from 2(a). 

__Note__: In both `mtcars` and `auto_data.csv`, the fuel efficiency is titled `mpg` and have the same units as the `mtcars` data. The weight column is titled `weight` in the csv file, and is in lbs, but is title `wt` in `mtcars`, where the units are thousands of lbs.

1. To the training data, fit the 0.5-quantile of `mpg` using the "weight" variable as the predictor. Fit two models: a linear model and a quadratic model.
2. Plot the two quantile regression curves overtop of the training data.
3. What error measurement is appropriate here? Hint: it's not mean squared error. 
4. Using the validation data, calculate the error of both models. You'll first have to convert the "weight" variable to be in the same units as your training data.
5. Use the error measurements and the plot to justify which of the two models is more appropriate.

__Hint__: A quadratic model is the same as a linear one with a new predictor that's computed as the square of the original predictor. You can either make the new column in your data frame, or indicate that you want a polynomial of degree 2 when specifying the `formula` by indicating `y ~ poly(x, 2)` for response `y` and predictor `x`. The formula `y ~ x + x^2` won't work.


## Exercise 3 (20%): SVM Concepts 

__Language__: Only low-level programming is allowed here -- no machine learning packages are to be used.

From Section 9.7 in the [ISLR book](http://www-bcf.usc.edu/~gareth/ISL/) (that starts on page 368), complete the following:

- Question 1(a)
- Question 3, parts (a)-(f) inclusive.

## Exercise 4 (30%): Fitting SVM 

__Language__: python recommended, but not required.

__Language hints__: Check out [`sklearn`'s implementation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) of SVM. 

__Attribution__: These questions are adapted from the [ISLR book](http://www-bcf.usc.edu/~gareth/ISL/), Section 9.7 questions 7(b) and (c).

For this exercise, you're expected to use the `auto_data.csv` dataset, not the `mtcars` data set. The `name` and `mpg` columns should _not_ be used as predictors, but use everything else. Here is code to load the dataset and separate the data into predictors and response (where `pd` is `pandas`):

```
dat = pd.read_csv("data/auto_data.csv")
y = dat["mileage"]
X = dat[["cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin"]]
```


Your task is to use SVM to predict whether a car has high or low mileage (column `mileage`) using two methods:

1. An SVM with a linear kernel (so, a basic support vector classifier)
2. An SVM with a radial basis kernel

Answer the following questions:

1. Split the data into random training and validation sets. Set aside 40% of the data for the validation set. OR, skip this step if you plan on doing cross validation.
2. Fit the two models over a grid of hyperparameters, and report the generalization error in all cases. You can report them in a table, or a plot. To make things easy, you don't have to choose a fine grid, as long as the optimal hyperparameters are located somewhere in the range of the grid.
3. For both methods, what hyperparameter(s) on your grid has the best generalization error? Of these two models, which one is better, the linear or radial?

