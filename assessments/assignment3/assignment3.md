# BAIT 509 Assignment 3

__Evaluates__: Class meetings 08 and 09.

__Due__: Wednesday, March 28 at 10:00am (i.e., the start of Class Meeting 10).


## Instructions (5%)

- You must use proper spelling and grammar.
- Use either R or python to complete this assignment (or both). 
- Submit your assignment through [UBC Connect](https://connect.ubc.ca/) by the deadline. 
- If you submit more than one file for your assignment, be sure to also include a README file to inform the grader of how to navigate your solutions.

## Exercise 1: Supervised learning beyond the mean

### 1a Probabilistic Forecasting

The `diamonds` dataset from the `ggplot2` package contains information about 53,940 diamonds. We will be using length, width, and depth (`x`, `y`, and `z`m respectively) to predict the quality of the cut (variable `cut`). Cut quality is categorical with five possible levels. 

You own a shop that sells diamond, and you receive word of two new diamonds, with the following dimensions: 

- Diamond 1: `x=4`, `y=4`, and `z=3`.
- Diamond 2: `x=6`, `y=6`, and `z=4`.

You can choose only one diamond to include in your store, but only have this information. You want the diamond with the highest cut quality.

1. Produce a probabilistic forecast of the cut quality for both diamonds, using a moving window (loess-like) approach with a window width of 0.5. It's sufficient to produce a bar plot showing the probabilities of each class, for each of the two diamonds. 
2. What cut quality would be predicted by a local classifier for both diamonds? Does this help you make a decision?
3. Looking at the probabilistic forecasts, make a case for one diamond over the other by weighing the pros and cons of each.

### 1b Local Quantile Regression

Use linear quantile regression to estimate the 0.25-, 0.5-, and 0.75-quantiles of fuel efficiency (`mpg`) from the weight (`wt`) of a vehicle. Use the `mtcars` dataset that comes with R. You might find the `rq` function in the `quantreg` package useful, or even the `geom_quantile` function in the `ggplot2` package for plotting. 

1. Plot the data with the three lines overtop. This gives us a "continuous version" of a boxplot (at least, the "box part" of the boxplot).
2. For a car that weighs 3500 lbs (i.e., `wt=3.5`), what is an estimate of the 0.75-quantile of `mpg`? Interpret this quantity.
3. What problems do we run into when estimating these quantiles for a car that weighs 1500 lbs (i.e., `wt=1.5`)? Hint: check out your three quantile estimates for this car.
