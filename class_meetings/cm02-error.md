# BAIT 509 Class Meeting 02
Wednesday, February 28, 2018  




# Outline

- Finish irreducible error from last time
- Reducible error and the fundamental tradeoff.

# Review from last time

From the "irreducible error" section in cm01:

- Even if we know the true probability distribution of the data, we can't predict $Y$ without error.
    - We can "beat" this error by adding informative predictors.
- We measure error by mean squared error (MSE) for regression, and the error rate for classification.

Activities:

- I'll give live-coding solutions to the Oracle Regression exercise.
    - Includes a demo of RMarkdown
    - I'll demonstrate the save-commit-push workflow in git.
- I'll let you work on the Oracle Classification exercise.

# Reducible Error

## What is it?

Last time, we saw what irreducible error is, and how to "beat" it. 

The other type of error is __reducible error__, which arises from not knowing the true distribution of the data (or some aspect of it, such as the mean or mode). We therefore have to _estimate_ this. Error in this estimation is known as the reducible error. 

__Example__: Consider the case of one predictor, and the true distribution of the data is $Y|X=x \sim N(5/x, 1)$ (and take $X \sim 1+Exp(1)$). I'll generate 100 realizations from this distribution, to form our data. In reality, we are only ever faced with this data, and know almost nothing about the distribution. As such, I decide to try linear regression to form my forecaster. Here's what I get:

<img src="cm02-error_files/figure-html/unnamed-chunk-1-1.png" style="display: block; margin: auto;" />

The difference between the true curve and the estimated curve is due to reducible error. 

In the classification setting, a misidentification of the mode is due to reducible error. 

(__Why the toy data set instead of real ones?__ Because I can embed characteristics into the data for pedagogical reasons. You'll see real data at least in the assignments and final project.)

## Bias and Variance

There are two key aspects to reducible error: __bias__ and __variance__. They only make sense in light of the hypothetical situation of building a model/forecaster over and over again as we generate a new data set over and over again.

- __Bias__ occurs when your estimates are systematically different from the truth. For regression, this means that the estimated mean is either usually bigger or usually smaller than the true mean. For a classifier, it's the systematic tendency to choosing an incorrect mode.
- __Variance__ refers to the variability of your estimates.

There is usually (always?) a tradeoff between bias and variance. It's referred to as the __bias/variance tradeoff__, and we'll see examples of this later.  

Let's look at the above linear regression example again. I'll generate 100 data sets, and fit a linear regression for each:

<img src="cm02-error_files/figure-html/unnamed-chunk-2-1.png" style="display: block; margin: auto;" />

The _spread_ of the linear regression estimates is the variance; the difference between the _center of the regression lines_ and the true mean curve is the bias. 

## Reducing reducible error

As the name suggests, we can reduce reducible error. Exactly how depends on the machine learning method, but in general:

- We can reduce variance by increasing the sample size, and adding more model assumptions.
- We can reduce bias by being less strict with model assumptions, OR by specifying them to be closer to the truth (which we never know).

Consider the above regression example again. Notice how my estimates tighten up when they're based on a larger sample size (1000 here, instead of 100):

<img src="cm02-error_files/figure-html/unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

Notice how, after fitting the linear regression $E(Y|X=x)=\beta_0 + \beta_1 (1/x)$ (which is a _correct_ model assumption), the regression estimates are centered around the truth -- that is, they are unbiased:

<img src="cm02-error_files/figure-html/unnamed-chunk-4-1.png" style="display: block; margin: auto;" />


## Error decomposition

We saw that we measure error using mean squared error (MSE) in the case of regression, and the error rate in the case of a classifier. These both contain all errors: irreducible error, bias, and variance:

MSE = bias^2 + variance + irreducible variance

A similar decomposition for error rate exists.

__Note__: If you look online, the MSE is often defined as the expected squared difference between a parameter and its estimate, in which case the "irreducible error" is not present. We're taking MSE to be the expected squared distance between a true "new" observation and our prediction (mean estimate). 
