BAIT 509 Class Meeting 03: Trees for Classification and Regression
================

Some problems (or at least potential problems) with the local methods introduced last time:

1.  They lack interpretation.
    -   It's not easy to say how the predictors influence the response from the fitted model.
2.  They typically require a data-rich situation so that the estimation variance is acceptable, without compromising the estimation bias.

Our setting this time is the usual: we have a response *Y* (either categorical or numeric), and hope to predict this response using *p* predictors *X*<sub>1</sub>, …, *X*<sub>*p*</sub>.

-   When the response is categorical, we aim to estimate the mode and take that as our prediction.
-   When the response is numeric, we aim to estimate the mean, and take that as our prediction.

Decision Stumps: A fundamental concept
--------------------------------------

Let's say I get an upset stomach once in a while, and I suspect certain foods might be responsible. My response and predictors are:

-   *Y*: sick or not sick (categorical)
-   *X*<sub>1</sub>: amount of eggs consumed in a day.
-   *X*<sub>2</sub>: amount of milk consumed in a day, in liters.

A **decision stump** is a decision on *Y* based on the value of *one* of the predictors.

![](cm03-trees_files/stump.png)

(Image attribution: Hyeju Jang, DSCI 571)

Learning procedure
------------------

Now that we know what a decision stump is, how can we choose the one that gives the highest prediction accuracy? We need to consider two things:

1.  which predictor will be involved in the decision, and
2.  the boundary on that predictor for which to make a prediction.

LAB PORTION
-----------

-   Why don't we want to fit a decision stump that results in the same decision being made in either case? (i.e., the same category being predicted). What distribution does the corresponding decision correspond to?
