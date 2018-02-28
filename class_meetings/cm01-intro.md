# BAIT 509 Class Meeting 01
Monday, February 26, 2018  




# Overview

In today's class, we'll discuss:

- About me, you, the course, and such.
- Discussions: asking effective questions, and intro to your participation assessment
- Getting acquainted with the course tools: GitHub, git (if you want), R/RStudio/RMarkdown, Jupyter Notebooks
- What machine learning is, beginning with the concept of irreducible error. 

# About

## About me

A data scientist and teaching fellow at UBC for the Master of Data Science program. My background:

- BSc Biology (Brock U)
- BSc Mathematics (Brock U)
- MSc Mathematical Statistics (Brock U)
- PhD Statistics (UBC)
    - Thesis on forecasting extremes for flooding via multivariate dependence modelling.

Computationally, I am primarily an R programmer. I'm a basic python user.

Lots of experience as a statistical consultant for business and academia. 

## About the TA's

This course has three TA's.

- Vaden and Mohamed are in comp sci.
- Rafi is in ECE.

You'll see them in class and office hours. 

## About you

Introduce yourself! Let's hear about:

- Your name
- Why MBAN?
- Something about yourself.

## About this course

This is a brand new course! You can expect things to evolve as we progress.

- Preview of the course website -- it contains all materials related to the course.
- Preview of the syllabus

Before we talk more about the assessments, let's first talk about the structure of our class meetings. 

## About class meetings

I intend most class meetings to take the __listen-do-listen-do__ form:

1. __Listen__: (15 min?) I'll start class meetings with high-level explanations of a topic.
2. __Do__: (40 min?) You do hands-on exploratory work to allow you to build concepts.
    - You "present" your work to initiate a class discussion; I clarify concepts as needed.
3. __Listen__: (25 min?) Once you have basic concepts down pat, I'll talk more about details. Maybe iterate back to 1 with another topic. 
4. __Do__: (40 min?) Open time for working on course material, typically assignments or in-class material. Instructor and a TA will be present to answer questions. 

Notes:

- This structure might evolve as the course progresses. 
- There will be a TA present to help you.
- In-class exercises will typically contain toy data sets; assignments will be more "real".

# Participation Skills


## Your participation assessment

Let's go over how you'll be assessed. See the [`participation`](../assessments/participation/participation.md) file in the `assessments` folder.


## Online discussion

By now, you're used to asking questions and contributing to discussions _in class_. But the world of _online discussions_ is quite different. 

You are encouraged to ask questions online via [GitHub issues on the BAIT509 repo](https://github.com/vincenzocoia/BAIT509/issues). Feel free to ask about anything related to the course, including (but not limited to):

- content
- assessments
- coding
- jobs
- requests

You are also encouraged to interact with these Issues: 

- Add a comment to the Issue if you have something to add; or propose a solution to their question.
- Add to the question.
- Give the question a "thumbs up" if you have this same question and like the way it was worded.
- Ask for more details.

Think for a moment why I'm including this in your _class notes_ instead of in the document outlining our expectations for your `participation`. When you interact like this outside of the BAIT509 repo, you become a valuable contributor to the data science community in general! For example, you would do these things if you experience a (legitimate) problem with an R package. See the [`ggplot2` Issues](https://github.com/tidyverse/ggplot2/issues) as an example; or, if an R package does not have a GitHub repo, you can contact the author directly. 

## Asking effective questions

It's an art to ask a question online _effectively_. The idea is to make things as easy as possible for someone to answer. Make it self-contained; don't make someone have to do unnecessary digging to answer the question:

- Provide relevant links (for example, to the relevant assignment or course notes).
- Make a reproducible example (in the case that it's a question involving code).
- Be detailed.

You'll probably find that the act of writing an effective question causes you to answer your own question! In this case, post your question anyway, with the corresponding solution. 

Note that the BAIT 509 GitHub issues are __public__ -- anyone with internet can view them. This is a good thing -- it is a display of your thoughtfulness, your willingness to participate, and your willingness to ask questions. 

On that note, don't be afraid to ask a question here! Discrimination and ridicule will not be tolerated, and this includes on the BAIT509 repo. The teaching team will provide guidance on a per-question basis, and students are also encouraged to provide ways in which a question can be made more effective. 

I recommend checking out the [STAT545 "how to get unstuck" page](http://stat545.com/help-general.html) for resources on how to get your questions answered. 

# Git and GitHub

We'll be using git with GitHub for course materials, including a place to put your work on the in-class exercises.

__git__: A version control program. It keeps track of file changes.

__GitHub__: A service to host your files, and integrates with `git`.

We'll come back to `git` later, but for now, let's focus on GitHub. 

You might find these readings helpful:

- I recommend reading ["Why Git? Why GitHub?"](http://happygitwithr.com/big-picture.html) by Jenny Bryan if you're not familiar with what these tools are all about, and why they are useful for data science. It's an incomplete chapter in an incomplete book, but still very useful.
- [This STAT 545 lesson](http://stat545.com/cm003-notes_and_exercises.html) aims to introduce git and GitHub.

# In-Class Exercises: Tooling, Part 1

For your first exercise, you'll first acquaint yourself with GitHub, then add a comment on the BAIT509 Issue about effective questions. Here are your instructions.

__Note__: Done early? Head to the "Lab" section of these course notes for additional tasks!

## Acquaint yourself with GitHub and the BAIT 509 repo

1. __Make an account__: If you haven't already, make an account on [github.com](https://github.com/).
    - Jenny's ["Register a GitHub account"](http://happygitwithr.com/github-acct.html) chapter is useful if you want to read more.
2. __Fill out the survey__: Let us know your github.com username by filling in the survey at https://goo.gl/forms/86bIrgJSUvJBVdef2
3. __Watch BAIT509__: Navigate to the [BAIT509 github page](https://github.com/vincenzocoia/BAIT509), and click "Watch" -> "Watching" (in the upper-right corner of the page). 
    - This will notify you by email whenever a new issue comment is made. We'll be using issues for remote discussion about course material and assignments. 
4. __Set up your own BAIT509 repo__ (repo="repository"): You'll use this repo to put your in-class work, and notes. You can do this in one of two ways:
    1. Maintain your own copy of the main BAIT 509 repo ~~(\*__recommended__\*)~~, by [`Fork`ing](https://help.github.com/articles/fork-a-repo/) the main BAIT509 repo. 
        - There's a button for this at the top-right corner of the main BAIT 509 repo page. 
        - This is useful so you can [keep your fork synced](https://help.github.com/articles/fork-a-repo/#keep-your-fork-synced) as the main repo evolves throughout the course -- more on this later. 
    2. Start fresh and create a new blank repo.
        - On second thought, I would recommend this if git and github are new to you.
5. __Add to your repo__: On your new BAIT 509 repo, add a new folder for you to put your in-class exercises, seeded with a README file. To do this for a folder called `my_folder`:
    1. Click "Create New File" on the main repo page.
    2. In the file name, type `my_folder/README.md`. Populate the contents with a brief description of what this folder will be used for (a one-liner is fine).
    3. Click "Commit new file".
6. __Education Account__: Your new repository is by default __public__/open to the world. I recommend you keep it this way. If you have time, and are interested in making the repository private without having to pay, try [requesting an Education account](https://education.github.com/).
    - _If you do make your repo private_, you'll need to [add the teaching team as collaborators](https://help.github.com/articles/inviting-collaborators-to-a-personal-repository/) to your repo so that we can evaluate your participation. See the course [README](../README.md) for our github usernames. 

## Evaluating questions

For this exercise, I'll make an Issue on the BAIT509 repo soliciting your comments regarding the quality of some online questions/discussions.

1. Find a question/issue or two that someone has posed online. Ideally, one should also have a response. See below for some example sites.
2. Add a comment to the BAIT509 Issue. For each thread you examined, specify:
    - A link to the thread/question
    - In what ways is the question/issue worded effectively/ineffectively? Why? What would make it better, if anything?
        - Note that people often reply with this type of feedback, too! Sometimes not in a pleasant way, unfortunately.
    - If there is a discussion surrounding this question/issue, did any comments add value to the discussion? Were any supportive? Were any destructive? How so?

We'll talk about some examples after you're done. Here are some sites you might find useful:

- [`ggplot2` GitHub Issues](https://github.com/tidyverse/ggplot2/issues) (a popular R package)
- [`scipy` GitHub Issues](https://github.com/scipy/scipy/issues) (a popular python package)
- [Stack Overflow](https://stackoverflow.com/)


## R

You'll need various tools to interact with R in this course.

1. If you haven't already, install R and RStudio.
    - RStudio is the editor we'll be using to code in R. You can use something else if you want, but I highly recommend not.
2. Install R packages by running these commands in the console (these are used at least for this manuscript -- we'll encounter more throughout the course).
    - `install.packages("tidyverse")` -- a suite of useful packages for data science.
    - `install.packages("ISLR")` -- the package to accompany the ISLR book.
    - `install.packages("knitr")`


# Introduction to Machine Learning

In this section, we'll go over some basic machine learning concepts.

## What machine learning is

What is Machine Learning (ML) (or Statistical Learning)? As the [ISLR book](http://www-bcf.usc.edu/~gareth/ISL/) puts it, it's a "vast set of tools for understanding data". Before we explain more, we need to consider the two main types of ML:

- __Supervised learning__. (_This is the focus of BAIT 509_). Consider a "black box" that accepts some input(s), and returns some type of output. Feed it a variety of input, and write down the output each time (to obtain a _data set_). _Supervised learning_ attempts to learn from these data to re-construct this black box. That is, it's a way of building a forecaster/prediction tool. 

You've already seen examples throughout MBAN. For example, consider trying to predict someone's wage (output) based on their age (input). Using the `Wage` data set from the `ISLR` R package, here are examples of inputs and outputs:


 age        wage
----  ----------
  18    75.04315
  24    70.47602
  45   130.98218
  43   154.68529
  50    75.04315
  54   127.11574

We try to model the relationship between age and wage so that we can predict the salary of a new individual, given their age. 

An example supervised learning technique is _linear regression_, which you've seen before in BABS 507/508. For an age `x`, let's use linear regression to make a prediction that's quadratic in `x`. Here's the fit:

<img src="cm01-intro_files/figure-html/unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

The blue curve represents our attempt to "re-construct" the black box by learning from the existing data. So, for a new individual aged 70, we would predict a salary of about \$100,000. A 50-year-old, about \$125,000.


- __Unsupervised learning__. (_BAIT 509 will not focus on this_). Sometimes we can't see the output of the black box. _Unsupervised learning_ attempts to find structure in the data without any output. 

For example, consider the following two gene expression measurements (actually two principal components). Are there groups that we can identify here?

<img src="cm01-intro_files/figure-html/unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

You've seen methods for doing this in BABS 507/508, such as k-means. 


## Variable terminology

In supervised learning:

- The output is a random variable, typically denoted $Y$. 
- The input(s) variables (which may or may not be random), if there are $p$ of them, are typically denoted $X_1$, ..., $X_p$ -- or just $X$ if there's one. 

There are many names for the input and output variables. Here are some (there are more, undoubtedly):

- __Output__: response, dependent variable. 
- __Input__: predictors, covariates, features, independent variables, explanatory variables, regressors. 

In BAIT 509, we will use the terminology _predictors_ and _response_.

## Variable types

Terminology surrounding variable types can be confusing, so it's worth going over it. Here are some non-technical definitions. 

- A __numeric__ variable is one that has a quantity associated with it, such as age or height. Of these, a numeric variable can be one of two things:
    - A variable is __continuous__ if you can increase/decrease the variable "smoothly". For example, temperature, or proportions.
    - A variable is __discrete__ if increasing/decreasing the variable can only happen "in steps". For example, number of apples in the fridge.
- A __categorical__ variable, as the name suggests, is a variable that can be one of many categories. For example, type of fruit; success or failure.  

## Types of Supervised Learning

There are two main types of supervised learning methods -- determined entirely by the type of response variable.

- __Regression__ is supervised learning when the response is numeric.
- __Classification__ is supervised learning when the response is categorical. 

We'll examine both equally in this course. 

Note: Don't confuse classification with _clustering_! The latter is an unsupervised learning method.

## What have you heard of so far?

Let's write down some supervised learning techniques that you've heard of. Are there any that you particularly are interested in learning about?

# Irreducible Error

The concept of __irreducible error__ is paramount to supervised learning. Next time, we'll look at the concept of _reducible_ error. 

When building a supervised learning model (like linear regression), we can never build a perfect forecaster -- even if we have infinite data!

Let's explore this notion. When we hypothetically have an infinite amount of data to train a model with, what we actually have is the _probability distribution_ of $Y$ given any value of the predictors. The uncertainty in this probability distribution is the __irreducible error__.

__Example__: Let's say $(X,Y)$ follows a (known) bivariate Normal distribution. Then, for any input of $X$, $Y$ has a _distribution_. Here are some examples of this distribution for a few values of the predictor variable (these are called _conditional_ distributions, because they're conditional on observing particular values of the predictors).

<img src="cm01-intro_files/figure-html/unnamed-chunk-5-1.png" style="display: block; margin: auto;" />

This means we cannot know what $Y$ will be, no matter what! What's one to do?

- In __regression__ (i.e., when $Y$ is numeric, as above), the go-to standard is to predict the _mean_ as our best guess. 
    - We typically measure error with the __mean squared error__ = average of (observed-predicted)^2. 
- In __classification__, the conditional distributions are categorical variables, so the go-to standard is to predict the _mode_ as our best guess (i.e., the category having the highest probability). 
    - A typical measurement of error is the __error rate__ = proportion of incorrect predictions.
    - A more "complete" picture of error is the __entropy__, or equivalently, the __information measure__. 

In Class Meeting 07, we'll look at different options besides the mean and the mode.

An important concept is that _predictors give us more information about the response_, leading to a more certain distribution. In the above example, let's try to make a prediction when we don't have knowledge of predictors. Here's what the distribution of the response looks like:

<img src="cm01-intro_files/figure-html/unnamed-chunk-6-1.png" style="display: block; margin: auto;" />

This is much more uncertain than in the case where we have predictors!

# In-class Exercises: Irreducible Error

Note: if you don't have `git` set up on your computer (this is different from github), for now, just drag-and-drop your R file to your github repo to add it to your repo. 

## Oracle regression

Suppose you have two independent predictors, $X_1, X_2 \sim N(0,1)$, and the conditional distribution of $Y$ is
$$ Y \mid (X_1=x_1, X_2=x_2) \sim N(5-x_1+2x_2, 1). $$
From this, it follows that:

- The conditional distribution of $Y$ given _only_ $X_1$ is
$$ Y \mid X_1=x_1 \sim N(5-x_1, 5). $$
- The conditional distribution of $Y$ given _only_ $X_2$ is
$$ Y \mid X_2=x_2 \sim N(5+2x_2, 2). $$
- The (marginal) distribution of $Y$ (not given any of the predictors) is
$$ Y \sim N(5, 6). $$

The following R function generates data from the joint distribution of $(X_1, X_2, Y)$. It takes a single positive integer as an input, representing the sample size, and returns a `tibble` (a fancy version of a data frame) with columns named `x1`, `x2`, and `y`, corresponding to the random vector $(X_1, X_2, Y)$, with realizations given in the rows. 


```r
genreg <- function(n){
    x1 <- rnorm(n)
    x2 <- rnorm(n)
    eps <- rnorm(n)
    y <- 5-x1+2*x2+eps
    tibble(x1=x1, x2=x2, y=y)
}
```


1. Generate data -- as much as you'd like.


```r
dat <- genreg(1000)
```


2. For now, ignore the $Y$ values. Use the means from the distributions listed above to predict $Y$ under four circumstances:
    1. Using both the values of $X_1$ and $X_2$.
    2. Using only the values of $X_1$.
    3. Using only the values of $X_2$.
    4. Using neither the values of $X_1$ nor $X_2$. (Your predictions in this case will be the same every time -- what is that number?)
    

```r
dat <- mutate(dat,
       yhat = 5,
       yhat1 = 5-x1,
       yhat2 = 5+2*x2,
       yhat12 = 5-x1+2*x2)
```
    

3. Now use the actual outcomes of $Y$ to calculate the mean squared error (MSE) for each of the four situations. 
    - Try re-running the simulation with a new batch of data. Do your MSE's change much? If so, choose a larger sample so that these numbers are more stable.
    

```r
(mse <- mean((dat$yhat - dat$y)^2))
```

```
## [1] 6.468105
```

```r
(mse1 <- mean((dat$yhat1 - dat$y)^2))
```

```
## [1] 5.248613
```

```r
(mse2 <- mean((dat$yhat2 - dat$y)^2))
```

```
## [1] 2.014189
```

```r
(mse12 <- mean((dat$yhat12 - dat$y)^2))
```

```
## [1] 0.9581933
```

```r
knitr::kable(tribble(
    ~ Case, ~ MSE,
    "No predictors", mse,
    "Only X1", mse1,
    "Only X2", mse2,
    "Both X1 and X2", mse12
))
```



Case                    MSE
---------------  ----------
No predictors     6.4681051
Only X1           5.2486131
Only X2           2.0141887
Both X1 and X2    0.9581933

    
4. Order the situations from "best forecaster" to "worst forecaster". Why do we see this order?

> They're ordered from worst to best in the above table. Adding more predictors reduces the error. Adding $X_2$ is better than adding $X_1$ (as seen by the lower MSE), because it's carries more information about $Y$ (i.e., it's more dependent with $Y$ than $X_1$ is).



## Oracle classification

Consider a categorical response that can take on one of three categories: _A_, _B_, or _C_. The conditional probabilities are:
$$ P(Y=A \mid X=x) = 0.2, $$
$$ P(Y=B \mid X=x) = 0.8/(1+e^{-x}), $$

To help you visualize this, here is a plot of $P(Y=B \mid X=x)$ vs $x$ (notice that it is bounded above by 0.8, and below by 0).


```r
ggplot(tibble(x=c(-7, 7)), aes(x)) +
    stat_function(fun=function(x) 0.8/(1+exp(-x))) +
    ylim(c(0,1)) +
    geom_hline(yintercept=c(0,0.8), linetype="dashed", alpha=0.5) +
    theme_bw() +
    labs(y="P(Y=B|X=x)")
```

<img src="cm01-intro_files/figure-html/unnamed-chunk-11-1.png" style="display: block; margin: auto;" />

Here's an R function to generate data for you, where $X\sim N(0,1)$. As before, it accepts a positive integer as its input, representing the sample size, and returns a tibble with column names `x` and `y` corresponding to the predictor and response. 


```r
gencla <- function(n) {
    x <- rnorm(n) 
    pB <- 0.8/(1+exp(-x))
    y <- map_chr(pB, function(t) 
            sample(LETTERS[1:3], size=1, replace=TRUE,
                   prob=c(0.2, t, 1-t-0.2)))
    tibble(x=x, y=y)
}
```


1. Calculate the probabilities of each category when $X=1$. What about when $X=-2$? With this information, what would you classify $Y$ as in both cases?
    - BONUS: Plot these two conditional distributions. 


```r
## X=1:
(pB <- 0.8/(1+exp(-1)))
```

```
## [1] 0.5848469
```

```r
(pA <- 0.2)
```

```
## [1] 0.2
```

```r
(pC <- 1 - pB - pA)
```

```
## [1] 0.2151531
```

```r
ggplot(tibble(p=c(pA,pB,pC), y=LETTERS[1:3]), aes(y, p)) +
    geom_col() +
    theme_bw() +
    labs(y="Probabilities", title="X=1")
```

<img src="cm01-intro_files/figure-html/unnamed-chunk-13-1.png" style="display: block; margin: auto;" />

```r
## X=-2
(pB <- 0.8/(1+exp(-(-2))))
```

```
## [1] 0.09536234
```

```r
(pA <- 0.2)
```

```
## [1] 0.2
```

```r
(pC <- 1 - pB - pA)
```

```
## [1] 0.7046377
```

```r
ggplot(tibble(p=c(pA,pB,pC), y=LETTERS[1:3]), aes(y, p)) +
    geom_col() +
    theme_bw() +
    labs("Probabilities", title="X=-2")
```

<img src="cm01-intro_files/figure-html/unnamed-chunk-13-2.png" style="display: block; margin: auto;" />

> When $X=1$, _B_ has the highest probability, so we predict that. When $X=-2$, _C_ has the highest probability, so we predict that.

2. In general, when would you classify $Y$ as _A_? _B_? _C_?

> For any $X<0$, the probability of _C_ will be the largest, so we predict that. For any $X>0$, the probability of _B_ will be the largest, so we predict that.

3. Generate data -- as much as you'd like.


```r
dat2 <- gencla(1000)
```

4. For now, ignore the $Y$ data. Make predictions on $Y$ from $X$.


```r
dat2$yhat <- if_else(dat2$x<0, "C", "B")
```


5. Now, using the true $Y$ values, calculate the error rate. What type of accuracy do you get?


```r
1-mean(dat2$yhat == dat2$y)
```

```
## [1] 0.449
```


## (BONUS) Random prediction

You might think that, if we know the conditional distribution of $Y$ given some predictors, why not take a random draw from that distribution as our prediction? After all, this would be simulating nature.

The problem is, this prediction doesn't do well. 

Re-do the regression exercise above (feel free to only do Case 1 to prove the point), but this time, instead of using the mean as a prediction, use a random draw from the conditional distributions. Calculate the MSE. How much worse is it? How does this error compare to the original Case 1-4 errors?

## (BONUS) A more non-standard regression

The regression example given above is your perfect, everything-is-linear-and-Normal world. Let's see an example of a joint distribution of $(X,Y)$ that's _not_ Normal. 

The joint distribution in question can be respresented as follows:
$$ Y|X=x \sim \text{Beta}(e^{-x}, 1/x), $$
$$ X \sim \text{Exp}(1). $$

Write a formula that gives a prediction of $Y$ from $X$ (you might have to look up the formula for the mean of a Beta random variable). Generate data, and evaluate the MSE. Plot the data, and the conditional mean as a function of $x$ overtop. 

## (BONUS) Oracle MSE

What statistical quantity does the mean squared error (MSE) reduce to when we know the true distribution of the data? Hint: if each conditional distribution has a certain variance, what then is the MSE?

What is the error rate in the classification setting?

# Lab: Tooling, Part 2

Be sure to let us know what your github username is by [filling out the survey](https://goo.gl/forms/86bIrgJSUvJBVdef2), if you haven't already!

## Git

Get `git` working on your local computer. You don't _really_ have to do this -- you can get by just by using GitHub and drag-and-drop -- but you might find this method clumsy. So maybe skip ahead for now, unless you're determined. Plus, it's a good skill to be able to use `git`. 

- If you haven't already, [install `git`](http://happygitwithr.com/install-git.html) on your computer. Then [introduce yourself to `git`](http://happygitwithr.com/hello-git.html).
- If you want, [install a `git` client](http://happygitwithr.com/git-client.html) (I like to use [sourcetree](https://www.sourcetreeapp.com/)). If you don't use a git client, the alternative is to use the command line.
- Try cloning your repo to your computer (I'll let you google that).
- Make a change to a file in that repo, locally. Commit the change. Then push the change.

__Note__ (merge conflicts): if you have a local cloned copy of your github repo, you can run into problems if you make changes to both the github version of the repo, and the local version. Though, you probably won't experience this unless you change the same lines in the same file. If you do this, you'll encounter a "merge conflict" when you try to push your changes. You'll have to fix this by hand, choosing which text to keep, and what to remove. 

## RMarkdown

Get cozy with writing in RMarkdown! You should be able to: 

- Make code chunks and write R code in them.
- `Knit` an RMarkdown document to either `.pdf`, `.html`, or `.md`.
- Ideally, write the "English part" of the document in markdown. 
    - [This tutorial](https://www.markdowntutorial.com/) will get you up to speed in 10 minutes.
    - Original documentation for markdown can be found on the [Daring Fireball site](https://daringfireball.net/projects/markdown/). 

I recommend reading the [Happy git with R book, Section IV](http://happygitwithr.com/rmd-test-drive.html) to get comfortable with these concepts. For now, let's see if you can upload/push an RMarkdown (`.Rmd`) file to your repo. In addition:

1. Submit a `Knit`ted file as a...
    - Level 1 challenge: `.html` file; 
    - Level 2 challenge: `.pdf` file; 
    - Level 3 challenge: `.md` file with github-flavoured markdown.
2. As you write your RMarkdown file, include your own R code in a chunk that does something -- anything. `2+2` is fine. A plot is better. 
3. Use at least 6 markdown features in your text (such as bold, italics, adding a hyperlink, header, etc.). 

## Jupyter Notebooks

We expect you to submit your assignments (at this point, at least some of them) in the form of a [_jupyter notebook_](http://jupyter.org/). Do something similar as you did with the RMarkdown document above:

1. Install Jupyter if you haven't already. Doing this via the anaconda distribution is recommended. 
2. Make a new jupyter notebook file. Add some markdown chunks; add some code chunks. Run the document.
3. Upload/push your jupyter notebook file to your github repo. 
