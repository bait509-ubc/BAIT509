# BAIT 509 Final Assignment

UBC MBAN 2020.

This assignment was created by Tomas Beuzen and Vincenzo Coia.

### Table of Contents

1. [Overview](#1)
2. [Submission instructions](#2)
3. [Components of the report](#3)
4. [Example datasets](#4)
5. [Grading](#5)

## 1. Overview <a id=1></a>

For this final assignment you will work in groups to build a supervised learning model to address a hypothetical (or real, if you have one) business question in the form of a report. You will work in groups of 3 that have been randomly assigned. You will be provided with some datasets to use for your assignment (see [Section 4](#4) below) but you are also welcome to use your own data (but only after getting it approved by a member of the teaching team).

The audience for the majority of the report is the BAIT 509 teaching team. This means you should use machine learning jargon, so that the teaching team can clearly evaluate your understanding of the topics taught in this course. The secondary audience is a hypothetical client who is not an expert in supervised learning. This audience will read _only_ the advice/conclusions section of your report and your language should be adjusted accordingly in this section (the report structure is indicated below in [Section 3](#3)).

## 2. Submission instructions <a id=2></a>

**All group members should make a complete submission to Canvas by the deadline, 14th Feb 11:59pm**.

Your submission should contain all files necessary to reproduce your analysis, as well as your final report. A typical submission may include the following:

1. A README file;
2. Code files (e.g., a Jupyter notebook);
3. Data file(s);
4. Report (in .pdf or .html format).

You might also have some output files (like plots or other visualizations).

Note that a short, clear README file is extremely important for reproducibility. In it, you should aim to orient a hypothetical data scientist who stumbles upon your work, but doesn't know anything about the project. After reading the README, they should know what the project is about, what files are what, and how to reproduce your results. More on README files [here](https://www.makeareadme.com/).

## 3. Components of the report <a id=3></a>

Your report should communicate a hypothetical business question, a refined statistical question that can be addressed by supervised learning, followed by an implementation of a machine learning workflow and final model (or more than one model, if appropriate). How you do this is up to you, though we've provided some guidance below for various components of the report that you should include. These don't necessarily have to correspond to the sections you include in your report - the amount and number of sections that you use is up to you, but keep in mind that you will be graded for conciseness and readability. The report should be a **maximum of 10 pages long** (not including appendices) - do not write more than you need to.

### 3.1 Background and Motivation

Briefly convey the (hypothetical) business question motivating your project, and describe the data available to address the question. At some point before you describe your machine learning workflow and model, you should plot your data - probably, you will make lots of plots for your own eyes, but you should include only a few useful ones in the report. This is important to get a sense of what the data are doing, what predictors might be useful, and whether there are potential issues with the data.

### 3.2 A discussion about questions

Identify the business question and corresponding statistical question. You are only expected to ask/answer one statistical question in your report. Be sure to include the following discussion points:

- In what ways is this statistical question useful? That is, in what ways does it address the business question? 
- In what ways does this statistical question fall short? That is, in what ways does it not address the business question?

### 3.3 A discussion about your supervised learning workflow and model(s)

You should describe your supervised learning workflow/procedure for developing your model(s) to answer your statistical question. Your workflow might include the following:

- Feature pre-processing (optional)
- Feature selection (optional)
- Hyperparameter tuning (mandatory)
- Model selection (mandatory)

Idealy you should _try_ fitting several models using supervised learning to answer your statistical question. You can use any supervised learning method - not just the ones discussed in BAIT 509, but make sure you understand the model you are using, *do not use code that you do not understand*! Once you've selected your final model(s), you should include a discussion on your choice, in the context of the following components (if one is not relevant, at least say so):

- Quantitative choice: This includes using numeric scores such as error/accuracy (for classification) or mean-squared-error (for regression) to choose an optimal model. Mentions of training, validation and testing should be included somewhere in your report!
- Qualitative choice: There are some decisions/assumptions in the modelling process that we can justify based on what we see in the data after plotting it, or based on things we know to be true (such as scientific relationships between variables).
- Human choice: Does one model just work better on the human level compared to others? For example, perhaps it's important to have an interpretable model, in which case models such as kNN, loess, and ensembles may not be appropriate (but linear regression and trees might be). 

It might not be relevant to pick _one_ model, per se. For example, it's common to fit several models to see that they all point to the same conclusion. And if one (or some) don't, it would be useful to discuss why this might be, and whether we should take this to heart when we try to draw overall conclusions.

### 3.4 Communication of results, and advice to a non-expert

The last component of your analysis is to communicate your results. _Here is where you are expected to write for a hypothetical client who is not an expert in supervised learning_. As such, you should not use jargon.

Provide insight/advice on the original business question, using results from your model(s). This probably means explaining what your machine learning model does, and how it can be used, especially in terms of addressing the business question. Is your model something that would be useful bundled up as an app? Or perhaps as part of some sort of computational pipeline? You aren't expected to take this project as far as developing software, but if some other product besides a report might be useful, you should say so!

## 4. Example datasets <a id=4></a>

Here are some sample datasets that you're free to use for your project. You can find 

* [Marshall Violent Crime](https://github.com/themarshallproject/city-crime)
* [College Scorecard](https://collegescorecard.ed.gov/data/)
* [Tech Mental Health Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)
* [Zillow real-estate data](https://www.kaggle.com/zillow/zecon)
* [Wine ratings](https://www.kaggle.com/zynicide/wine-reviews/data)
* [College football team stats 2019](https://www.kaggle.com/jeffgallini/college-football-team-stats-2019)
* [Mental health in tech](https://www.kaggle.com/osmi/mental-health-in-tech-survey)
* [Ramen ratings dataset](https://www.kaggle.com/residentmario/ramen-ratings)
* [New York City AirBnB dataset](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
* [Avocado prices](https://www.kaggle.com/neuromusic/avocado-prices)
* [Kickstarter projects](https://www.kaggle.com/kemical/kickstarter-projects)
* [River flow data](https://github.com/bait509-ubc/BAIT509/blob/master/assignments/final_project/data/river_flow.csv); see example below.

It's up to your group to come up with a hypothetical business question based on a dataset. It doesn't have to be strictly business related, as long as it's some big-picture question. 

Below is an example problem setup, using the river flow data.

### 4.1 Example: Flood Forecasting

__Background and business question__:
- Your client, Alberta Environment and Parks, wants to predict floods of the Bow River at Banff, Alberta one or two days in advance (this is the made-up hypothetical business question).
- The client is currently using the "last-value-carried-forward" method to forecast (i.e., using yesterday's river discharge to predict today's discharge), and wants to see "if you can do better" (whatever that means).
- If it matters, a river discharge of about 200m^3/s is when we start to get concerned about flooding (this number is made up - you can change it if you want, or make your own situation). 

__Data and statistical question__:
- You have daily observations of river discharge starting March 14, 1984 to the end of 2014. Data are from the Water Survey of Canada (Environment and Climate Change Canada).
- You can use machine learning to predict tomorrow's river discharge (a regression problem) or to predict if there will be a flood tomorrow or not (classification problem). This is your response variable.
- Some things that are predictive of river flow are the day of year, and lagged flow (that is, to predict the river flow tomorrow or the day after, then information of today's flow, yesterday's flow, two-days-ago's flow, etc. are predictive). These could form your features.

__Hints__:

- Pandas has a lot of functionality for working with dates and times, leverage this functionality to extract useful features from the data like year, month, day, etc.
- Pandas also has functions such as `rolling` and `shift` which are useful for lagging a timeseries. 

## 5. Grading <a id=5></a>

Your submission will be graded based on the following rubric. _All group members will receive the same grade_. It is each student’s responsibility to ensure that all group members contribute equally to the assignment. In case of any group related issues, please discuss with the instructor.

- __Tidy submission, and navigability (5%)__
    - _All group members' full names are on the report_.
    - Your report should be easy to navigate. This includes being arranged/organized in a sensible way. 
    - Your project files should be easy to navigate, and it should be easy to find out what's what, through a README file. 
    - Your submission is complete, and does not require compiling.
- __Low-level writing (10%)__
    - Your report should be concise and informative, not cluttered with irrelevant information. (Note that the Appendix is a more foregiving section, as it's usually reserved for lengthier, more detailed things that one would not be interested in unless they want to dig deeper into your analysis). 
    - You should use proper English, spelling, and grammar.
    - See the [MDS writing rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_writing.md) if you want more details about what we'll be looking for. 
- __High-level writing (10%)__
    - When writing to the hypothetical non-expert client in the advice/conclusions section of your report, you should be informative, yet not use machine learning jargon.
- __Code (20%)__
    - See the MDS code [quality](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_quality.md) and [accuracy](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_accuracy.md) rubrics for details about what we'll be looking for when examining your analysis code. 
- __Reasoning (55%)__
    - Discussion about the statistical question is insightful.
    - Discussion on the model choice is sensible and insightful. Machine learning concepts are well understood.
    - Other discussions pertaining to the business question are useful and insightful. 
    - Note that the weight of various discussions within this category will vary, but generally, the discussion of machine learning models will carry the highest weight -- perhaps 4/5 machine learning, 1/5 statistical question (depending on the complexity of the business question). 
    - For more details on what we're looking for, see the [MDS reasoning rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_reasoning.md).
- __Plagiarism__
    - As always, plagiarism will be taken *extremely seriously* for this assignment.
    - This includes code plagiarism. See the heading "Code Plagiarism" in the [course syllabus](https://github.com/bait509-ubc/BAIT509/blob/master/BAIT509_syllabus.pdf).
    - You must cite all original sources that you use in your assignment.
    - If you're unsure about whether something constitues plagiarism or not, please speak with the instructor.