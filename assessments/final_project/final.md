# BAIT 509 Final Project

__Due date__: Tuesday, April 3, at 24:00. 

__Submission instructions__: All group members should make a complete submission to Connect by the deadline. 

__Group assignments__: Groupings are random. See the file on Connect for the final project groups.

Your task is to build a supervised learning model to address a hypothetical (or real, if you have one) business question, in the form of a report, in a group.

We will provide some datasets for you to use (see Section 2 below). You can use your own data, but only after getting it approved by a member of the teaching team. 

The audience for the majority of the report is the BAIT 509 teaching team. This means you can (and should) use machine learning jargon, so that the teaching team can evaluate your understanding more clearly. The secondary audience is a hypothetical client who is not an expert in supervised learning, and applies _only_ for the advice/conclusions section (indicated below).

## 1. Components of the report

Your report should communicate a hypothetical business question, a refined statistical question (or more than one, if appropriate) that can be addressed by supervised learning, followed by a model (or more than one model, if appropriate). How you do this is up to you, though we've provided some guidance below for various components of the report that you should include. These don't necessarily have to correspond to the sections you include in your report -- the amount and number of sections that you use is up to you, but keep in mind that you will be graded for conciseness and readability.

### Background and Motivation

Briefly convey the (hypothetical) business question motivating your project, and describe the available data. If you're using your own data, be sure to also describe the necessary background for the BAIT 509 teaching team to understand the data and the problem.

At some point before you describe your model(s), you should plot your data -- probably, lots of plots for your own eyes, and a few useful ones to include in the report. This is important to get a sense of what the data are doing, what predictors might be useful, and whether there are potential issues with the data.

### A discussion about questions

Identification of the business question and corresponding statistical question. You are only expected to ask one statistical question. Be sure to include the following discussion points:

- In what ways is this statistical question useful? That is, in what ways does it contain the essence of the business question? 
- In what ways does this statistical question fall short? That is, in what ways does it miss the essence of the business question?

### A discussion about the model(s)

Fit one (or more, if appropriate) model using supervised learning to answer your statistical question. You can use any supervised learning method -- not just the ones discussed in BAIT 509. You should _try_ fitting a few different models, though, including at least one that we tried in class, and weed-out the ones that are not useful.

Include a discussion on why you are proposing this model in particular, over others. Be sure to include the following components (if one is not relevant, at least say so):

- Quantitative choice: This includes using numeric scores such as MSE to choose an optimal model (such as choosing your tuning parameters, and/or choosing between different types of models altogether, like a kNN model vs a random forest model).
- Qualitative choice: There are some decisions/assumptions in the modelling process that we can justify based on what we see in the data after plotting it, or based on things we know to be true (such as scientific relationships between variables).
- Human choice: Does one model just work better on the human level compared to others? For example, perhaps it's important to have an interpretable model, in which case models such as kNN, loess, and ensembles may not be appropriate (but linear regression and trees might be). 

It might not be relevant to pick _one_ model, per se. For example, it's common to fit several models to see that they all point to the same conclusion. And if one (or some) don't, it would be useful to discuss why this might be, and whether we should take this to heart when we try to draw overall conclusions. 


### Communication of results, and advice to a non-expert

The last component of your analysis is to communicate your results. _Here is where you are expected to write for a hypothetical client who is not an expert in supervised learning_. As such, you should not use jargon.

Provide insight/advice on the original business question, using results from your model(s). This probably means explaining what your machine learning model does, and how it can be used, especially in light of the business question.

Is your model something that would be useful bundled up as an app? Or perhaps as part of some sort of computational pipeline? You aren't expected to take this project as far as developing software around the app, but if some other product besides a report might be useful, say so.

### Analysis details

Be sure to also provide enough information so that a reader can reproduce your analysis. This includes code, and instructions on how to run it. This info should probably _not_ go in the body of the report (lest it gets cluttered). An Appendix is useful for this purpose. Code is usually best left as separate files, in which case, you should include a README file that explains what the various files are, and how to navigate the files.

## 2. Data Examples

Here are some sample datasets that you're free to use for your project. 

* [Marshall Violent Crime](./data/marshall/)
* [College Scorecard](https://collegescorecard.ed.gov/data/)
* [Tech Mental Health Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)
* [Zillow real-estate data](https://www.kaggle.com/zillow/zecon)
* [Wine ratings](https://www.kaggle.com/zynicide/wine-reviews/data)
* River flow data (in the `data` folder; see details below)

I'll leave it up to your group to come up with a hypothetical business question. It doesn't have to be strictly business related, as long as it's some big-picture question. 

Below is an example problem setup, using the river flow data. Feel free to use it as your project setup.

### Flood Forecasting

__Business question__: Your client, an environmental consulting firm, wants to predict floods of the Bow River at Banff, Alberta one or two days in advance. Suppose your client is currently using the last-value-carried-forward method to forecast, and wants to see "if you can do better" (whatever that means). If it matters, a river discharge of about 200m^3/s is when we start to get concerned about flooding (this number is made up -- you can change it if you want, or make your own situation). 

__Data__: You have daily observations of river discharge starting March 14, 1984 to the end of 2014. Data are from the Water Survey of Canada (Environment and Climate Change Canada). You can find the data in the `flow.csv` file.

__Hints__:

- Some things that are predictive of river flow are the day of year, and lagged flow (that is, to predict the river flow tomorrow or the day after, then today's flow, yesterday's flow, etc. are predictive).
- The `lubridate` package in R is useful for working with dates. In particular, use the `yday()` function to obtain the "day of year" from the date. The `year()` function is also useful -- it extracts the year from a date.
- The `lead` and `lag` function from the `dplyr` package in R are useful for lagging (or "leading", the opposite) a vector. 

## 3. Grading

Your submission will be graded based on the following rubric. All group members will receive the same grade. 

- __Tidy submission, and navigability (5%)__
    - _All group members' full names are on the report_.
    - Your report should be easy to navigate. This includes being arranged/organized in a sensible way. 
    - If you have multiple files, they should be easy to navigate, and should be easy to find out what's what, through a README file. 
    - Your submission is complete, and does not require compiling.
    - Your submission is not different from other members in the group (at least, from the majority submission).
- __Low-level writing (10%)__
    - Your report should be concise and informative, not cluttered with irrelevant information. (Note that the Appendix is a more foregiving section, as it's usually reserved for lengthier, more details things that one would not be interested in unless they want to dig deeper into your analysis). 
    - You should use proper English, spelling, and grammar.
    - See the [MDS writing rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_writing.md) if you want more details about what we'll be looking for. 
- __High-level writing (10%)__
    - When writing to the hypothetical non-expert client, you should be informative, yet not use machine learning jargon.
- __Code (20%)__
    - See the [MDS code rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_code.md) for details about what we'll be looking for when examining your analysis code. 
- __Reasoning (55%)__
    - Discussion about the statistical question is insightful.
    - Discussion on the model choice is sensible and insightful. Machine learning concepts are well understood.
    - Other discussions pertaining to the business question are useful and insightful. 
    - Note that the weight of various discussions within this category will vary, but generally, the discussion of machine learning models will carry the highest weight -- perhaps 4/5 machine learning, 1/5 statistical question (depending on the complexity of the business question). For more details on what we're looking for, see the [MDS reasoning rubric](https://github.com/UBC-MDS/public/blob/master/rubric/rubric_reasoning.md).
