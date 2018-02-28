# BAIT 509 Final Project

__Due date__: Tuesday, April 3, at 24:00. 

Your task is to address a business question using machine learning, in the form of a report. 

We will provide some datasets for you to use, along with a business question, but feel free to use your own data and business question if you like -- just be sure to get it approved by the instructor.

The primary audience of this report is the BAIT 509 teaching team. The secondary audience is a hypothetical client who is not an expert in supervised learning, and applies _only_ for the advice/conclusions section.

## 1. Components

Your report should address the components discussed in Class Meeting 06. How you do this is up to you, though we've provided some guidance below for each component. The amount and number of sections that you use in your report is up to you. 

### Background and Motivation

Briefly convey the business question motivating your project, and describe the available data. If you're using your own data, be sure to also describe the necessary background for the BAIT 509 teaching team to understand the problem.

### A discussion about questions

Identification of the business question and corresponding statistical question. You are only expected to ask one statistical question, but identifying more than one indicates that you've gone above and beyond. Be sure to include the following discussion points:

- In what ways is this statistical question useful? That is, in what ways does it contain the essence of the business question? 
- In what ways does this statistical question fall short? That is, in what ways does it miss the essence of the business question?

### A discussion about the model(s)

Fit one (or more, if appropriate) model using supervised learning to answer your statistical question. You can use any supervised learning method -- not just the ones discussed in BAIT 509. Include a discussion on why you are proposing this model in particular, over others. Be sure to include the following two components:

- Human choice: There are some decisions in the modelling process that we can justify based on logic, or based on things we know to be true in the real world. This may be just cause for instigating model assumptions, or believing in certain supervised learning methods over others.
- Numerical choice: This might include parameter estimation, and possibly the selection of one model over others using numerical scores. 

Be sure to also provide enough information so that a reader can reproduce your results/model fits. You can put information in an Appendix if this requires too much detail.

It might not be relevant to pick _one_ model, per se. For example, it's common to fit several models to see that they all point to the same conclusion. And if one (or some) don't, it would be useful to discuss why this might be, and whether we should take this to heart when we try to draw overall conclusions. 

### Communication of results, and advice to a non-expert

The last component of your analysis is to communicate your results. _Here is where you are expected to write for a hypothetical client who is not an expert in supervised learning_. As such, you should not use jargon.

Provide insight/advice on the original business question, using results from your model(s). It might also be relevant to explain what exactly your model predicts. 


## 2. Data Examples

### Flood Forecasting

Your client, an environmental consulting firm, wants to predict floods one or two days in advance. Suppose your client is currently using the last-value-carried-forward method to forecast, and wants to see "if you can do better" (whatever that means). 

Hint: The `lubridate` package in R is useful for working with dates. In particular, use the `yday()` function to obtain the "day of year" from the date. The `year()` function is also useful -- it extracts the year from a date.
