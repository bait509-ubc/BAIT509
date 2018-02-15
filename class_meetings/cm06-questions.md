Forming Good Scientific Questions
================

Today, we're going to take a break from supervised learning methods, and look at the process involved to address a question/problem faced by an organization by data science.

Generally, four parts of the process are important to recognize as you work on the project. In order from high to low level:

1.  The Business Question
2.  The Statistical Question(s)
3.  The data and model
4.  The data product

Doing an analysis is about distilling from the highest level to the lowest level. As such, there are three distillations to keep in mind: 1-2, 2-3, and 3-4.

-   1-2 is about asking the right questions.
-   2-3 is about building a useful model.
-   3-4 is about communicating the results.

But the key to remember is that an analysis doesn't (*and shouldn't*) be a linear progression from the highest level to the lowest, but that the process is iterative, revisiting any of the distillations at any point after embarking on the project. This is because *none of the components are independent*. Making progress on any of the three distillations gives you more information as to what the problem is.

There is a procedure for distilling higher level components to lower level.

1.  Distilling business questions to statistical is about
2.  Distilling statistical question to a model is

In all of this, it's most beneficial to consider developing your analysis by small, simple steps. Always start with a basic approach, and gradually tweak the various components. Strive to get some sort of crude end product that resembles what your final product might look like, right off the bat. Not only is it more useful to have a "working version" of your product right off the bat, but it gives you more information about the project that will inform other distillations.

#### Readings

[This blog post by datapine](https://www.datapine.com/blog/data-analysis-questions/) does a good job motivating the problem of asking good questions, and how to address the problem.

Prerequisites
-------------

Before going into this, you need to know what data are available, and what state they're in. It would be ideal to make exploratory plots to get a feel for this.

1-2 Distilling Questions
------------------------

We'll talk about taking a problem faced by an organization, and how to distill that into something explorable. - A Business Question is a high-level question - A Statistical Question is a lower-level, more specific question that is addressable by machine learning/statistics.

Some main points:

-   The client is the expert on knowing what the overarching problem is.
-   You are the expert on knowing how to extract information from data.
-   Concept that the first part of an analysis is to distill a business question to a statistical question, and that this is an iterative process as you learn more about (and clarify) the overarching problem.
-   You should remember that their level of data science experience will vary. They may suggest a specific analysis for you to do, but never confuse this specific analysis for the overarching problem. Always
-   The statistical questions you form will depend on the data that are available. For example, you can't form a statistical question about the performance of a division of your business if there are no data available on that division, or if the data you collected aren't indicators of performance.

2-3 Distilling an analysis
--------------------------

The temptation here is to jump right into a complex analysis. Always start with a very basic approach -- perhaps just linear regression, for example.

3-4 Communicating
-----------------

The data product could be a report, a presentation, an app, a software package/pipeline, and any combination of these things. These almost always (or at least, almost always *should*) contain data visualizations.

Your client might request this outright, but their suggestion might not be the best option. Perhaps they request a report and presentation communicating your findings, when a more appropriate product also includes an interactive app that allows them to explore your findings for themselves.

Either way, the key here is communication. The challenges here include:

-   Communication with language: understanding your audience, and tailoring your language for them. This usually means no jargon allowed. The key is to talk more about the output and the general idea of your model(s), but not machine learning or statistical jargon.
-   Communication with visual design: this is about choosing what visuals are the most effective for communicating (called design choice).

This course is not about developing a data product. This is a topic that can be a discipline on its own, so we will not dwell any more on this.

Activity
--------

For today's practical component, you'll be working with your team to start your final projects for this course. Your objective is to come up with a proposal for your first embarkment on your project, and to present it.

1.  Refine the overarching problem into statistical questions.
2.  Propose some analyses that might be useful to answer those questions.
3.  Propose some data products that might be useful for a hypothetical end user.
    -   Keep in mind your end product is always only a report, because this course does not cover things such as app development and design, but do think about what *might* be useful.

As I mentioned, always start small and simple. But, it's fine to list some more agressive, hopeful goals. For each of the above, feel free to describe things from "simplest" to "most hopeful/wild success".
