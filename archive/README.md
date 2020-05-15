# BAIT509 - Business Applications of Machine Learning

This is the home page for the 2019 iteration of the course BAIT 509 at the University of British Columbia, Vancouver, Canada. The core syllabus can be found at [sauder_syllabus.pdf](sauder_syllabus.pdf), but anything listed on this website will take precedence.

This repository is avaiable as a [website](https://bait509-ubc.github.io/BAIT509/).

## Learning Objectives

By the end of the course, students are expected to be able to:

- Explain what machine learning (ML) is, in the context of errors and model functions;
- Understand and implement the machine learning paradigms in both R and python for a variety of ML methods;
- Identify a data table based on a machine learning problem;
- Understand the types of error, and how this influences model choice/goodness;
- Build and justify a ML model; and,
- Understand how ML fits into the greater scope of solving a business problem.

## Teaching Team

At your service!

| Name         | Position   | GitHub Handle | 
| :---:        | :---:      | :---:         |
| [Tomas Beuzen](https://tomasbeuzen.github.io/) | Instructor | @tbeuzen      |
|              | TA         |               |
|              | TA         |               |
|              | TA         |               |

## Class Meetings

Details about class meetings will appear here as they become available. Readings are optional, but should be useful. 

|  #   | Topic | Recommended Readings |
|------|-------|-------|
| [cm01](/class_meetings/cm01-intro.md); worksheet ([.R](/class_meetings/cm01-worksheet.R)) | Intro to the course, tools, and ML | [ISLR](http://www-bcf.usc.edu/~gareth/ISL/) Section 2.1 |
| [cm02](/class_meetings/cm02-error.md); worksheet ([.html](/class_meetings/cm02-worksheet.html) / [.Rmd](/class_meetings/cm02-worksheet.Rmd)) | Irreducible and Reducible Error | [ISLR](http://www-bcf.usc.edu/~gareth/ISL/) Section 2.2 (you can stop in 2.2.3 once you get to the "The Bayes Classifier" subsection). |
| [cm03](/class_meetings/cm03-local.md); model fitting in python ([.html](/class_meetings/cm03-model_fitting-python.html) / [.ipynb](/class_meetings/cm03-model_fitting-python.ipynb)); model fitting in R ([.html](/class_meetings/cm03-model_fitting-r.html) / [.Rmd](/class_meetings/cm03-model_fitting-r.Rmd)) | Local methods | [ISLR](http://www-bcf.usc.edu/~gareth/ISL/)'s "K-Nearest Neighbors" section (in Section 2.2.3) on page 39; and Section 7.6 ("Local Regression"). |
| [cm04](/class_meetings/cm04-selection.md); cross-validation example ([.R](/class_meetings/cm04-worksheet.R)) | Model Selection | [ISLR](http://www-bcf.usc.edu/~gareth/ISL/) Section 5.1; we'll be touching on 6.1, 6.2, and 6.3 from [ISLR](http://www-bcf.usc.edu/~gareth/ISL/), but only briefly. |
| [cm05](/class_meetings/cm05-trees.md); CART example ([.R](/class_meetings/cm05-worksheet.R)) | Classification and Regression Trees | [ISLR](http://www-bcf.usc.edu/~gareth/ISL/) Section 8.1 |
| [cm06](/class_meetings/cm06-questions.md); model function example ([.R](/class_meetings/cm06-worksheet.R)) | Refining business questions | [This blog post by datapine](https://www.datapine.com/blog/data-analysis-questions/) does a good job motivating the problem of asking good questions. [This blog post by altexsoft](https://www.altexsoft.com/blog/business/supervised-learning-use-cases-low-hanging-fruit-in-data-science-for-businesses/) does a good job outlining the use of supervised learning in business. |
| [cm07](/class_meetings/cm07-ensembles.md); random forest example ([.R](/class_meetings/cm07-worksheet.R)) | Ensembles | [ISLR](http://www-bcf.usc.edu/~gareth/ISL/) Section 8.2 |
| [cm08](/class_meetings/cm08-beyond_mean_mode.md); worksheet ([.R](/class_meetings/cm08-worksheet.R)) | Beyond the mean and mode | |
| [cm09](/class_meetings/cm09-svm.md) (worksheet a continuation of yesterday's) | SVM | Section 9.1, 9.2, 9.4 in [ISLR](http://www-bcf.usc.edu/~gareth/ISL/). The details aren't all that important. 9.3 is quite advanced, but I'll be discussing the main idea behind it in class. |
| [cm10](/class_meetings/cm10.md) SVM and cross validation worksheet ([.ipynb](https://raw.githubusercontent.com/vincenzocoia/BAIT509/master/class_meetings/cm10-worksheet.ipynb)) | SVM continuation; wrapup; alternatives to accuracy | [Alternative measures](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/), and [ROC](https://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/) |

## Office Hours

Want to talk about the course outside of lecture? Let's talk during these dedicated times.

| Teaching Member | When                 | Where    |
| :---:           | :---:                | :---:    |
| Tomas Beuzen    | Tuesdays 13:00-14:00 | ESB 1045 |
|                 |                      |          |
|                 |                      |          |
|                 |                      |          |
|                 |                      |          |
|                 |                      |          |


## Assessments

Links to assessments will be made available when they are ready. The deadlines listed here are the official ones, and take precendence over the ones listed in the [sauder syllabus](https://github.com/vincenzocoia/BAIT509/blob/master/sauder_syllabus.pdf).

| Assessment        | Due    | Weight |
|:---:              |:---:   |:---:   |
| [Assignment 1](/assessments/assignment1/assignment1.html) ([.ipynb](/assessments/assignment1/assignment1.ipynb))  | January ~~12~~ ~~17~~ 19 at 18:00 | 20% | 
| [Assignment 2](/assessments/assignment2/assignment2.md)  | January 26 at 18:00 | 20% |
| [Assignment 3](/assessments/assignment3/assignment3.md)  | February 2 at 18:00 | 20% |
| [Final Project](/assessments/project/project.md) | February 8 at 23:59 | 30% |
| [Participation](/assessments/participation/participation.md) | January 31 at 18:00 | 10% |

Please submit everything to [UBC Canvas](https://canvas.ubc.ca/).

## Additional Resources

- [An Introduction to Statistical Learning with R](http://www-bcf.usc.edu/~gareth/ISL/)
- [Data Mining: Practical Machine Learning Tools and Techniques](https://www.cs.waikato.ac.nz/ml/weka/book.html)
- [A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [A Course in Machine Learning](http://ciml.info/)
