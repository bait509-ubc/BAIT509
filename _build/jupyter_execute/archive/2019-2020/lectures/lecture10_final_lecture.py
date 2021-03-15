# BAIT 509: Business Applications of Machine Learning
## Lecture 10 - Final Lecture
Tomas Beuzen, 3rd February 2020

# Lecture outline
- [0. Recap (5 mins)](#0)
- [1. Moving Forward with Machine Learning (5 mins)](#1)
- [2. A Pearl of Wisdom: Pandas Profiling (5 mins)](#2)
- [3. Remaining time is to work on your final project](#3)

# 0. Recap  (5 mins) <a id=0></a>

- Understand how the Random Forest algorithm works
- Discuss concepts of bagging and boosting
- Describe metrics other than error and r2 for measuring ML performance
- Instructor evaluations open [on Canvas](https://canvas.ubc.ca/courses/30777/external_tools/6073)

# 1. Moving Forward with Machine Learning (5 mins) <a id=1></a>

- If you're interested in continuing to learn/use ML I recommend the following:
    - Start doing [Kaggle competitions](https://www.kaggle.com/competitions)
    - Practice, practice, practice: try out these techniques on other datasets you've encountered in your studies, or in real life!
    - Go back and dig more into the theory with texts such as:
        - [Mathematics for Machine Learning](https://mml-book.github.io/)
        - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
        - [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
        - [Data Mining Practical Machine Learning Tools and Techniques](https://www.cs.waikato.ac.nz/ml/weka/book.html)

# 2. A Pearl of Wisdom: Pandas Profiling (5 mins) <a id=2></a>

- the magic bullet to EDA?
- quickly generate summaries of dataframes including dtypes, stats, visuals, etc.
- [Pandas profiling](https://github.com/pandas-profiling/pandas-profiling) is not part of base Pandas
- If using conda, install with: `conda install -c conda-forge pandas-profiling`

import pandas as pd
from pandas_profiling import ProfileReport

df = (pd.read_csv('data/weatherAUS.csv')
        .drop(columns=['Date', 'Location', 'RainTomorrow']))
profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile

# 3. Remaining time is to work on your final assignment <a id=3></a>

import matplotlib.pyplot as plt

plt.bar()
plt.xticks()