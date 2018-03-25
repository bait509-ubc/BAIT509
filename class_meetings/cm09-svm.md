---
title: 'BAIT 509 Class Meeting 09'
subtitle: "Support Vector Machines"
date: "Monday, March 26, 2018"
output: 
    html_document:
        keep_md: true
        toc: true
        toc_depth: 2
        number_sections: true
        theme: cerulean
        toc_float: true
---

# Overview

- Maximal Margin Classifier, and hyperplanes
- Support Vector Classifiers (SVC)
- Support Vector Machines (SVM)
- Extensions to multiple classes
- SVM's in python
- Feedback on Assignment 1
- Lab: work on Assignment 3

# The setup

Today, we'll dicuss a new method for __binary classification__ -- that is, classification when there are two categories. The method is called __Support Vector Machines__. We'll build up to it by considering two special cases:

1. The Maximal Margin Classifier (too restrictive to use in practice)
2. The Support Vector Classifier (linear version of SVM)

We'll demonstrate concepts when there are two predictors, because it's more difficult to visualize in higher dimensions. But concepts generalize.

Let's start by loading some useful R packages to demonstrate concepts.


```r
suppressPackageStartupMessages(library(tidyverse))
knitr::opts_chunk$set(fig.width=6, fig.height=3, fig.align="center")
```


# Maximal Margin Classifier

<img src="cm09-svm_files/figure-html/unnamed-chunk-2-1.png" style="display: block; margin: auto;" />


