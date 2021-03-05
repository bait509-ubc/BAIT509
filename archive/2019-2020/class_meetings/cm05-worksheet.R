# Fitting a CART in R
library(tidyverse) # library(readr)
library(tree)
dat <- read_csv("https://raw.githubusercontent.com/vincenzocoia/BAIT509/master/assessments/assignment1/data/titanic.csv")
str(dat)
fit <- tree(Survived ~ Fare, data = dat)
predict(fit) # Vector of predictions on training data
plot(fit, type="uniform")
plot(fit)
class(fit)  # Calls plot.tree()
?plot.tree
?plot.lm
