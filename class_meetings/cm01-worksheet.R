library(tidyverse)
str(iris)
mean(iris$Sepal.Width)
median(iris$Sepal.Width)
mean((3.057333 - iris$Sepal.Width)^2)

fit <- lm(Sepal.Width ~ Petal.Width + Species, data=iris)
mean((predict(fit) - iris$Sepal.Width)^2)
