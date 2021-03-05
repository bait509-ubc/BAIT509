library(randomForest)
library(tidyverse)
library(tree)
fit <- randomForest(mpg ~ ., data=mtcars, ntree=1000)
## Out of bag predictions:
yhat_oob <- predict(fit)
## Total aggregate predictions:
yhat_agg <- predict(fit, newdata = mtcars)
## Out of sample error:
mean((mtcars$mpg - yhat_oob)^2)
plot(fit)

## Regression tree:
fit2 <- tree(mpg ~ ., data=mtcars)
cv <- cv.tree(fit2)
qplot(cv$size, cv$dev)
