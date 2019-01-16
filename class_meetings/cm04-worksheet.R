## Example of cross validation in R
library(caret)
qplot(Petal.Width, Sepal.Width, data=iris)
ctrl <- trainControl(method="cv", number=5)
k <- data.frame(k = 1:5)
train(Sepal.Width ~ Petal.Width, 
	  data      = iris, 
	  trControl = ctrl, 
	  tunegrid  = k,
	  method    = "knn")
