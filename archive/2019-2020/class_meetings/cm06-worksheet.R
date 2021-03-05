## Fitting a model function in R: using regression tree
suppressPackageStartupMessages(library(tree))
suppressPackageStartupMessages(library(tidyverse))
fit <- tree(Sepal.Width ~ Petal.Width, data=iris)
summary(fit)
xgrid <- seq(min(iris$Petal.Width), max(iris$Petal.Width), length.out=1000)
yhat <- predict(fit, newdata = data.frame(Petal.Width = xgrid))
qplot(xgrid, yhat, geom = "line") +
	geom_point(
		data = iris, 
		mapping = aes(
			x = Petal.Width,
			y = Sepal.Width
		)
	)

