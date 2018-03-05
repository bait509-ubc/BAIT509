library(knitr)
library(tidyverse)

set.seed(87)
dat <- tibble(x = c(rnorm(100), rnorm(100)+5)-3,
              y = sin(x^2/5)/x + rnorm(200)/10 + exp(1))
kable(head(dat))

dat$d <- abs(dat$x-0)
dat_knn <- arrange(dat, dat$d)
dat_knn
dat_knn_final <- dat_knn[1:5,]
dat_knn_final

y_predict <- mean(dat_knn_final$y)
y_predict
