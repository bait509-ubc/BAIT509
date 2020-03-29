
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

# subsetting for loess
r <- 0.1
dat_loess <- subset(dat, d<r)
dat_loess

y_predict_loess <- mean(dat_knn_final$y)
y_predict_loess


#############
#regression curve
###############

library(tidyverse)
xgrid <- seq(-5, 4, length.out=1000)
xgrid
kNN_estimates <- map_dbl(xgrid, function(x){
  dat$d <- abs(dat$x-x)
  dat_knn <- arrange(dat, d)
  dat_knn_final <- dat_knn[1:5,]
  mean(dat_knn_final$y)
})
kNN_estimates



loess_estimates <- map_dbl(xgrid, function(x){
  dat$d <- abs(dat$x-x)
  dat_loess <- subset(dat, d<1)
  mean(dat_loess$y)
})

est <- tibble(x=xgrid, kNN=kNN_estimates, loess=loess_estimates) %>% 
  gather(key="method", value="estimate", kNN, loess)
ggplot() +
  geom_point(data=dat, mapping=aes(x,y), colour=4) +
  geom_line(data=est, mapping=aes(x,estimate, group=method, colour=7)) +
  theme_bw()

