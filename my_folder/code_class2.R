# In -Class 1

install.packages(c("ISLR","tidyverse","knitr"))
library(ISLR)
library(tidyverse)
library(knitr)

genreg <- function(n){
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  eps <- rnorm(n)
  y <- 5-x1+2*x2+eps
  tibble(x1=x1, x2=x2, y=y)
}

mydata <- genreg(100)
mydata

dat <- mutate(mydata,
              yhat <- 0,
              yhatx <- 5-x1,
              yhatx2 <- 5-2*x2,
              yhatx12 <- 5-x1+2*x2)

dat

#compute errror 
mse <- mean((dat$yhat - dat$y)^2)

gencla <- function(n) {
  x <- rnorm(n) 
  pB <- 0.8/(1+exp(-x))
  y <- map_chr(pB, function(x) 
    sample(LETTERS[1:3], size=1, replace=TRUE,
           prob=c(0.2, x, 1-x)))
  tibble(x=x, y=y)
}

cladata <- gencla(100)
head(cladata)

pB1 <- 0.8/(1+exp(-1))
pB1
x_1 =0

# if X = 1
predat <- mutate(cladata,
                 x_1 <- 1,
                 pB <- 0.8/(1+exp(-(x_1))),
                 y <- map_chr(pB, function(x_1) 
                   sample(LETTERS[1:3], size=1, replace=TRUE,
                          prob=c(0.2, x_1, 1-x_1-0.2))))
head(predat)

# if X = -2

pB2 <- 0.8/(1+exp(-2))
pB2
x_1 =0
predat2 <- mutate(cladata,
                 x_1 <- -2,
                 pB <- 0.8/(1+exp(-x_1)),
                 y <- map_chr(pB, function(x_1) 
                   sample(LETTERS[1:3], size=1, replace=TRUE,
                          prob=c(0.2, x_1, 1-x_1-0.2))))
head(predat2)

