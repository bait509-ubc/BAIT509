library(ggplot2)
library(Lahman)
ggplot(Teams, aes(x=R)) + geom_histogram()
ggplot(Teams, aes(x=R)) + geom_density()

## Probabilistic forecast with 2 predictors
## kNN with k=30
meanH <- mean(Teams$H)
sdH <- sd(Teams$H)
meanW <- mean(Teams$W)
sdW <- sd(Teams$W)
Teams$Hscale <- (Teams$H - meanH) / sdH
Teams$Wscale <- (Teams$W - meanW) / sdW
H0scale <- (1500 - meanH) / sdH
W0scale <- (70 - meanW) / sdW
Teams$dist <- sqrt((Teams$Hscale - H0scale)^2 + 
				   (Teams$Wscale - W0scale)^2)
yrelevant <- dplyr::arrange(Teams, dist)[1:30, "R"]
mean(yrelevant)
qplot(yrelevant, geom="density")
