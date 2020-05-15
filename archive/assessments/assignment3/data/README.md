## Info about `auto_data.csv`

The `auto_data.csv` dataset is from the `ISLR` R package, under the `Auto` object. The according to the documentation of the `Auto` object, the source of the data is as follows:

> This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. The dataset was used in the 1983 American Statistical Association Exposition.

### Making the data

The data in `auto_data.csv` can be made by running the following code in R:

```
library(dplyr)
library(ISLR)
mutate(Auto, mileage = as.integer(mpg>median(mpg)))
```

R version 3.4.2 (2017-09-28): Short Summer

`ISLR` version 1.2

`dplyr` version 0.7.4

### Variables in the data

The `mileage` column contains 0 for "low mileage", and 1 for "high mileage". 

For the remaining variables, the documentation of the `Auto` object says the following:

- `mpg`: miles per gallon
- `cylinders`: Number of cylinders between 4 and 8
- `displacement` Engine displacement (cu. inches)
- `horsepower`: Engine horsepower
- `weight`: Vehicle weight (lbs.)
- `acceleration`: Time to accelerate from 0 to 60 mph (sec.)
- `year`: Model year (modulo 100)
- `origin`: Origin of car (1. American, 2. European, 3. Japanese)
- `name`: Vehicle name
