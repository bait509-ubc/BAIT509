## Breast cancer data

This breast cancer domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia.  Thanks go to M. Zwitter and M. Soklic for providing the data.

`breast_cancer.csv` file obtained by running the following R code:

```
library(tidyverse)
"http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data" %>% 
	read_csv(col_names = FALSE) %>% 
	setNames(c("Class", "age", "menopause", "tumor_size", "inv_nodes", "node_caps", 
			   "deg_malig", "breast", "breast_quad", "irradiat")) %>% 
	write_csv("breast_cancer.csv")
```