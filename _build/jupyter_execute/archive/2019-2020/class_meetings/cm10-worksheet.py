# SVM's in python: worksheet

BAIT 509 Class Meeting 10

Let's work with the breast cancer dataset:

from sklearn import datasets
dat = datasets.load_breast_cancer()
y = dat.target
X = dat.data

Here are the predictors of breast cancer:

dat.feature_names

And their dimension:

X.shape

## Fitting an SVM model

#### 0\. Scale the data 

([this page](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) on sklearn's documentation is useful).
 
- Initiate transformer with `StandardScaler()` from `sklearn.preprocessing`.
- Fit the transformer using the `.fit()` method.
- Use the scaler to transform `X`, with the `.transform()` method.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
Xscale = scaler.transform(X)

#### 1\. Fit a SVC (i.e., linear SVM)

Fitting an SVM can be done using `sklearn.svm`'s `SVC` module. 

from sklearn import svm

Do the fitting here:

my_svc = svm.SVC(C=1, kernel="linear")
my_svc.fit(Xscale, y)

What's the accuracy? Try changing the `C` parameter.

sum(my_svc.predict(Xscale) == y) / len(y)

#### 2\. Fit a radial-basis SVM

Try again, this time with radial SVM. What's the accuracy? Try changing the parameters. 

my_svc = svm.SVC(C=1, kernel="rbf", gamma=100)
my_svc.fit(Xscale, y)
sum(my_svc.predict(Xscale) == y) / len(y)

## Cross validation 

Evaluate generalization over a grid of parameters. Use the linear kernel. Here is the module we'll import:

from sklearn.model_selection import GridSearchCV

Define a grid of `C` (hyperparameter/tuning parameter) values:

C = [1, 10, 20]

Initiate the model fit as usual; ignore specification of `C`.

model = svm.SVC(kernel="linear")

From the initiated model, initiate cross validation using the `GridSearchCV()` function, like so:

model_cv = GridSearchCV(model, param_grid={"C":C}, cv=10)

Now, "fit" the cross validation with the `.fit()` method (as if you're fitting a model). Warning: this will be slow if you did not scale the data!

model_cv.fit(Xscale, y)

You can obtain the best parameters and best scores by appending `.best_params_` and `.best_score_`:

model_cv.best_params_
#model_cv.best_score_

You can obtain info about all folds by appending `.cv_results_`. What are the test scores of the fourth fold for each value of `C`?

model_cv.cv_results_["split4_test_score"]

