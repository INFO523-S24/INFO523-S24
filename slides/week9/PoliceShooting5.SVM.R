################################ SVM ########################

# Needed Packages
install.packages("tidyverse")
install.packages("kernlab")
install.packages("e1071")
install.packages("ISLR")
install.packages("RColorBrewer")
install.packages("performanceEstimation")

library(tidyverse)    # data manipulation and visualization
library(kernlab)      # SVM 
library(e1071)        # SVM, one-against-one for multi-class classification.
#For multiclass-classification with k levels, k>2, libsvm uses 
#the 'one-against-one'-approach, in which k(k-1)/2 binary classifiers 
#are trained; the appropriate class is found by a voting scheme.
library(RColorBrewer) # customized coloring of plots
library(performanceEstimation)

setwd("C:/Users/hongcui/Documents/courses/INFO523 DataMining/INFO523_Cui/lecture-R-case/PoliceShooting");
load("shooting.RData")

#use shooting directly: factor, num, logi, date: 
#shooting <- shooting[, c(-1)] #remove id only 0.61-1
#shooting <- shooting[, c(-1, -2, -3)] #removed id,name,date: 0.61-0.81
shooting <- shooting[, c(-1, -2, -3, -5, -6, -18, -21, -22)] 
#removed id,name,date, armed, age, vpRatio, year, month: 0.60-0.80
#shooting <- shooting[, c(6, 8, 17, 20)] #keep only age, race, popLevel, povLevel 0.51-0.52

shooting <- na.omit(shooting) #default, this svm does not handle missing values, so remove any row with a NA: 
shooting <- as.data.frame(unclass(shooting), stringsAsFactors = TRUE) #need to convert characters to factors, the dataset will contain factor and numerical types (factor, number, logic)

# set seed for pseudorandom number generator, run this for each test
set.seed(1)
samples <- sample(1:nrow(shooting), 0.8*nrow(shooting))
train <- shooting[samples, ]
test <- shooting[-samples,]

# construct a model
#one-hot encoding for factor data, done automatically by e1071
model <- svm(race~., data = train, kernel = "linear", type="C-classification", cost=1, na.action = na.omit)

score <- function(model, test){
    pred <- predict(model, test, na.action=na.omit)
    t <- table(pred, test$race)
    return (score = sum(diag(t))/sum(t))
}

summary(model)
score(model, test) #0.60
score(model, train) #greater the cost, more fit the model is to the train (low bias, high variance -- risk overfitting)


##########################################  other hyperparameters
set.seed(1)
load("shooting.RData")
shooting <- shooting[, c(-1, -2, -3, -5, -6, -18, -21, -22)] 
#removed id,name,date, armed, age, vpRatio, year, month: 0.60-0.80
shooting <- na.omit(shooting) #default, this svm does not handle missing values, so remove any row with a NA: 
shooting <- as.data.frame(unclass(shooting), stringsAsFactors = TRUE) #need to convert characters to factors, the dataset will contain facotr, numerical types (factor, number, logic)

#kernels
#run 5-fold cross validation
model <- svm(race~., data = shooting, kernel = "linear", type="C-classification", na.action = na.omit, cross=5)
model$accuracies

set.seed(1)
radial <- svm(race~., data = shooting, kernel = "radial",type="C-classification", gamma = 0.1, cost = 1, cross=5)
radial$accuracies

set.seed(1)
poly <- svm(race~., data = shooting, kernel = "polynomial",type="C-classification", gamma = 0.1,  degree = 3, cost = 1, cross=5)
poly$accuracies

set.seed(1)
sig <- svm(race~., data = shooting, kernel = "sigmoid",type="C-classification", gamma = 0.1, cost = 1, cross=5)
sig$accuracies

#sigmoid seems to be the worst.

#class weight
set.seed(1)
wts <- nrow(shooting)/table(shooting$race)
#model <- svm(race~., data = shooting, kernel = "linear", type="C-classification", na.action = na.omit, class.weights=wts,  cross=5)
model <- svm(race~., data = shooting, kernel = "linear", type="C-classification", na.action = na.omit, class.weights=1/wts,  cross=5)
model$accuracies 
#didn't help


########tuning: estimate performances using different combination of parameters:grid search
set.seed(1)
#takes time,skip
tune.out <- tune(svm, race~., data = shooting, kernel = "polynomial", type="C-classification", na.action=na.omit,
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5), 
                               gamma = c(0.001, 0.01, 0.1, 1, 5), 
                               degree = c(2, 3, 4)))
tune.out$best.parameters
tune.out$best.performance #error rate, the lower the better

set.seed(1)

#CHECK
#default misclassification error is used for categorical predictions and the mean squared error for numeric predictions.
tune.out <- tune(svm, race~., data = shooting, kernel = "linear", type="C-classification", na.action=na.omit,
                 ranges = list(cost = c(0.001, 0.01)), 
                 tunecontrol = tune.control(sampling = "boot", nboot=10, boot.size=8/10))
tune.out
tune.out$best.parameters
# extract the best model
(bestmod <- tune.out$best.model)

#performanceEstimation in DMwR. This function is buggy, does not work in all situations. 
set.seed(1)
res <- performanceEstimation(
  c(PredTask(race ~ .,shooting)),
  c(workflowVariants(learner="svm",
                     learner.pars=list(cost=c(0.01, 1), kernel=c("linear", "radial")))),
  EstimationTask(metrics="acc", method=CV(nReps=1, nFolds=3))
)
summary(res)
res$shooting.race


############################### Use a small dataset to examine the model ###############################
data(iris)

## classification mode
# default with factor response:
(model <- svm(Species ~ ., data = iris))

# alternatively the traditional interface:
x <- subset(iris, select = -Species)
y <- iris$Species
(model <- svm(x, y)) 

print(model)
summary(model)
model$SV #the resulting support vectors (possibly scaled): 51 vectors
model$index
model$coefs #a_t_j * y_t_j in the decision function (see "Characteristics of the Solution" in lecture slides) 
model$rho # 'b' in the decision function (see "Characteristics of the Solution" in lecture slides) 
model$nSV #total number of support vectors (sum of SVs for all three classes = 51)


# test with train data
pred <- predict(model, x)
# (same as:)
pred <- fitted(model)

# Check accuracy:
table(pred, y)

# compute decision values to understand why svm classify instances the way it did
#
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")
#columns: one vs. one classifiers
#values: if value >0, label=first class, if <0, label=second class
#Final class: votes for the classes
pred
#check cases 50 and 51. 

# visualize (classes by color, SV by crosses):
plot(cmdscale(dist(iris[,-5])),
     col = as.integer(iris[,5]),
     pch = c("o","+")[1:150 %in% model$index + 1])


#could also ask svm to return the probabilities of an observation belonging to the classes
model <- svm(Species ~ ., data = iris, probability=TRUE)
pred <- predict(model, iris, probability=TRUE) #must use Prob = TRUE for both svm and predict functions.
head(attr(pred, "probabilities"))


########################test out effects of cost and gamma on a two class problem
load("shooting.RData")
shooting <- shooting[, c(-1)] 
shooting <- na.omit(shooting) #default, this svm does not handle missing values, so remove any row with a NA: 
shooting <- as.data.frame(unclass(shooting), stringsAsFactors = TRUE) 
shooting <- droplevels(shooting[shooting$race %in% c("B", "W"),])
score <- function(model, test){
  pred <- predict(model, test, na.action=na.omit)
  t <- table(pred, test$race)
  return (score = sum(diag(t))/sum(t))
}

set.seed(1)
samples <- sample(1:nrow(shooting), 0.8*nrow(shooting))
train <- shooting[samples, ]
test <- shooting[-samples,]

#c: 1, g: 0.1  => train 0.87  test 0.75
#1, 1  => train 1 test 0.69 [overfit]
#10, 0.1 => 1 0.72 [overfit]
#0, 0.01 => 0.65  0.69 
model <- svm(race~., data = train, kernel = "radial", type="C-classification", cost=1, gamma=1, na.action = na.omit)

score(model, train) #, when gamma=10000, train performance = 1
score(model, test) #, when gamma=10000, train performance = 0.68

#smaller gamma: decision boundary not very curvy or pointy,  higher bias and lower variance, risk underfitting
#higher gamma: lower bias and high variance, risk overfitting

#smaller C: allow more errors. high bias, low variance, underfitting
#larger C: allow few errors. low bias, high variance, overfitting

#bias: model not flexible enough and unable to fit the data. underfitting
#variance: model fit training too well, overfitting (resulting in large variance in performance on different datasets)

  