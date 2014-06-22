# Machine Learning Coursera Project

### Libraries

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 2.15.3
## Warning: package 'ggplot2' was built under R version 2.15.3
```

```r
library(plyr)
```

```
## Warning: package 'plyr' was built under R version 2.15.3
```


First, let's set a seed for pseudo-random numbers in order to do all results reproducible

```r
set.seed(280484)
```


### Read the data from the csv files

```r
train <- read.csv("pml-training.csv", na.strings = c("NA", ""))
test <- read.csv("pml-testing.csv")
```


### Data cleansing and number of regressors reduction
There are some variables that have only NAs values.
There are other variables that are indicators and IDs, not regressors (as user name, timestamp, etc.)
We eliminate both type of variables, reducing the dataset number of "columns"

```r
NAs <- apply(train, 2, function(x) {
    sum(is.na(x))
})
clean_train <- train[, which(NAs == 0)]
removeIndex <- grep("num_window|timestamp|X|user_name|new_window", names(clean_train))
clean_train2 <- clean_train[, -removeIndex]
```


### Train - Test within "training" data

For comparing different methods within the train data, we create a separate "validation" set.
Validation test is not going to be used for training purposes, but for testing methods.

```r
inTrain <- createDataPartition(y = clean_train2$classe, p = 0.6, list = FALSE)
training <- clean_train2[inTrain, ]
validation <- clean_train2[-inTrain, ]
```


### Fit the model using Bagging Trees

```r
modFit <- train(classe ~ ., data = training, method = "treebag", trControl = trainControl(method = "cv", 
    number = 4))
```


### Out-of-sample accuracy
Looking at the result of the crossvalidaton done by the model-fitting process we can get the accuracy. For the out of sample error, we will look at the results of applying the model to the validation set.

```r
predictions_validation <- predict(modFit, validation)
confusionMatrix(predictions_validation, validation$classe)
```

```
## Warning: package 'e1071' was built under R version 2.15.3
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2212   28    4    2    1
##          B    7 1457   19    7    6
##          C    7   22 1326   28   12
##          D    0    7   18 1244   13
##          E    6    4    1    5 1410
## 
## Overall Statistics
##                                         
##                Accuracy : 0.975         
##                  95% CI : (0.971, 0.978)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.968         
##  Mcnemar's Test P-Value : 0.000143      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.991    0.960    0.969    0.967    0.978
## Specificity             0.994    0.994    0.989    0.994    0.998
## Pos Pred Value          0.984    0.974    0.951    0.970    0.989
## Neg Pred Value          0.996    0.990    0.993    0.994    0.995
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.282    0.186    0.169    0.159    0.180
## Detection Prevalence    0.286    0.191    0.178    0.163    0.182
## Balanced Accuracy       0.992    0.977    0.979    0.981    0.988
```

Accuracy in validation test is 97%-98%, so our out of sample error would we in the interval of 2%-3%
