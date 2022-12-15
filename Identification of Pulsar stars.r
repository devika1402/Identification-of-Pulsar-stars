#   INSTALLING NECESSARY PACKAGES

# install.packages("caret", dependencies = TRUE)
# install.packages("skimr")
# install.packages("corrplot") 
# install.packages("mice")
# install.packages('tidyverse')
# install.packages('ggplot2')
# install.packages('neuralnet')

#   INCLUDING LIBRARIES
library(caret)
library(mice)
library(skimr)
library(corrplot)
library(tidyverse)
library(ggplot2)
library(neuralnet)
library(nnet)

# LOADING THE DATASET
df_train = read.csv('/Users/devikarajasekar/Documents/SCHOOL/Identification of Pulsar stars/Project/pulsar_data_train.csv')
df_test = NULL


# CLEANING and TRANSFORMING the dataset

  #Before imputing NULL values
  str(df_train)
  head(df_train)
  summary(df_train)
  
  # Generating a summary of the dataset with histograms
  skimmed <- skim(df_train) 
  skimmed
  
  sum(is.na.data.frame(df_train))

  X <- df_train[,1:8]
  y <- df_train[,9]
  
  md.pattern(X) #returns a tabular form of missing value present in each variable in a data set
  
  # IMPUTING THE DATASET
  imputed_Data <- mice(X, m=5, maxit = 50, method = 'pmm', seed = 500)
  summary(imputed_Data)
  df_train <- complete(imputed_Data,2)
  
  # CHECKING FOR NA VALUES AFTER IMPUTATION
  sum(is.na.data.frame(df_train))
  
  # Append the Y variable
  df_train$target_class <- y
  df_train
  
  # Generating a summary of the dataset with histograms
  skimmed <- skim(df_train)
  skimmed
  
  X = df_train[,1:8]

# With the missing values handled, our training dataset is now ready to undergo variable transformations if required. 
  # caret library provides the following data transformations:

#  range: Normalize values so it ranges between 0 and 1
#  center: Subtract Mean
#  scale: Divide by standard deviation
#  BoxCox: Remove skewness leading to normality. Values must be > 0
#  YeoJohnson: Like BoxCox, but works for negative values.
#  expoTrans: Exponential transformation, works for negative values.
#  pca: Replace with principal components
#  ica: Replace with independent components
#  spatialSign: Project the data to a unit circle

# For our problem, let’s convert all the numeric variables to range between 0 and 1, by setting method=range in preProcess().
  
# TRANSFORMING ATTRIBUTES
preProcess_range_model <- preProcess(df_train, method='range')
df_train <- predict(preProcess_range_model, newdata = df_train)

apply(df_train[, 1:8], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

# APPENDING THE TARGET CLASS
df_train$target_class <- y
head(df_train)

# ANALYSIS OF THE DATASET AFTER DATA TRANSFORMATION 

#1) Dimension of the dataset and column names
dim(df_train)
colnames(df_train)
sapply(df_train, class)
head(df_train)

#2) Summarize the class distribution
percentage <- prop.table(table(df_train$target_class)) * 100
cbind(freq=table(df_train$target_class), percentage=percentage)

#3) Correlation between different attributes using Heatmap
corrplot::corrplot(cor(df_train))


#4) Finding the important variables:

# Now that the preprocessing is complete, let’s visually examine how the predictors influence the Y (Purchase). 
# In this problem, the X variables are numeric whereas the Y is categorical. So how to gauge if a given X is an 
# important predictor of Y? A simple common sense approach is, if you group the X variable by the categories of 
# Y, a significant mean shift amongst the X’s groups is a strong indicator (if not the only indicator) that X 
# will have a significant role to help predict Y. 
# It is possible to watch this shift visually using box plots and density plots. 
# In fact, caret’s featurePlot() function makes it so convenient by simply setting the X and Y parameters and plot='box'.

df_train$target_class <- as.factor(df_train$target_class)
featurePlot(x = df_train[, 1:8], 
            y = df_train$target_class, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

featurePlot(x = df_train[, 1:8], 
            y = df_train$target_class, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
  
      
#5) Looking at the available ML models algorithms in caret

modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames

# Selecting three models to implement: MARS, KNN from caret and Neural Networks

#6) SPLITTING DATASET INTRO TRAIN AND TEST DATA
set.seed(100) 
indexes <- createDataPartition(as.factor(df_train$target_class), p = 0.8, list = F)
train <- df_train[indexes, ]
test <- df_train[-indexes, ]


#8) BUILDING MODELS

# I. Train a Multivariate Adaptive Regression Splines (MARS) model
  model_mars = train(target_class ~ ., data=train, method='earth')
  fitted <- predict(model_mars)
  model_mars
  
  plot(model_mars, main="Model Accuracies with MARS")
  
  varimp_mars <- varImp(model_mars)
  plot(varimp_mars, main="Variable Importance with MARS")
  
  # Predict on test Data
  predicted_mars <- predict(model_mars, test)
  head(predicted_mars)
  
  # Summarize results
  confusionMatrix(predicted_mars, test[,9])

# II. KNN (K-NEAREST NEIGHBOURS)
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  set.seed(100)
  knn_fit <- train(target_class ~., data = train, method = "knn",
                   trControl=trctrl,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)
  knn_fit
  
  # Predict on test Data
  predicted_knn <- predict(knn_fit, test)
  head(predicted_knn)
  
  plot(knn_fit, main="Model Accuracies with KNN")
  
  varimp_KNN <- varImp(knn_fit)
  plot(varimp_KNN, main="Variable Importance with KNN")
  
  # Summarize results
  confusionMatrix(predicted_knn, test[,9])

# III. Neural Network
  train_params <- trainControl(method = "repeatedcv", number = 10, repeats=3)
  nnet_model <- train(train[,-9], as.factor(train$target_class),
                      method = "nnet",trControl= train_params,
                      preProcess=c("scale","center"),
                      na.action = na.omit)

  #Predictions on the test set
  nnet_predictions_test <-predict(nnet_model, test)

  # Confusion matrix on test set
  table(test$target_class, nnet_predictions_test)
  # Accuracy of testing data:
  (2261+194)/nrow(test) #accuracy = (TP+TN)/(TP+TN+FP+FN)
  
  plot(nnet_predictions_test, main="Model Accuracies with Neural Networks")
  
  varimp_NN <- varImp(nnet_model)
  plot(varimp_NN, main="Variable Importance with Neural Networks")
  
  # Summarize results
  confusionMatrix(nnet_predictions_test, as.factor(test[,9]))